"""Fetch the arena.ai text leaderboard and save it as a dated CSV snapshot.

The leaderboard page (https://arena.ai/leaderboard/text) is a Next.js app that
ships the full leaderboard as structured JSON inside its flight payload
(self.__next_f.push(...) script chunks). We parse that payload instead of the
rendered HTML table: it carries clean fields (modelDisplayName,
modelOrganization, license, full-precision rating) and the official data date
(voteCutoffISOString), and it is far more stable than the page markup.

arena's own price columns are kept as-is and OpenRouter's prices are joined
on alongside them (see fetch_openrouter_prices), because arena's numbers go
stale and disagree with providers' list prices for a sizeable minority of
models. Nothing is overwritten, so the switch stays auditable in the history.

Snapshots are named data/lmsys_snapshot_<voteCutoffDate>.csv. Every failure
path exits non-zero so the GitHub Actions run turns red instead of silently
producing nothing. That includes the OpenRouter join: a missed day is
recoverable, a snapshot with a silently missing price source is not.
"""

import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import requests

from fetch_openrouter_prices import (
    attach_openrouter_prices,
    build_price_index,
    fetch_openrouter_models,
    report,
)

# --- Configuration ---
LEADERBOARD_URL = "https://arena.ai/leaderboard/text"
DATA_DIR = "data"
FILENAME_TEMPLATE = os.path.join(DATA_DIR, "lmsys_snapshot_{}.csv")
# Refuse to save a snapshot with fewer models than this. Protects against
# partially rendered pages silently shrinking the history (the full board
# has 300+ models).
MIN_EXPECTED_MODELS = 100
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

FLIGHT_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]\)')


def decode_flight_payload(html: str) -> str:
    """Concatenate and unescape all Next.js flight payload chunks."""
    parts = []
    for chunk in FLIGHT_CHUNK_RE.findall(html):
        try:
            # The chunk is a JS string literal body; JSON string decoding
            # handles the same escape sequences.
            parts.append(json.loads('"' + chunk + '"'))
        except json.JSONDecodeError:
            continue
    return "".join(parts)


def extract_json_value(blob: str, start: int):
    """Parse the JSON array/object starting at blob[start] via bracket matching."""
    opener = blob[start]
    closer = {"[": "]", "{": "}"}[opener]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(blob)):
        ch = blob[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(blob[start : i + 1])
    raise ValueError("Unbalanced brackets while extracting JSON value from flight payload")


def extract_snapshot(html: str):
    """Extract the text leaderboard from page HTML.

    Returns (DataFrame with Model_Name/ELO_Score/Provider/License, cutoff_date).
    Raises ValueError if the expected payload structure is missing.
    """
    blob = decode_flight_payload(html)
    if not blob:
        raise ValueError("No Next.js flight payload found in page HTML")

    # Anchor on the text arena so we never grab another leaderboard's entries.
    anchor = blob.find('"arenaSlug":"text"')
    if anchor == -1:
        raise ValueError('Could not find "arenaSlug":"text" in flight payload')

    entries_key = blob.find('"entries":', anchor)
    if entries_key == -1:
        raise ValueError('Could not find "entries" after text arena anchor')
    entries = extract_json_value(blob, entries_key + len('"entries":'))
    if not entries:
        raise ValueError("Leaderboard entries array is empty")

    cutoff_match = re.search(r'"voteCutoffISOString":"(\d{4}-\d{2}-\d{2})', blob[anchor:])
    if not cutoff_match:
        raise ValueError("Could not find voteCutoffISOString in flight payload")
    cutoff_date = datetime.strptime(cutoff_match.group(1), "%Y-%m-%d").date()

    rows = []
    for e in entries:
        rows.append(
            {
                "Model_Name": e["modelDisplayName"],
                "ELO_Score": round(float(e["rating"]), 2),
                "Provider": e["modelOrganization"],
                "License": e.get("license") or "Unknown",
                "Votes": e.get("votes"),
                "Input_Price": e.get("inputPricePerMillion"),
                "Output_Price": e.get("outputPricePerMillion"),
                "Context_Length": e.get("contextLength"),
            }
        )
    df = pd.DataFrame(rows)

    incomplete = df[df["Model_Name"].isna() | df["Provider"].isna() | df["ELO_Score"].isna()]
    if not incomplete.empty:
        raise ValueError(f"{len(incomplete)} entries have missing critical fields:\n{incomplete}")

    return df, cutoff_date


def snapshots_differ(df_a: pd.DataFrame, df_b: pd.DataFrame) -> bool:
    if set(df_a.columns) != set(df_b.columns):
        return True
    cols = sorted(df_a.columns)
    sort_keys = ["Model_Name", "Provider", "ELO_Score"]
    a = df_a[cols].sort_values(sort_keys).reset_index(drop=True)
    b = df_b[cols].sort_values(sort_keys).reset_index(drop=True)
    return not a.equals(b)


def save_snapshot(df: pd.DataFrame, cutoff_date) -> str:
    """Write the snapshot CSV for cutoff_date. Returns a description of what happened."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = FILENAME_TEMPLATE.format(cutoff_date.isoformat())

    if os.path.exists(path):
        existing = pd.read_csv(path)
        if set(existing.columns) >= {"Model_Name", "ELO_Score", "Provider", "License"} and not snapshots_differ(existing, df):
            return f"unchanged: {path} already matches today's data"
        df.to_csv(path, index=False)
        return f"updated: {path} (content changed for {cutoff_date})"

    df.to_csv(path, index=False)
    return f"created: {path}"


def main():
    print(f"Fetching {LEADERBOARD_URL}")
    response = requests.get(LEADERBOARD_URL, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()

    df, cutoff_date = extract_snapshot(response.text)
    print(f"Extracted {len(df)} models, vote cutoff date {cutoff_date}")

    if len(df) < MIN_EXPECTED_MODELS:
        raise ValueError(
            f"Only {len(df)} models extracted (expected >= {MIN_EXPECTED_MODELS}); "
            "refusing to save a suspiciously small snapshot"
        )

    print("Fetching prices from OpenRouter")
    df = attach_openrouter_prices(df, build_price_index(fetch_openrouter_models()))
    print(report(df))

    result = save_snapshot(df, cutoff_date)
    print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

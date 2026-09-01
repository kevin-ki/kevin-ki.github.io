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

Snapshots are named data/lmsys_snapshot_<voteCutoffDate>.csv and are
append-only: see merge_into_snapshot for why a rerun may fill blanks and add
models but never rewrite a value already on record.

Every failure path exits non-zero so the GitHub Actions run turns red instead
of silently producing nothing. That includes the OpenRouter join: a missed day
is recoverable, a snapshot with a silently missing price source is not.
"""

import json
import math
import numbers
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


def is_blank(value) -> bool:
    """True when a cell holds nothing.

    An empty string counts as blank, not as a value. arena returns an empty
    organisation for some models, and pandas reads that back from CSV as NaN,
    so treating the two as different made every run "fill" the same 50 cells
    and commit the same file forever.
    """
    if pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def values_agree(recorded, incoming) -> bool:
    """True when a recorded cell and an incoming one mean the same thing.

    Numbers are compared with a tolerance. A price survives the CSV round trip
    as a very slightly different double, and reporting 1.027 against
    1.0270000000000001 as a conflict would bury the ones that matter.
    """
    if isinstance(recorded, numbers.Real) and isinstance(incoming, numbers.Real):
        return math.isclose(float(recorded), float(incoming), rel_tol=1e-9)
    return recorded == incoming


def merge_into_snapshot(existing: pd.DataFrame, fresh: pd.DataFrame):
    """Fold a fresh fetch into an existing snapshot without rewriting history.

    A snapshot records what arena published for one vote cutoff, and the cutoff
    stays put for days, so several daily runs land on the same file. arena's
    price field is not stable across those runs: the same model comes back at
    the list price on one fetch and at exactly half on the next. Two fetches of
    the 2026-08-27 cutoff disagreed on fifteen models, gemini-2.5-pro doubling
    while gpt-5.1 and gpt-5.2 halved, with ELO and votes identical throughout.
    Overwriting on every run therefore rewrote the record to whichever value
    the last fetch happened to catch.

    The merge is append-only at the cell level instead: a value already on
    record is never changed, a blank cell may be filled, and a model arena adds
    later joins as a new row. Models that disappear upstream stay. Conflicts
    are reported rather than applied, and reported rather than raised, because
    a flip-flopping upstream price is noise and not a failed run.

    Returns (merged frame, summary lines, changed).
    """
    columns = list(dict.fromkeys([*existing.columns, *fresh.columns]))
    existing = existing.reindex(columns=columns)
    fresh = fresh.reindex(columns=columns)

    on_record = {row["Model_Name"]: dict(row) for _, row in existing.iterrows()}
    filled, added, conflicts = 0, 0, []

    for _, row in fresh.iterrows():
        recorded = on_record.get(row["Model_Name"])
        if recorded is None:
            on_record[row["Model_Name"]] = dict(row)
            added += 1
            continue
        for column in columns:
            incoming = row[column]
            if is_blank(incoming):
                continue
            if is_blank(recorded[column]):
                recorded[column] = incoming
                filled += 1
            elif not values_agree(recorded[column], incoming):
                conflicts.append((row["Model_Name"], column, recorded[column], incoming))

    merged = pd.DataFrame(list(on_record.values()), columns=columns)
    merged = merged.sort_values("ELO_Score", ascending=False).reset_index(drop=True)

    summary = [f"merge: {added} models added, {filled} blank cells filled"]
    if conflicts:
        summary.append(
            f"  {len(conflicts)} upstream values differ from the record and were kept as recorded:"
        )
        for name, column, recorded_value, incoming in conflicts[:5]:
            summary.append(f"    {name} {column}: kept {recorded_value}, upstream now {incoming}")
        if len(conflicts) > 5:
            summary.append(f"    ... and {len(conflicts) - 5} more")
    return merged, summary, bool(added or filled)


def save_snapshot(df: pd.DataFrame, cutoff_date) -> str:
    """Write the snapshot CSV for cutoff_date. Returns a description of what happened."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = FILENAME_TEMPLATE.format(cutoff_date.isoformat())

    if not os.path.exists(path):
        df.sort_values("ELO_Score", ascending=False).to_csv(path, index=False)
        return f"created: {path} with {len(df)} models"

    merged, summary, changed = merge_into_snapshot(pd.read_csv(path), df)
    for line in summary:
        print(line)
    if not changed:
        return f"unchanged: {path} already holds everything this fetch added"
    merged.to_csv(path, index=False)
    return f"updated: {path} ({len(merged)} models on record for {cutoff_date})"


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

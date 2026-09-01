"""Join OpenRouter's published prices onto an arena.ai leaderboard snapshot.

arena.ai ships a price per model, but its numbers drift from what providers
actually charge. deepseek-v4-flash sat at a batch-tier price for weeks, and for
a recurring set of models arena reports exactly half the list price
(claude-sonnet-4-6 at $7.50 against a real $15, gemini-2.5-pro at $5.00 against
a real $10). OpenRouter publishes the standard per-token price it bills, so we
join it on where the model can be matched and keep arena's value as a fallback
and an audit trail. Both prices land in the snapshot; nothing is overwritten.

Matching is the hard part, because the two sources name models differently:

    claude-opus-4-6-high    -> claudeopus46  (anthropic/claude-opus-4.6)
    muse-spark-1.2 (xHigh)  -> musespark12   (meta/muse-spark-1.2)
    gemma-4-31b             -> gemma431b     (google/gemma-4-31b-it)

Only suffixes that cannot change the per-token price may be stripped, and the
rules below are deliberately strict, because a wrong match is worse than no
match: it silently replaces a correct price with another model's.

  * Reasoning effort ("-high", "(xHigh)", "-thinking") is safe to strip. Effort
    makes a model emit more tokens, it does not make a token cost more, so the
    base model's price is the correct price for the variant.
  * Tier words ("flash", "pro", "max", "mini") are never stripped. glm-5.3-flash
    is $0.25/M and glm-5.3 is $4.40/M, a factor of 17.
  * Dates and "-preview" are never stripped either. gpt-4-0125-preview is GPT-4
    Turbo at $30/M, while gpt-4 is $60/M; stripping the date maps the cheaper
    model onto the pricier one. Models needing that kind of match keep arena's
    price.
  * Batch endpoints (":batch") are excluded outright. They run anywhere from 4x
    cheaper (gemini-3.7-flash) to 2x more expensive (glm-5.3-flash) than the
    standard endpoint.

Run standalone to backfill an existing snapshot:

    python fetch_openrouter_prices.py data/lmsys_snapshot_2026-08-27.csv
"""

import re
import sys
from dataclasses import dataclass

import pandas as pd
import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 30
# OpenRouter quotes USD per token; the leaderboard works in USD per million.
PER_MILLION = 1_000_000
# Decimals kept on a per-million price. Multiplying a per-token price by a
# million leaves representation noise ($0.13199999999999998), which then reads
# as a changed price on the next run. Six decimals is finer than any provider
# publishes.
PRICE_PRECISION = 6

# Suffixes denoting reasoning effort, which never changes the per-token price.
# Ordered longest first so "xhigh" is not mistaken for "high".
EFFORT_SUFFIXES = (
    "nonthinking",
    "reasoning",
    "thinking",
    "minimal",
    "medium",
    "xhigh",
    "high",
    "low",
)
# Shortest canonical key we will strip down to, so "high" never eats a real name.
MIN_KEY_LENGTH = 4


@dataclass(frozen=True)
class OpenRouterPrice:
    model_id: str
    input_price: float
    output_price: float


def canonical_key(name: str) -> str:
    """Reduce a model name to a comparison key.

    Drops the provider prefix, unwraps parenthesised variant markers, and
    removes every separator, so that arena's "claude-opus-4-6" and
    OpenRouter's "anthropic/claude-opus-4.6" both become "claudeopus46".
    """
    key = name.lower()
    key = re.sub(r"^[^/]+/", "", key)
    key = re.sub(r"\(([^)]*)\)", r" \1 ", key)
    return re.sub(r"[^a-z0-9]+", "", key)


def strip_effort_suffix(key: str) -> str:
    """Remove trailing reasoning-effort markers from a canonical key."""
    changed = True
    while changed:
        changed = False
        for suffix in EFFORT_SUFFIXES:
            if key.endswith(suffix) and len(key) - len(suffix) >= MIN_KEY_LENGTH:
                key = key[: -len(suffix)]
                changed = True
    return key


def fetch_openrouter_models() -> list[dict]:
    """Fetch the OpenRouter model catalogue. Raises on any failure."""
    response = requests.get(OPENROUTER_MODELS_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    models = response.json().get("data")
    if not models:
        raise ValueError("OpenRouter returned no models")
    return models


def build_price_index(models: list[dict]) -> dict[str, OpenRouterPrice]:
    """Map canonical keys to prices, dropping anything ambiguous.

    Batch endpoints and deranked models (a leading "~") are skipped. The "-it"
    instruction-tuned marker gets an extra alias, since arena omits it. When two
    OpenRouter models collapse onto one key with different prices, the key is
    dropped rather than resolved arbitrarily.
    """
    candidates: dict[str, list[OpenRouterPrice]] = {}
    for model in models:
        model_id = model["id"]
        if ":" in model_id or model_id.startswith("~"):
            continue
        pricing = model.get("pricing") or {}
        try:
            entry = OpenRouterPrice(
                model_id=model_id,
                input_price=round(float(pricing["prompt"]) * PER_MILLION, PRICE_PRECISION),
                output_price=round(float(pricing["completion"]) * PER_MILLION, PRICE_PRECISION),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if entry.output_price <= 0:
            continue

        key = canonical_key(model_id)
        candidates.setdefault(key, []).append(entry)
        if key.endswith("it") and len(key) - 2 >= MIN_KEY_LENGTH:
            candidates.setdefault(key[:-2], []).append(entry)

    index: dict[str, OpenRouterPrice] = {}
    for key, entries in candidates.items():
        prices = {(e.input_price, e.output_price) for e in entries}
        if len(prices) == 1:
            index[key] = entries[0]
    return index


def lookup(index: dict[str, OpenRouterPrice], model_name: str):
    """Return (price, match_kind) for a model name, or (None, "none")."""
    key = canonical_key(model_name)
    if key in index:
        return index[key], "exact"
    without_effort = strip_effort_suffix(key)
    if without_effort != key and without_effort in index:
        return index[without_effort], "effort"
    return None, "none"


def attach_openrouter_prices(df: pd.DataFrame, index: dict[str, OpenRouterPrice]) -> pd.DataFrame:
    """Add OR_Input_Price, OR_Output_Price, OR_Model_Id and Price_Match columns."""
    or_input, or_output, or_id, match_kind = [], [], [], []
    for name in df["Model_Name"]:
        price, kind = lookup(index, name)
        or_input.append(price.input_price if price else None)
        or_output.append(price.output_price if price else None)
        or_id.append(price.model_id if price else None)
        match_kind.append(kind)

    df = df.copy()
    df["OR_Input_Price"] = or_input
    df["OR_Output_Price"] = or_output
    df["OR_Model_Id"] = or_id
    df["Price_Match"] = match_kind
    return df


def report(df: pd.DataFrame) -> str:
    """Summarise coverage and where the two sources disagree."""
    matched = df[df["OR_Output_Price"].notna()]
    priced = df[df["Output_Price"].notna() & (df["Output_Price"] > 0)]
    both = matched[matched["Output_Price"].notna() & (matched["Output_Price"] > 0)]
    disagree = both[
        (both["OR_Output_Price"] - both["Output_Price"]).abs() / both["Output_Price"] > 0.25
    ]
    lines = [
        f"OpenRouter prices matched for {len(matched)} of {len(df)} models "
        f"({len(priced)} carry an arena price)",
        f"  exact {int((df['Price_Match'] == 'exact').sum())}, "
        f"effort {int((df['Price_Match'] == 'effort').sum())}, "
        f"unmatched {int((df['Price_Match'] == 'none').sum())}",
        f"  {len(disagree)} of {len(both)} matched models differ from arena by more than 25%",
    ]
    for _, row in disagree.nlargest(8, "OR_Output_Price").iterrows():
        lines.append(
            f"    {row['Model_Name']:<32} arena={row['Output_Price']:>8.2f} "
            f"openrouter={row['OR_Output_Price']:>8.3f}  {row['OR_Model_Id']}"
        )
    return "\n".join(lines)


def main(path: str) -> None:
    print(f"Fetching {OPENROUTER_MODELS_URL}")
    index = build_price_index(fetch_openrouter_models())
    print(f"Built price index with {len(index)} unambiguous keys")

    df = attach_openrouter_prices(pd.read_csv(path), index)
    print(report(df))
    df.to_csv(path, index=False)
    print(f"updated: {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <snapshot.csv>", file=sys.stderr)
        sys.exit(2)
    try:
        main(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

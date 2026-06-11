"""One-off backfill of missing leaderboard snapshots from the Wayback Machine.

The daily scraper silently produced nothing between 2026-02-06 and 2026-06-11
(arena.ai redesign). archive.org has near-daily captures of the leaderboard in
that window, and each capture contains the same Next.js flight payload the
live page ships, so we can rebuild full-fidelity snapshots.

Usage:
    python backfill_wayback.py [--from 20260202] [--to 20260611]

Saves one CSV per distinct voteCutoff date via the same save logic as the
daily scraper (existing identical files are left alone; files with different
content, like the corrupted 2026-02-06 snapshot, are overwritten).
"""

import argparse
import sys
import time
from datetime import date

import requests

from fetch_daily_snapshot import (
    MIN_EXPECTED_MODELS,
    REQUEST_HEADERS,
    extract_snapshot,
    save_snapshot,
)

CDX_API = "http://web.archive.org/cdx/search/cdx"
TARGET_URL = "arena.ai/leaderboard/text"
# id_ flag returns the original archived bytes without Wayback's HTML rewriting
CAPTURE_URL_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/https://arena.ai/leaderboard/text"
FETCH_DELAY_SECONDS = 4
MAX_RETRIES = 3


def list_captures(from_date: str, to_date: str) -> list[str]:
    """Return one capture timestamp per day from the Wayback CDX API."""
    params = {
        "url": TARGET_URL,
        "from": from_date,
        "to": to_date,
        "output": "json",
        "fl": "timestamp",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # one capture per day
    }
    response = requests.get(CDX_API, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise ValueError("CDX API returned no captures for the requested window")
    return [row[0] for row in rows[1:]]  # skip header row


def fetch_capture(timestamp: str) -> str:
    url = CAPTURE_URL_TEMPLATE.format(timestamp=timestamp)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=120)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            last_error = e
            wait = 5 * attempt
            print(f"  attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch capture {timestamp} after {MAX_RETRIES} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_date", default="20260202")
    parser.add_argument("--to", dest="to_date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    timestamps = list_captures(args.from_date, args.to_date)
    print(f"Found {len(timestamps)} daily captures between {args.from_date} and {args.to_date}")

    saved, unchanged, failures = [], [], []
    seen_cutoff_dates = set()

    for i, ts in enumerate(timestamps, 1):
        print(f"[{i}/{len(timestamps)}] capture {ts}")
        try:
            html = fetch_capture(ts)
            df, cutoff_date = extract_snapshot(html)
            if len(df) < MIN_EXPECTED_MODELS:
                raise ValueError(f"only {len(df)} models extracted")
            if cutoff_date in seen_cutoff_dates:
                print(f"  cutoff {cutoff_date} already handled this run; skipping")
                continue
            seen_cutoff_dates.add(cutoff_date)
            result = save_snapshot(df, cutoff_date)
            print(f"  {len(df)} models, cutoff {cutoff_date} -> {result}")
            if result.startswith("unchanged"):
                unchanged.append(cutoff_date)
            else:
                saved.append(cutoff_date)
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append((ts, str(e)))
        time.sleep(FETCH_DELAY_SECONDS)

    print("\n=== Backfill summary ===")
    print(f"Snapshots written: {len(saved)}")
    print(f"Already up to date: {len(unchanged)}")
    print(f"Failed captures: {len(failures)}")
    for ts, err in failures:
        print(f"  {ts}: {err}")

    if failures and not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()

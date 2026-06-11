# Fix silent pipeline failure + backfill + Next.js redesign (2026-06-11)

Context: arena.ai redesign (formerly lmarena.ai) broke the scraper on 2026-02-06.
Every error path printed and exited 0, so Actions stayed green for 4 months.
Scope grew: user requested a full Next.js redesign in the zued brand style with
provider logos and an ELO timeline as hero chart.

## Plan

- [x] Rewrite `fetch_daily_snapshot.py`
  - Parse Next.js flight payload (`self.__next_f.push`) instead of HTML table
  - Snapshot date from `voteCutoffISOString`, fail loud, >=100 models floor
  - Verified live: 366 models, clean names
- [x] Write `backfill_wayback.py` (Wayback CDX, one CSV per cutoff date)
- [ ] Backfill running (at ~2026-04-28 of ~125 captures)
- [x] Repair glued provider prefixes in 37 old snapshots (2230 rows)
  - Heuristic: strip provider prefix only when followed by a letter
  - First attempt over-stripped legit names (DeepSeek-V3 etc.), restored via git, redone
- [x] Fix `generate_visualization.py` noise (fixed div_id, data date instead of now())
- [x] Workflow cleanup (no crawl4ai, no force push, pinned requirements.txt, py3.12)
- [x] Next.js site v1 in `site/` (zued style, ELO timeline, logos, static export)
- [x] Scraper schema extended: Votes, Input_Price, Output_Price, Context_Length
- [x] Enrichment backfill (only 2026-04-28/30 left, narrow retry scheduled)
- [x] Site v2: headline fix, rename, timeline top 30 tiered, open-source race,
      price/performance scatter, bundle 1.26MB -> 830KB
- [x] Site v3: frontier value list, standings removed
- [x] Site v4: 95% quality floor on value list, bar race section
- [x] Post-v4 direct edits: bar race top 15, input pricing in value list,
      bar race moved ahead of ELO timeline (order: race, timeline, open, value)
- [x] Final workflow: fetch -> commit data -> build site -> deploy to Pages
- [x] End-to-end verify: fetch idempotent, build loads 91 CSVs/37 enriched,
      visual check in browser (hero, bar race top 15, open race, value list
      with in/out pricing, floored scatter)
- [x] Removed obsolete generate_visualization.py, trimmed requirements.txt

## Review

Root cause of the 4-month outage: arena.ai redesign returned 200 via redirect;
every scraper error path printed and exited 0, so Actions stayed green while
saving nothing. Fixes: flight-payload parser, fail-loud exits, sanity floor,
no force-push, deterministic outputs.

Data: 91 snapshots 2025-05-03..2026-06-10, zero integrity problems, 37 files
enriched with Votes/Input_Price/Output_Price/Context_Length. Gap 2026-02..06
rebuilt from Wayback captures at full fidelity (~316-366 models/day).
2722-row provider-prefix corruption (back to 2025) repaired.

Site: Next.js 16 static export in site/, zued design system, four sections
(bar race, ELO timeline top 30, open-vs-proprietary race, value-for-money
frontier list with 95% floor + scatter). Deploy via Actions to Pages; user
must flip Pages source to "GitHub Actions".

Open follow-ups: none blocking. Optional: favicon/wordmark art, OG image.

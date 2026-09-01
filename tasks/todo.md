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

## 2026-09-01: Price source and snapshot immutability

- [x] Join OpenRouter's published prices onto each snapshot (fetch_openrouter_prices.py)
- [x] Rank the value frontier on a blended price, 3 parts output to 1 part input
- [x] Make snapshots append-only so reruns stop rewriting the record

## Review

arena's price column is unreliable in two distinct ways. It goes stale:
deepseek-v4-flash sat at a batch-tier price for weeks, 7x its real cost, which
is what pushed it off the value list. And it is not stable across fetches: two
pulls of the same 2026-08-27 cutoff disagreed on 15 models, gemini-2.5-pro
doubling while gpt-5.1 and gpt-5.2 halved, with ELO and votes identical. The
"arena reports exactly half" pattern is that instability, not a fixed offset.

OpenRouter now supplies the price wherever a model can be matched (122 of ~300
priced models), arena stays the fallback, and both land in the snapshot so the
switch is auditable. Matching is deliberately strict: only reasoning-effort
suffixes are stripped, since effort changes how many tokens a model emits, not
what a token costs. Tier words, dates, "-preview" and ":batch" endpoints are
never stripped. glm-5.3-flash and glm-5.3 are 17x apart; gpt-4-0125-preview is
half the price of gpt-4. A wrong match is worse than no match.

Ranking moved to a blended price, 3 output to 1 input. At 3:1 the frontier
holds exactly the models an output-only ranking produced, verified under both
price sources, so the blend moves the numbers and not the membership. At 1:1
grok-4.20 drops out, which is why the weight is a named constant.

Net effect on the list: deepseek-v4-flash is now the number one value pick,
mimo-v2.5 and gemma-4-31b drop out, both correctly dominated once
glm-5.3-flash carries its real $0.25 rather than arena's $0.50.

Snapshots are append-only at the cell level now: blanks may be filled and new
models added, recorded values never change, and conflicts are logged rather
than applied. This does not make arena's price right, it makes the record
deterministic. Where arena is the only source and it flip-flops, the first
observation stands and the disagreement shows up in the run log. The
2026-08-27 file, the only one the old overwrite semantics damaged, was
repaired by replaying the new rule over its first recorded version.

Two bugs surfaced when the merge first ran against live data, both now fixed.
Multiplying a per-token price by a million leaves representation noise, so a
third of the reported conflicts were 1.027 against 1.0270000000000001; derived
prices are rounded at the source and numbers are compared with a tolerance.
And arena returns an empty modelOrganization for 50 models, which pandas reads
back from CSV as NaN, so every run "filled" the same 50 cells and rewrote the
file forever; a cell now counts as blank when it is NaN or whitespace. A run
with nothing new to add reports unchanged and touches nothing.

Fixed after that: arena leaves modelOrganization empty for 50 models, but
every one of them has a real organisation recorded elsewhere in the history,
so "Unknown" would have thrown away a name we already hold. The fetcher now
recovers the provider from the newest snapshot that has one and falls back to
"Unknown" only if no snapshot ever knew it, which today is never. 800 rows
across 16 files were backfilled the same way. The integrity check in
extract_snapshot is blank-aware now: it tested isna(), which an empty string
passes, and that is how fifty empty providers went unnoticed for months.

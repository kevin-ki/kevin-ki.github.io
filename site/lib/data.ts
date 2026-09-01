import fs from "node:fs";
import path from "node:path";
import { cache } from "react";

export interface ModelRow {
  model: string;
  elo: number;
  provider: string;
  license: string;
  /** Optional enriched columns; only present in newer snapshot CSVs. */
  votes?: number;
  /** USD per million input tokens. */
  inputPrice?: number;
  /** USD per million output tokens. */
  outputPrice?: number;
  /** USD per million input tokens as published by OpenRouter, when matched. */
  orInputPrice?: number;
  /** USD per million output tokens as published by OpenRouter, when matched. */
  orOutputPrice?: number;
  contextLength?: number;
}

export interface Snapshot {
  date: string;
  rows: ModelRow[];
}

/** Minimal quote-aware CSV parser (handles "", embedded commas, CRLF). */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (c !== "\r") {
      field += c;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

const FILE_PATTERN = /^lmsys_snapshot_(\d{4}-\d{2}-\d{2})\.csv$/;

/**
 * Reads every lmsys_snapshot_YYYY-MM-DD.csv from ../data at build time.
 * Returns snapshots sorted ascending by date.
 */
export const loadSnapshots = cache((): Snapshot[] => {
  const dataDir = path.join(process.cwd(), "..", "data");
  const files = fs
    .readdirSync(dataDir)
    .filter((f) => FILE_PATTERN.test(f))
    .sort();

  if (files.length === 0) {
    throw new Error(`No lmsys_snapshot_*.csv files found in ${dataDir}`);
  }

  let skipped = 0;
  let enriched = 0;
  const snapshots: Snapshot[] = files.map((file) => {
    const date = FILE_PATTERN.exec(file)![1];
    const parsed = parseCsv(fs.readFileSync(path.join(dataDir, file), "utf8"));
    const [header, ...lines] = parsed;
    const col = {
      model: header.indexOf("Model_Name"),
      elo: header.indexOf("ELO_Score"),
      provider: header.indexOf("Provider"),
      license: header.indexOf("License"),
    };
    if (Object.values(col).some((i) => i < 0)) {
      throw new Error(`Unexpected header in ${file}: ${header.join(",")}`);
    }
    // Enriched columns exist only in newer files (and values may be empty).
    const optCol = {
      votes: header.indexOf("Votes"),
      inputPrice: header.indexOf("Input_Price"),
      outputPrice: header.indexOf("Output_Price"),
      orInputPrice: header.indexOf("OR_Input_Price"),
      orOutputPrice: header.indexOf("OR_Output_Price"),
      contextLength: header.indexOf("Context_Length"),
    };
    if (optCol.outputPrice >= 0) enriched++;
    const optNum = (line: string[], idx: number): number | undefined => {
      if (idx < 0) return undefined;
      const raw = (line[idx] ?? "").trim();
      if (raw === "") return undefined;
      const n = Number(raw);
      return Number.isFinite(n) ? n : undefined;
    };

    const rows: ModelRow[] = [];
    for (const line of lines) {
      if (line.length === 1 && line[0] === "") continue; // trailing blank line
      const model = (line[col.model] ?? "").trim();
      const elo = Number(line[col.elo]);
      if (!model || !Number.isFinite(elo)) {
        skipped++;
        continue;
      }
      rows.push({
        model,
        elo,
        provider: (line[col.provider] ?? "").trim(),
        license: (line[col.license] ?? "").trim(),
        votes: optNum(line, optCol.votes),
        inputPrice: optNum(line, optCol.inputPrice),
        outputPrice: optNum(line, optCol.outputPrice),
        orInputPrice: optNum(line, optCol.orInputPrice),
        orOutputPrice: optNum(line, optCol.orOutputPrice),
        contextLength: optNum(line, optCol.contextLength),
      });
    }
    rows.sort((a, b) => b.elo - a.elo);
    return { date, rows };
  });

  console.log(
    `[data] Loaded ${snapshots.length} snapshot CSVs from ${dataDir} ` +
      `(${snapshots[0].date} to ${snapshots[snapshots.length - 1].date}), ` +
      `${enriched} with price/votes columns` +
      (skipped > 0 ? `, skipped ${skipped} malformed rows` : ""),
  );
  return snapshots;
});

/* ---------------------------------- chart ---------------------------------- */

export const CHART_PALETTE = [
  "#DA6A5E",
  "#5C92E8",
  "#E8B43D",
  "#33B386",
  "#8F6CD4",
] as const;

export interface SeriesPoint {
  date: string;
  elo: number;
}

export interface ChartSeries {
  model: string;
  provider: string;
  color: string;
  /** True when the palette had to be reused (more than 5 providers). */
  dimmed: boolean;
  /** 1-based rank by latest ELO. */
  rank: number;
  latestElo: number;
  points: SeriesPoint[];
}

/** Latest-snapshot rows deduped by model name (rows are pre-sorted by ELO). */
function dedupedLatestRows(snapshots: Snapshot[]): ModelRow[] {
  const latest = snapshots[snapshots.length - 1];
  const seen = new Set<string>();
  const out: ModelRow[] = [];
  for (const row of latest.rows) {
    if (seen.has(row.model)) continue;
    seen.add(row.model);
    out.push(row);
  }
  return out;
}

/**
 * Providers ranked by their best model's latest ELO, mapped to the chart
 * palette. Shared by the timeline and the price scatter so a provider keeps
 * the same color everywhere.
 */
export function buildProviderColors(
  snapshots: Snapshot[],
): Map<string, { color: string; dimmed: boolean }> {
  const providerOrder: string[] = [];
  for (const row of dedupedLatestRows(snapshots)) {
    if (!providerOrder.includes(row.provider)) providerOrder.push(row.provider);
  }
  return new Map(
    providerOrder.map((p, i) => [
      p,
      { color: CHART_PALETTE[i % CHART_PALETTE.length], dimmed: i >= CHART_PALETTE.length },
    ]),
  );
}

/** Top N models by latest ELO, each with its full ELO history. */
export function buildChartSeries(snapshots: Snapshot[], topN = 30): ChartSeries[] {
  const top = dedupedLatestRows(snapshots).slice(0, topN);
  const providerColor = buildProviderColors(snapshots);

  return top.map((row, idx) => {
    const points: SeriesPoint[] = [];
    for (const snap of snapshots) {
      const match = snap.rows.find((r) => r.model === row.model);
      if (match) points.push({ date: snap.date, elo: match.elo });
    }
    const { color, dimmed } = providerColor.get(row.provider)!;
    return {
      model: row.model,
      provider: row.provider,
      color,
      dimmed,
      rank: idx + 1,
      latestElo: row.elo,
      points,
    };
  });
}

/* ------------------------------ open vs closed ------------------------------ */

/**
 * A license counts as open when it is anything other than Proprietary and not
 * unknown. Handles the "Propretary" typo present in some snapshots; "Other"
 * and "N/A" are treated as unknown rather than open.
 */
export function isOpenLicense(license: string): boolean {
  const l = license.trim().toLowerCase();
  return !["proprietary", "propretary", "unknown", "n/a", "other", ""].includes(l);
}

export interface OpenRacePoint {
  date: string;
  /** Best proprietary ELO in this snapshot (null when absent). */
  proprietary: number | null;
  /** Best open-license ELO in this snapshot (null when absent). */
  open: number | null;
}

export interface OpenRace {
  points: OpenRacePoint[];
  /** Current ELO gap, best proprietary minus best open (latest snapshot). */
  latestGap: number;
  bestProprietary: ModelRow;
  bestOpen: ModelRow & { rank: number };
}

export interface OpenStandingRow extends ModelRow {
  /** Overall rank in the latest snapshot (across all licenses). */
  rank: number;
}

/** Best proprietary vs best open ELO across every snapshot. */
export function buildOpenRace(snapshots: Snapshot[]): OpenRace {
  const points: OpenRacePoint[] = snapshots.map((snap) => {
    let proprietary: number | null = null;
    let open: number | null = null;
    for (const row of snap.rows) {
      // rows are sorted by ELO descending; first hit per bucket wins
      if (isOpenLicense(row.license)) {
        if (open === null) open = row.elo;
      } else if (proprietary === null) {
        proprietary = row.elo;
      }
      if (open !== null && proprietary !== null) break;
    }
    return { date: snap.date, proprietary, open };
  });

  const latestRows = dedupedLatestRows(snapshots);
  const bestProprietary = latestRows.find((r) => !isOpenLicense(r.license));
  const openIdx = latestRows.findIndex((r) => isOpenLicense(r.license));
  if (!bestProprietary || openIdx < 0) {
    throw new Error(
      "Latest snapshot is missing a proprietary or open-license model; cannot compute the open race.",
    );
  }
  const bestOpen = { ...latestRows[openIdx], rank: openIdx + 1 };

  return {
    points,
    latestGap: bestProprietary.elo - bestOpen.elo,
    bestProprietary,
    bestOpen,
  };
}

/** Top N open-license models from the latest snapshot, with overall rank. */
export function buildBestOpenModels(snapshots: Snapshot[], topN = 5): OpenStandingRow[] {
  const out: OpenStandingRow[] = [];
  dedupedLatestRows(snapshots).forEach((row, idx) => {
    if (out.length < topN && isOpenLicense(row.license)) {
      out.push({ ...row, rank: idx + 1 });
    }
  });
  return out;
}

/* ----------------------------- price vs performance ----------------------------- */

export type PriceSource = "openrouter" | "arena";

export interface ScatterPoint {
  model: string;
  provider: string;
  license: string;
  color: string;
  elo: number;
  /** Blended USD per million tokens; the number the ranking is built on. */
  price: number;
  /** USD per million output tokens behind the blend. */
  outputPrice: number;
  /** USD per million input tokens behind the blend. */
  inputPrice: number;
  /** Which price list the pair came from. */
  priceSource: PriceSource;
  /** On the Pareto frontier: no model is both cheaper and stronger. */
  frontier: boolean;
}

export interface PriceScatter {
  points: ScatterPoint[];
  /** Latest-snapshot models that had a usable price pair (incl. zero-priced). */
  pricedCount: number;
  /** How many of those were priced from OpenRouter rather than arena.ai. */
  openrouterCount: number;
  /** Frontier models ordered by ascending price. */
  frontier: ScatterPoint[];
}

/**
 * Minimum share of the strongest priced model's ELO a model must reach to
 * qualify for the value frontier. Cheap models more than 5% behind the leader
 * are plotted as plain dots but never connected by the coral line.
 */
export const VALUE_QUALITY_FLOOR = 0.95;

/**
 * Weight of the output price against 1 part input in the blended price.
 *
 * Providers quote the two separately and output dominates a chat workload, so
 * neither number alone is what a model costs to run. At 3:1 the frontier holds
 * exactly the same models an output-only ranking produced, so the blend moves
 * the numbers on the page, not who is on it. At 1:1 grok-4.20 drops out, which
 * is why the weight is a constant and not a guess buried in a formula.
 */
export const BLEND_OUTPUT_WEIGHT = 3;

interface EffectivePrice {
  input: number;
  output: number;
  blended: number;
  source: PriceSource;
}

/**
 * The prices to rank a model on, preferring OpenRouter's over arena's.
 *
 * arena's price column goes stale (deepseek-v4-flash sat at a batch-tier price
 * for weeks, 7x its real cost) and reports half the list price for a recurring
 * set of models, so where the daily fetcher could match the model on
 * OpenRouter we take that pair instead. Both halves must come from the same
 * source: blending one price list's output with another's input would describe
 * a model nobody can actually buy.
 */
function effectivePrice(row: ModelRow): EffectivePrice | undefined {
  const candidates: [number | undefined, number | undefined, PriceSource][] = [
    [row.orInputPrice, row.orOutputPrice, "openrouter"],
    [row.inputPrice, row.outputPrice, "arena"],
  ];
  for (const [input, output, source] of candidates) {
    if (input === undefined || output === undefined) continue;
    return {
      input,
      output,
      blended:
        (BLEND_OUTPUT_WEIGHT * output + input) / (BLEND_OUTPUT_WEIGHT + 1),
      source,
    };
  }
  return undefined;
}

/** Price vs ELO for the latest snapshot, with the Pareto frontier marked. */
export function buildPriceScatter(snapshots: Snapshot[]): PriceScatter {
  const providerColor = buildProviderColors(snapshots);
  const priced = dedupedLatestRows(snapshots)
    .map((row) => ({ row, price: effectivePrice(row) }))
    .filter(
      (e): e is { row: ModelRow; price: EffectivePrice } =>
        e.price !== undefined,
    );
  // Zero-priced rows cannot sit on a log axis; drop them from the plot.
  const plottable = priced.filter((e) => e.price.blended > 0);

  const points: ScatterPoint[] = plottable.map(({ row, price }) => ({
    model: row.model,
    provider: row.provider,
    license: row.license,
    color: providerColor.get(row.provider)!.color,
    elo: row.elo,
    price: price.blended,
    outputPrice: price.output,
    inputPrice: price.input,
    priceSource: price.source,
    frontier: false,
  }));

  // Quality floor: only models within 5% of the strongest priced model's ELO
  // qualify for the frontier. Applied before the frontier walk, so cheap but
  // weak models never anchor the coral line. Because every excluded model
  // scores below every qualifying one, "nothing cheaper scores higher" still
  // holds globally for the resulting picks.
  const leaderElo = Math.max(...points.map((p) => p.elo));
  const floor = VALUE_QUALITY_FLOOR * leaderElo;

  // Pareto frontier: walk by ascending price (ELO desc on ties); a point is on
  // the frontier when it beats every cheaper model's ELO.
  const byPrice = points
    .filter((p) => p.elo >= floor)
    .sort((a, b) => a.price - b.price || b.elo - a.elo);
  let bestElo = -Infinity;
  const frontier: ScatterPoint[] = [];
  for (const p of byPrice) {
    if (p.elo > bestElo) {
      p.frontier = true;
      frontier.push(p);
      bestElo = p.elo;
    }
  }

  return {
    points,
    pricedCount: priced.length,
    openrouterCount: priced.filter((e) => e.price.source === "openrouter").length,
    frontier,
  };
}

/* ------------------------------ value for money ------------------------------ */

export interface ValueRow {
  model: string;
  provider: string;
  license: string;
  elo: number;
  /** Blended USD per million tokens; the number the ranking is built on. */
  price: number;
  /** USD per million output tokens behind the blend. */
  outputPrice: number;
  /** USD per million input tokens behind the blend. */
  inputPrice: number;
  /** Which price list the pair came from. */
  priceSource: PriceSource;
  /** Percent of the leader's ELO this model reaches (0 to 100). */
  scorePct: number;
  /** Percent of the leader's blended price this model costs (0 to 100). */
  pricePct: number;
  isLeader: boolean;
}

/**
 * The Pareto frontier of the price scatter as a ranked list, cheapest first.
 * Derived from the scatter's frontier so the list and the chart always agree.
 * Percentages compare each model to the strongest priced model, which is the
 * frontier's most expensive end (it beats every cheaper model by definition).
 */
export function buildValueList(scatter: PriceScatter): ValueRow[] {
  const { frontier } = scatter;
  if (frontier.length === 0) {
    throw new Error("Price scatter has an empty frontier; cannot build the value list.");
  }
  const leader = frontier[frontier.length - 1];
  return frontier.map((p) => ({
    model: p.model,
    provider: p.provider,
    license: p.license,
    elo: p.elo,
    price: p.price,
    outputPrice: p.outputPrice,
    inputPrice: p.inputPrice,
    priceSource: p.priceSource,
    scorePct: (p.elo / leader.elo) * 100,
    pricePct: (p.price / leader.price) * 100,
    isLeader: p === leader,
  }));
}

/* --------------------------------- bar race --------------------------------- */

export interface BarRaceModel {
  model: string;
  provider: string;
  color: string;
}

export interface BarRaceEntry {
  /** Index into BarRaceData.models. */
  m: number;
  elo: number;
}

export interface BarRaceFrame {
  date: string;
  /** Top models of this snapshot, strongest first. */
  entries: BarRaceEntry[];
}

export interface BarRaceData {
  models: BarRaceModel[];
  frames: BarRaceFrame[];
}

/**
 * The top N models of every snapshot, for the "who led when" bar race.
 * Models are stored once and referenced by index to keep the serialized
 * client props small. Colors come from buildProviderColors so a provider
 * matches the timeline and the scatter; providers that only appear in
 * historic snapshots get the next palette slots in first-seen order.
 */
export function buildBarRace(snapshots: Snapshot[], topN = 12): BarRaceData {
  const providerColor = buildProviderColors(snapshots);
  const extraColor = new Map<string, string>();
  const colorFor = (provider: string): string => {
    const known = providerColor.get(provider);
    if (known) return known.color;
    let color = extraColor.get(provider);
    if (!color) {
      color = CHART_PALETTE[(providerColor.size + extraColor.size) % CHART_PALETTE.length];
      extraColor.set(provider, color);
    }
    return color;
  };

  const models: BarRaceModel[] = [];
  const indexByModel = new Map<string, number>();

  const frames: BarRaceFrame[] = snapshots.map((snap) => {
    const seen = new Set<string>();
    const entries: BarRaceEntry[] = [];
    for (const row of snap.rows) {
      if (entries.length >= topN) break;
      if (seen.has(row.model)) continue; // rows are pre-sorted by ELO desc
      seen.add(row.model);
      let idx = indexByModel.get(row.model);
      if (idx === undefined) {
        idx = models.length;
        indexByModel.set(row.model, idx);
        models.push({
          model: row.model,
          provider: row.provider,
          color: colorFor(row.provider),
        });
      }
      entries.push({ m: idx, elo: row.elo });
    }
    return { date: snap.date, entries };
  });

  return { models, frames };
}

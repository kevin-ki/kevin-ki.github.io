import { BarRace } from "@/components/BarRace";
import { EloChart } from "@/components/EloChart";
import { LicensePill } from "@/components/LicensePill";
import { OpenRaceChart } from "@/components/OpenRaceChart";
import { PriceScatter } from "@/components/PriceScatter";
import { ProviderLogo } from "@/components/ProviderLogo";
import { ValueList } from "@/components/ValueList";
import {
  buildBarRace,
  buildBestOpenModels,
  buildChartSeries,
  buildOpenRace,
  buildPriceScatter,
  buildValueList,
  loadSnapshots,
} from "@/lib/data";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function formatDate(iso: string) {
  const [y, m, d] = iso.split("-").map(Number);
  return `${MONTHS[m - 1]} ${d}, ${y}`;
}

function SectionHeading({
  number,
  label,
  title,
  description,
}: {
  number: string;
  label: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-8">
      <p className="section-label">
        <span className="text-coral">{number}</span>
        <span className="mx-2 text-faint">·</span>
        {label}
      </p>
      <h2 className="mt-3 font-display text-2xl font-bold tracking-tight text-fg sm:text-3xl">
        {title}
      </h2>
      <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
        {description}
      </p>
    </div>
  );
}

function StatChip({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-2xl border border-cardborder bg-card px-5 py-4">
      <div className="font-mono text-lg leading-none text-fg sm:text-xl">
        {value}
      </div>
      <div className="section-label mt-2.5">{label}</div>
    </div>
  );
}

export default function Page() {
  const snapshots = loadSnapshots();
  const latest = snapshots[snapshots.length - 1];
  const first = snapshots[0];
  const series = buildChartSeries(snapshots, 30);
  const barRace = buildBarRace(snapshots, 15);
  const openRace = buildOpenRace(snapshots);
  const bestOpen = buildBestOpenModels(snapshots, 5);
  const scatter = buildPriceScatter(snapshots);
  const valueRows = buildValueList(scatter);
  const leader = valueRows[valueRows.length - 1];
  const zeroPriced = scatter.pricedCount - scatter.points.length;

  return (
    <div className="mx-auto max-w-[1100px] px-6">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-hairline py-6">
        <span className="font-display text-[15px] font-bold tracking-tight text-fg">
          arena.ai
          <span className="font-semibold text-muted"> · text models</span>
        </span>
        <span className="font-mono text-xs text-muted">
          Data as of {latest.date}
        </span>
      </header>

      <main>
        {/* Hero */}
        <section className="pb-20 pt-20 sm:pt-28">
          <h1 className="max-w-3xl font-display text-4xl font-bold leading-[1.08] tracking-tight text-fg sm:text-6xl">
            The text models of arena.ai,
            <br className="hidden sm:block" /> tracked through{" "}
            <span className="text-coral">time</span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            arena.ai shows today&apos;s leaderboard. This site keeps every daily
            ELO snapshot, so you can see how the race actually unfolded.
          </p>
          <div className="mt-12 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatChip
              value={String(latest.rows.length)}
              label="Models tracked"
            />
            <StatChip value={String(snapshots.length)} label="Snapshots" />
            <StatChip
              value={`${first.date} → ${latest.date}`}
              label="Range covered"
            />
          </div>
        </section>

        {/* Section 01: Who led when */}
        <section className="pb-24">
          <SectionHeading
            number="01"
            label="Who led when"
            title="The top of the board, replayed"
            description={`The top 15 of every snapshot as a bar race. Press play, or scrub the slider to any date.`}
          />
          <div className="overflow-hidden rounded-2xl border border-cardborder bg-card p-4 sm:p-6">
            <BarRace data={barRace} />
          </div>
        </section>

        {/* Section 02: Timeline */}
        <section className="pb-24">
          <SectionHeading
            number="02"
            label="Timeline"
            title="ELO over time"
            description={`The top 30 models by latest arena score, traced across every snapshot. The top 10 are drawn in full with name labels; ranks 11 to 30 sit faintly behind them. Hover any line to bring it forward.`}
          />
          <div className="overflow-hidden rounded-2xl border border-cardborder bg-card p-4 sm:p-6">
            <EloChart series={series} dates={snapshots.map((s) => s.date)} />
          </div>
        </section>

        {/* Section 03: Open-source race */}
        <section className="pb-24">
          <SectionHeading
            number="03"
            label="Open-source race"
            title="Open weights vs proprietary"
            description={`The best proprietary model against the best openly licensed model in every snapshot. The gap tells you how far open weights trail the closed frontier.`}
          />
          <div className="overflow-hidden rounded-2xl border border-cardborder bg-card p-4 sm:p-6">
            <OpenRaceChart points={openRace.points} />
            <p className="mt-4 border-t border-hairline pt-4 text-sm leading-relaxed text-muted">
              Current gap:{" "}
              <span className="font-mono text-fg">
                {openRace.latestGap.toFixed(1)}
              </span>{" "}
              ELO points. The best open model right now is{" "}
              <span className="font-medium text-fg">{openRace.bestOpen.model}</span>{" "}
              ({openRace.bestOpen.license}, rank {openRace.bestOpen.rank} overall)
              vs{" "}
              <span className="font-medium text-fg">
                {openRace.bestProprietary.model}
              </span>{" "}
              at the top.
            </p>
          </div>

          {/* Best open models right now */}
          <div className="mt-4 rounded-2xl border border-cardborder bg-card px-5 py-2 sm:px-7">
            <p className="section-label pb-1 pt-4">Best open models right now</p>
            <ul>
              {bestOpen.map((row) => (
                <li
                  key={row.model}
                  className="flex items-center gap-3 border-b border-hairline py-3 last:border-b-0"
                >
                  <ProviderLogo provider={row.provider} size={18} />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-fg">
                      {row.model}
                    </div>
                    <div className="text-xs text-muted">{row.provider}</div>
                  </div>
                  <span className="hidden sm:inline-block">
                    <LicensePill license={row.license} />
                  </span>
                  <span className="ml-auto font-mono text-sm text-fg">
                    {Math.round(row.elo)}
                  </span>
                  <span className="w-16 text-right font-mono text-xs text-faint">
                    #{row.rank} overall
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Section 04: Value for money */}
        <section className="pb-24">
          <SectionHeading
            number="04"
            label="Value for money"
            title="Best value for money"
            description={`The value picks from the ${formatDate(latest.date)} snapshot, cheapest first. A model makes this list when it stays within 5% of the leader's score and nothing cheaper scores higher in the arena.`}
          />
          <div className="rounded-2xl border border-cardborder bg-card px-5 py-2 sm:px-7">
            <ValueList rows={valueRows} />
            <p className="border-t border-hairline py-4 text-sm leading-relaxed text-muted">
              Percentages compare each model to{" "}
              <span className="font-medium text-fg">{leader.model}</span>, the
              current number one by arena score. Models more than 5% behind
              the leader&apos;s score are not considered, cheap but too far
              back. Prices are blended per million tokens, three parts output
              to one part input, since neither number alone is what a model
              costs to run. They come from OpenRouter for{" "}
              <span className="font-mono text-fg">
                {scatter.openrouterCount}
              </span>{" "}
              of {scatter.pricedCount} priced models and from arena.ai for the
              rest, because arena&apos;s figures go stale and read half the
              list price for a recurring set of models. Models without price
              data
              {zeroPriced > 0
                ? ` (and ${zeroPriced} zero-priced models)`
                : ""}{" "}
              are excluded.
            </p>
          </div>
          <div className="mt-4 overflow-hidden rounded-2xl border border-cardborder bg-card p-4 sm:p-6">
            <PriceScatter points={scatter.points} />
            <p className="mt-4 border-t border-hairline pt-4 text-sm leading-relaxed text-muted">
              Every dot is a model. The coral line connects the value picks
              above: at each price, nothing cheaper scores higher, and models
              more than 5% behind the leader&apos;s score are not considered.{" "}
              <span className="font-mono text-fg">{scatter.pricedCount}</span>{" "}
              of {latest.rows.length} models in the latest snapshot had price
              data{zeroPriced > 0
                ? ` (${zeroPriced} zero-priced models are omitted from the log axis)`
                : ""}
              .
            </p>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-hairline py-10">
        <p className="text-sm leading-relaxed text-muted">
          Text Models of arena.ai keeps a daily history of the{" "}
          <a
            href="https://arena.ai/leaderboard/text"
            className="text-fg underline decoration-cardborder underline-offset-4 transition-colors hover:decoration-coral"
          >
            arena.ai text leaderboard
          </a>
          . Snapshots collected daily.
        </p>
        <p className="mt-2 text-sm text-muted">
          Source code and raw snapshots on{" "}
          <a
            href="https://github.com/kevin-ki/kevin-ki.github.io"
            className="text-fg underline decoration-cardborder underline-offset-4 transition-colors hover:decoration-coral"
          >
            GitHub
          </a>
          .
        </p>
      </footer>
    </div>
  );
}

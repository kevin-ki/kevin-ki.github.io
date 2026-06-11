import type { ValueRow } from "@/lib/data";
import { LicensePill } from "./LicensePill";
import { ProviderLogo } from "./ProviderLogo";

/** "0.39%" below one percent, "6.0%" above; keeps tiny shares legible. */
function pct(v: number): string {
  return `${v.toFixed(v < 1 ? 2 : 1)}%`;
}

function price(p: number): string {
  return `$${p.toFixed(2)}/M`;
}

export function ValueList({ rows }: { rows: ValueRow[] }) {
  return (
    <ol>
      <li
        aria-hidden
        className="flex items-center gap-3 border-b border-cardborder py-3"
      >
        <span className="section-label flex-1">Cheapest first</span>
        <span className="section-label w-24 text-right sm:w-28">Price $/M</span>
        <span className="section-label w-12 text-right sm:w-14">ELO</span>
      </li>
      {rows.map((row, idx) => (
        <li
          key={row.model}
          className="border-b border-hairline py-3.5 last:border-b-0"
        >
          <div className="flex items-center gap-3">
            <span className="w-5 shrink-0 font-mono text-xs text-faint">
              {String(idx + 1).padStart(2, "0")}
            </span>
            <ProviderLogo provider={row.provider} size={18} />
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-fg">
                {row.model}
              </div>
              <div className="text-xs text-muted">{row.provider}</div>
            </div>
            <span className="hidden md:inline-block">
              <LicensePill license={row.license} />
            </span>
            <span className="ml-auto w-24 shrink-0 text-right sm:w-28">
              <span className="block font-mono text-sm text-fg">
                {price(row.outputPrice)} out
              </span>
              {row.inputPrice !== undefined && (
                <span className="block font-mono text-xs text-muted">
                  {price(row.inputPrice)} in
                </span>
              )}
            </span>
            <span className="w-12 shrink-0 text-right font-mono text-sm text-fg sm:w-14">
              {Math.round(row.elo)}
            </span>
          </div>
          <p className="mt-1.5 pl-8 text-xs leading-relaxed text-muted sm:pl-[62px]">
            {row.isLeader ? (
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-coral">
                the leader
              </span>
            ) : (
              <>
                reaches{" "}
                <span className="font-mono text-fg">{pct(row.scorePct)}</span>
                {" of the leader's score at "}
                <span className="font-mono text-coral">{pct(row.pricePct)}</span>
                {" of its price"}
              </>
            )}
          </p>
        </li>
      ))}
    </ol>
  );
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear, scaleUtc } from "d3-scale";
import { curveMonotoneX, line } from "d3-shape";
import type { ChartSeries, SeriesPoint } from "@/lib/data";
import { ProviderLogo } from "./ProviderLogo";

interface EloChartProps {
  series: ChartSeries[];
  /** All snapshot dates (ISO yyyy-mm-dd), ascending. */
  dates: string[];
}

const HEIGHT = 460;
const MARGIN = { top: 18, right: 12, bottom: 36, left: 48 };
const LABEL_GUTTER = 212;
/** Ranks 1..PRIMARY_COUNT get full-strength lines and end labels. */
const PRIMARY_COUNT = 10;
const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const toDate = (iso: string) => new Date(`${iso}T00:00:00Z`);

function formatTick(d: Date) {
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

function formatDateLong(iso: string) {
  const d = toDate(iso);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

export function EloChart({ series, dates }: EloChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(1040);
  const [hover, setHoverState] = useState<{ dateIdx: number; seriesIdx: number } | null>(
    null,
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.max(320, w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const showLabels = width >= 720;
  const plotRight = width - MARGIN.right - (showLabels ? LABEL_GUTTER : 0);
  const plotBottom = HEIGHT - MARGIN.bottom;

  const primary = useMemo(
    () => series.filter((s) => s.rank <= PRIMARY_COUNT),
    [series],
  );

  const { x, y, xs } = useMemo(() => {
    const x = scaleUtc()
      .domain([toDate(dates[0]), toDate(dates[dates.length - 1])])
      .range([MARGIN.left, plotRight]);
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series) {
      for (const p of s.points) {
        if (p.elo < lo) lo = p.elo;
        if (p.elo > hi) hi = p.elo;
      }
    }
    const pad = (hi - lo) * 0.06 || 10;
    const y = scaleLinear()
      .domain([lo - pad, hi + pad])
      .range([plotBottom, MARGIN.top])
      .nice();
    return { x, y, xs: dates.map((d) => x(toDate(d))) };
  }, [series, dates, plotRight, plotBottom]);

  const paths = useMemo(() => {
    const gen = line<SeriesPoint>()
      .x((p) => x(toDate(p.date)))
      .y((p) => y(p.elo))
      .curve(curveMonotoneX);
    return series.map((s) => gen(s.points) ?? "");
  }, [series, x, y]);

  // Right-edge labels for the primary tier, nudged apart to avoid collisions.
  const labels = useMemo(() => {
    const minGap = 24;
    const ls = primary
      .map((s) => ({
        seriesIdx: series.indexOf(s),
        ly: y(s.points[s.points.length - 1].elo),
      }))
      .sort((a, b) => a.ly - b.ly);
    if (ls.length > 0) {
      ls[0].ly = Math.max(ls[0].ly, MARGIN.top);
      for (let k = 1; k < ls.length; k++) {
        ls[k].ly = Math.max(ls[k].ly, ls[k - 1].ly + minGap);
      }
      for (let k = ls.length - 1; k >= 0; k--) {
        const limit = k === ls.length - 1 ? plotBottom : ls[k + 1].ly - minGap;
        ls[k].ly = Math.min(ls[k].ly, limit);
      }
    }
    return ls;
  }, [primary, series, y, plotBottom]);

  const valueByDate = useMemo(
    () => series.map((s) => new Map(s.points.map((p) => [p.date, p.elo]))),
    [series],
  );

  const yTicks = y.ticks(6);
  const xTicks = x.ticks(width < 760 ? 4 : 7);

  function handleMove(e: React.PointerEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px < MARGIN.left - 16 || px > plotRight + 16) {
      setHoverState(null);
      return;
    }
    // Nearest snapshot date on x ...
    let dateIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(xs[i] - px);
      if (d < bestDist) {
        bestDist = d;
        dateIdx = i;
      }
    }
    // ... then the nearest line on y at that date.
    const date = dates[dateIdx];
    let seriesIdx = -1;
    let bestDy = Infinity;
    for (let i = 0; i < series.length; i++) {
      const elo = valueByDate[i].get(date);
      if (elo === undefined) continue;
      const dy = Math.abs(y(elo) - py);
      if (dy < bestDy) {
        bestDy = dy;
        seriesIdx = i;
      }
    }
    setHoverState(seriesIdx < 0 ? null : { dateIdx, seriesIdx });
  }

  const active =
    hover === null
      ? null
      : {
          cx: xs[hover.dateIdx],
          date: dates[hover.dateIdx],
          s: series[hover.seriesIdx],
          elo: valueByDate[hover.seriesIdx].get(dates[hover.dateIdx])!,
        };

  const tooltipFlip = active !== null && active.cx > plotRight - 250;

  return (
    <div ref={containerRef} className="relative w-full select-none">
      <svg
        ref={svgRef}
        width={width}
        height={HEIGHT}
        role="img"
        aria-label="ELO scores of the top 30 models over time"
        onPointerMove={handleMove}
        onPointerLeave={() => setHoverState(null)}
        className="block"
      >
        {/* horizontal hairline gridlines + y ticks */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={MARGIN.left}
              x2={plotRight}
              y1={y(t)}
              y2={y(t)}
              stroke="#1E1B1D"
              strokeWidth={1}
            />
            <text
              x={MARGIN.left - 10}
              y={y(t)}
              dy="0.32em"
              textAnchor="end"
              className="font-mono"
              fontSize={10}
              fill="#555555"
            >
              {t}
            </text>
          </g>
        ))}

        {/* x ticks */}
        {xTicks.map((t) => (
          <text
            key={t.getTime()}
            x={x(t)}
            y={HEIGHT - 12}
            textAnchor="middle"
            className="font-mono"
            fontSize={10}
            fill="#555555"
          >
            {formatTick(t)}
          </text>
        ))}

        {/* crosshair */}
        {active && (
          <line
            x1={active.cx}
            x2={active.cx}
            y1={MARGIN.top}
            y2={plotBottom}
            stroke="#2A2729"
            strokeWidth={1}
          />
        )}

        {/* series lines: secondary tier (ranks 11-30) under the primary tier */}
        {series.map((s, i) => {
          const isPrimary = s.rank <= PRIMARY_COUNT;
          const isActive = hover?.seriesIdx === i;
          return (
            <path
              key={s.model}
              d={paths[i]}
              fill="none"
              stroke={s.color}
              strokeWidth={isActive ? 2.25 : isPrimary ? 2 : 1}
              strokeOpacity={isActive ? 1 : isPrimary ? 0.95 : 0.35}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          );
        })}

        {/* end-of-line dots, primary tier only */}
        {primary.map((s) => {
          const last = s.points[s.points.length - 1];
          return (
            <circle
              key={s.model}
              cx={x(toDate(last.date))}
              cy={y(last.elo)}
              r={3}
              fill={s.color}
            />
          );
        })}

        {/* hover dot on the highlighted line */}
        {active && (
          <circle
            cx={active.cx}
            cy={y(active.elo)}
            r={4}
            fill="#0B090A"
            stroke={active.s.color}
            strokeWidth={2}
          />
        )}
      </svg>

      {/* right-edge labels: provider logo + model name (primary tier) */}
      {showLabels &&
        labels.map(({ seriesIdx, ly }) => {
          const s = series[seriesIdx];
          return (
            <div
              key={s.model}
              className="pointer-events-none absolute flex items-center gap-1.5"
              style={{
                left: plotRight + 12,
                top: ly,
                transform: "translateY(-50%)",
                maxWidth: LABEL_GUTTER - 16,
              }}
            >
              <span className="h-px w-2 shrink-0" style={{ background: s.color }} />
              <ProviderLogo provider={s.provider} size={14} />
              <span
                className="truncate font-mono text-[11px] leading-none"
                style={{ color: hover?.seriesIdx === seriesIdx ? "#F2F2F2" : "#8C8C8C" }}
              >
                {s.model}
              </span>
            </div>
          );
        })}

      {/* tooltip for the highlighted line */}
      {active && (
        <div
          className="pointer-events-none absolute z-10 rounded-xl border border-cardborder bg-card px-3.5 py-3 shadow-[0_8px_28px_rgba(0,0,0,0.5)]"
          style={{
            top: Math.min(Math.max(y(active.elo) - 48, MARGIN.top), plotBottom - 96),
            left: tooltipFlip ? undefined : active.cx + 14,
            right: tooltipFlip ? width - active.cx + 14 : undefined,
            minWidth: 200,
          }}
        >
          <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
            {formatDateLong(active.date)}
          </div>
          <div className="flex items-center gap-2">
            <ProviderLogo provider={active.s.provider} size={16} />
            <span className="max-w-[200px] truncate text-xs font-medium text-fg">
              {active.s.model}
            </span>
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-6">
            <span className="text-xs text-muted">{active.s.provider}</span>
            <span className="font-mono text-sm" style={{ color: active.s.color }}>
              {Math.round(active.elo)}
            </span>
          </div>
          <div className="mt-1 text-right font-mono text-[10px] text-faint">
            rank {String(active.s.rank).padStart(2, "0")} today
          </div>
        </div>
      )}

      {/* compact legend when right-edge labels are hidden */}
      {!showLabels && (
        <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2">
          {primary.map((s) => (
            <span key={s.model} className="flex items-center gap-1.5">
              <span className="h-px w-3 shrink-0" style={{ background: s.color }} />
              <ProviderLogo provider={s.provider} size={13} />
              <span className="font-mono text-[11px] text-muted">{s.model}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

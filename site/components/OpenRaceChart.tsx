"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear, scaleUtc } from "d3-scale";
import { curveMonotoneX, line } from "d3-shape";
import type { OpenRacePoint } from "@/lib/data";

interface OpenRaceChartProps {
  points: OpenRacePoint[];
}

const HEIGHT = 380;
const MARGIN = { top: 18, right: 16, bottom: 36, left: 48 };
const PROPRIETARY_COLOR = "#DA6A5E";
const OPEN_COLOR = "#33B386";
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

export function OpenRaceChart({ points }: OpenRaceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(1040);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

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

  const plotRight = width - MARGIN.right;
  const plotBottom = HEIGHT - MARGIN.bottom;

  const { x, y, xs } = useMemo(() => {
    const x = scaleUtc()
      .domain([toDate(points[0].date), toDate(points[points.length - 1].date)])
      .range([MARGIN.left, plotRight]);
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of points) {
      for (const v of [p.proprietary, p.open]) {
        if (v === null) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const pad = (hi - lo) * 0.08 || 10;
    const y = scaleLinear()
      .domain([lo - pad, hi + pad])
      .range([plotBottom, MARGIN.top])
      .nice();
    return { x, y, xs: points.map((p) => x(toDate(p.date))) };
  }, [points, plotRight, plotBottom]);

  const [propPath, openPath] = useMemo(() => {
    const make = (get: (p: OpenRacePoint) => number | null) =>
      line<OpenRacePoint>()
        .defined((p) => get(p) !== null)
        .x((p) => x(toDate(p.date)))
        .y((p) => y(get(p) as number))
        .curve(curveMonotoneX)(points) ?? "";
    return [make((p) => p.proprietary), make((p) => p.open)];
  }, [points, x, y]);

  const yTicks = y.ticks(6);
  const xTicks = x.ticks(width < 760 ? 4 : 7);

  function handleMove(e: React.PointerEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    if (px < MARGIN.left - 16 || px > plotRight + 16) {
      setHoverIdx(null);
      return;
    }
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(xs[i] - px);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    setHoverIdx(best);
  }

  const hover = hoverIdx === null ? null : { ...points[hoverIdx], cx: xs[hoverIdx] };
  const tooltipFlip = hover !== null && hover.cx > plotRight - 240;

  return (
    <div ref={containerRef} className="relative w-full select-none">
      {/* legend */}
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1">
        <span className="flex items-center gap-2">
          <span className="h-0.5 w-4" style={{ background: PROPRIETARY_COLOR }} />
          <span className="font-mono text-[11px] text-muted">best proprietary</span>
        </span>
        <span className="flex items-center gap-2">
          <span className="h-0.5 w-4" style={{ background: OPEN_COLOR }} />
          <span className="font-mono text-[11px] text-muted">best open license</span>
        </span>
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={HEIGHT}
        role="img"
        aria-label="Best proprietary versus best open-license ELO over time"
        onPointerMove={handleMove}
        onPointerLeave={() => setHoverIdx(null)}
        className="block"
      >
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

        {hover && (
          <line
            x1={hover.cx}
            x2={hover.cx}
            y1={MARGIN.top}
            y2={plotBottom}
            stroke="#2A2729"
            strokeWidth={1}
          />
        )}

        <path
          d={propPath}
          fill="none"
          stroke={PROPRIETARY_COLOR}
          strokeWidth={2}
          strokeOpacity={0.95}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <path
          d={openPath}
          fill="none"
          stroke={OPEN_COLOR}
          strokeWidth={2}
          strokeOpacity={0.95}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {hover &&
          (
            [
              [hover.proprietary, PROPRIETARY_COLOR],
              [hover.open, OPEN_COLOR],
            ] as const
          ).map(
            ([v, color]) =>
              v !== null && (
                <circle
                  key={color}
                  cx={hover.cx}
                  cy={y(v)}
                  r={3.5}
                  fill="#0B090A"
                  stroke={color}
                  strokeWidth={1.75}
                />
              ),
          )}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-xl border border-cardborder bg-card px-3.5 py-3 shadow-[0_8px_28px_rgba(0,0,0,0.5)]"
          style={{
            top: MARGIN.top + 30,
            left: tooltipFlip ? undefined : hover.cx + 14,
            right: tooltipFlip ? width - hover.cx + 14 : undefined,
            minWidth: 190,
          }}
        >
          <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
            {formatDateLong(hover.date)}
          </div>
          <ul className="space-y-1">
            <li className="flex items-center gap-2">
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: PROPRIETARY_COLOR }}
              />
              <span className="text-xs text-fg">Proprietary</span>
              <span className="ml-auto pl-4 font-mono text-xs text-muted">
                {hover.proprietary === null ? "n/a" : Math.round(hover.proprietary)}
              </span>
            </li>
            <li className="flex items-center gap-2">
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: OPEN_COLOR }}
              />
              <span className="text-xs text-fg">Open</span>
              <span className="ml-auto pl-4 font-mono text-xs text-muted">
                {hover.open === null ? "n/a" : Math.round(hover.open)}
              </span>
            </li>
            {hover.proprietary !== null && hover.open !== null && (
              <li className="flex items-center gap-2 border-t border-hairline pt-1.5">
                <span className="text-xs text-muted">Gap</span>
                <span className="ml-auto pl-4 font-mono text-xs text-fg">
                  {(hover.proprietary - hover.open).toFixed(1)}
                </span>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

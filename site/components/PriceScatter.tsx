"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear, scaleLog } from "d3-scale";
import type { ScatterPoint } from "@/lib/data";
import { ProviderLogo } from "./ProviderLogo";

interface PriceScatterProps {
  points: ScatterPoint[];
}

const HEIGHT = 460;
const MARGIN = { top: 22, right: 24, bottom: 40, left: 48 };
const CORAL = "#E8716D";
const HOVER_RADIUS = 26;

function formatPrice(p: number): string {
  if (p >= 1) {
    return `$${Number.isInteger(p) ? p : p.toFixed(2).replace(/\.?0+$/, "")}`;
  }
  return `$${p.toFixed(p < 0.1 ? 3 : 2).replace(/0+$/, "").replace(/\.$/, "")}`;
}

interface LabelBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

function intersects(a: LabelBox, b: LabelBox) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

export function PriceScatter({ points }: PriceScatterProps) {
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
  const showFrontierLabels = width >= 640;

  const { x, y, xTicks } = useMemo(() => {
    let pLo = Infinity;
    let pHi = -Infinity;
    let eLo = Infinity;
    let eHi = -Infinity;
    for (const p of points) {
      if (p.price < pLo) pLo = p.price;
      if (p.price > pHi) pHi = p.price;
      if (p.elo < eLo) eLo = p.elo;
      if (p.elo > eHi) eHi = p.elo;
    }
    const x = scaleLog()
      .domain([pLo / 1.4, pHi * 1.4])
      .range([MARGIN.left, plotRight]);
    const pad = (eHi - eLo) * 0.06 || 10;
    const y = scaleLinear()
      .domain([eLo - pad, eHi + pad])
      .range([plotBottom, MARGIN.top])
      .nice();
    const xTicks = [0.01, 0.1, 1, 10, 100, 1000].filter(
      (t) => t >= pLo / 1.4 && t <= pHi * 1.4,
    );
    return { x, y, xTicks };
  }, [points, plotRight, plotBottom]);

  const positions = useMemo(
    () => points.map((p) => ({ px: x(p.price), py: y(p.elo) })),
    [points, x, y],
  );

  const frontier = useMemo(
    () =>
      points
        .map((p, i) => ({ p, i }))
        .filter(({ p }) => p.frontier)
        .sort((a, b) => a.p.price - b.p.price),
    [points],
  );

  const frontierPath = useMemo(
    () =>
      frontier
        .map(({ i }, k) => `${k === 0 ? "M" : "L"}${positions[i].px},${positions[i].py}`)
        .join(""),
    [frontier, positions],
  );

  // Frontier labels with simple collision avoidance: try a few candidate
  // anchor positions per label and keep the first that fits.
  const frontierLabels = useMemo(() => {
    const placed: LabelBox[] = [];
    const out: { i: number; lx: number; ly: number; anchor: "start" | "end" }[] = [];
    const charW = 6.1; // ~10px mono
    const h = 12;
    const padBox = (b: LabelBox): LabelBox => ({
      x: b.x - 4,
      y: b.y - 3,
      w: b.w + 8,
      h: b.h + 6,
    });
    for (const { p, i } of frontier) {
      const { px, py } = positions[i];
      const w = p.model.length * charW;
      const candidates: { lx: number; ly: number; anchor: "start" | "end" }[] = [
        { lx: px + 9, ly: py - 8, anchor: "start" },
        { lx: px + 9, ly: py + 16, anchor: "start" },
        { lx: px - 9, ly: py - 8, anchor: "end" },
        { lx: px - 9, ly: py + 16, anchor: "end" },
        { lx: px + 9, ly: py - 22, anchor: "start" },
        { lx: px - 9, ly: py + 30, anchor: "end" },
      ];
      let pick = candidates[0];
      for (const c of candidates) {
        const bx = c.anchor === "start" ? c.lx : c.lx - w;
        const box = { x: bx, y: c.ly - h + 2, w, h };
        const inPlot =
          bx >= MARGIN.left - 4 &&
          bx + w <= width - 4 &&
          box.y >= 4 &&
          c.ly <= plotBottom - 2;
        if (inPlot && !placed.some((b) => intersects(b, padBox(box)))) {
          pick = c;
          placed.push(box);
          break;
        }
      }
      out.push({ i, ...pick });
    }
    return out;
  }, [frontier, positions, width, plotBottom]);

  const yTicks = y.ticks(6);

  function handleMove(e: React.PointerEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    let best = -1;
    let bestDist = HOVER_RADIUS * HOVER_RADIUS;
    for (let i = 0; i < positions.length; i++) {
      const dx = positions[i].px - px;
      const dy = positions[i].py - py;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    setHoverIdx(best < 0 ? null : best);
  }

  const hover = hoverIdx === null ? null : { p: points[hoverIdx], ...positions[hoverIdx] };
  const tooltipFlip = hover !== null && hover.px > plotRight - 250;

  return (
    <div ref={containerRef} className="relative w-full select-none">
      <svg
        ref={svgRef}
        width={width}
        height={HEIGHT}
        role="img"
        aria-label="Blended price versus ELO for models in the latest snapshot"
        onPointerMove={handleMove}
        onPointerLeave={() => setHoverIdx(null)}
        className="block"
      >
        {/* y gridlines + ticks */}
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

        {/* x gridlines + ticks (log decades) */}
        {xTicks.map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={MARGIN.top}
              y2={plotBottom}
              stroke="#1E1B1D"
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={HEIGHT - 14}
              textAnchor="middle"
              className="font-mono"
              fontSize={10}
              fill="#555555"
            >
              {`$${t}`}
            </text>
          </g>
        ))}
        <text
          x={(MARGIN.left + plotRight) / 2}
          y={HEIGHT - 1}
          textAnchor="middle"
          className="font-mono"
          fontSize={9}
          fill="#555555"
        >
          blended price, USD per million tokens (log scale)
        </text>

        {/* frontier line under the dots */}
        <path
          d={frontierPath}
          fill="none"
          stroke={CORAL}
          strokeWidth={1}
          strokeDasharray="4 4"
          strokeOpacity={0.8}
        />

        {/* dots */}
        {points.map((p, i) => {
          const isHover = hoverIdx === i;
          return (
            <circle
              key={p.model}
              cx={positions[i].px}
              cy={positions[i].py}
              r={p.frontier ? 4 : 3}
              fill={p.color}
              fillOpacity={isHover ? 1 : p.frontier ? 0.95 : 0.55}
              stroke={p.frontier ? CORAL : isHover ? "#F2F2F2" : "none"}
              strokeWidth={p.frontier ? 1.5 : isHover ? 1 : 0}
            />
          );
        })}

        {/* frontier labels */}
        {showFrontierLabels &&
          frontierLabels.map(({ i, lx, ly, anchor }) => (
            <text
              key={points[i].model}
              x={lx}
              y={ly}
              textAnchor={anchor}
              className="font-mono"
              fontSize={10}
              fill={hoverIdx === i ? "#F2F2F2" : "#8C8C8C"}
            >
              {points[i].model}
            </text>
          ))}
      </svg>

      {/* tooltip */}
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-xl border border-cardborder bg-card px-3.5 py-3 shadow-[0_8px_28px_rgba(0,0,0,0.5)]"
          style={{
            top: Math.min(Math.max(hover.py - 56, 4), plotBottom - 110),
            left: tooltipFlip ? undefined : hover.px + 14,
            right: tooltipFlip ? width - hover.px + 14 : undefined,
            minWidth: 210,
          }}
        >
          <div className="flex items-center gap-2">
            <ProviderLogo provider={hover.p.provider} size={16} />
            <span className="max-w-[210px] truncate text-xs font-medium text-fg">
              {hover.p.model}
            </span>
          </div>
          <div className="mt-0.5 text-xs text-muted">{hover.p.provider}</div>
          <ul className="mt-2 space-y-1 border-t border-hairline pt-2">
            <li className="flex items-baseline justify-between gap-6">
              <span className="text-xs text-muted">Blended</span>
              <span className="font-mono text-xs text-fg">
                {formatPrice(hover.p.price)} / M
              </span>
            </li>
            <li className="flex items-baseline justify-between gap-6">
              <span className="text-xs text-muted">Input</span>
              <span className="font-mono text-xs text-fg">
                {formatPrice(hover.p.inputPrice)} / M
              </span>
            </li>
            <li className="flex items-baseline justify-between gap-6">
              <span className="text-xs text-muted">Output</span>
              <span className="font-mono text-xs text-fg">
                {formatPrice(hover.p.outputPrice)} / M
              </span>
            </li>
            <li className="flex items-baseline justify-between gap-6">
              <span className="text-xs text-muted">ELO</span>
              <span className="font-mono text-xs" style={{ color: hover.p.color }}>
                {Math.round(hover.p.elo)}
              </span>
            </li>
          </ul>
          {hover.p.frontier && (
            <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.18em] text-coral">
              efficient frontier
            </div>
          )}
        </div>
      )}
    </div>
  );
}

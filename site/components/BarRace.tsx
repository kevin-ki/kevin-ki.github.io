"use client";

import { useEffect, useMemo, useState } from "react";
import type { BarRaceData } from "@/lib/data";
import { ProviderLogo } from "./ProviderLogo";

interface BarRaceProps {
  data: BarRaceData;
}

const ROW_H = 38;
const BAR_H = 22;
const VISIBLE_ROWS = 12;
const TICK_MS = 600;
const EASE = "450ms cubic-bezier(0.25, 0.1, 0.25, 1)";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function formatDateLong(iso: string) {
  const [y, m, d] = iso.split("-").map(Number);
  return `${MONTHS[m - 1]} ${d}, ${y}`;
}

function PlayIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden>
      <path d="M2.5 1.5 L10.5 6 L2.5 10.5 Z" fill="currentColor" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden>
      <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
      <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
    </svg>
  );
}

export function BarRace({ data }: BarRaceProps) {
  const { models, frames } = data;
  const lastIdx = frames.length - 1;
  // Default to the latest snapshot so the section is informative untouched.
  const [idx, setIdx] = useState(lastIdx);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setIdx((i) => {
        if (i >= lastIdx) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [playing, lastIdx]);

  const frame = frames[idx];

  const { rankByModel, eloByModel, lo, span } = useMemo(() => {
    const rankByModel = new Map<number, number>();
    const eloByModel = new Map<number, number>();
    let min = Infinity;
    let max = -Infinity;
    frame.entries.forEach((e, rank) => {
      rankByModel.set(e.m, rank);
      eloByModel.set(e.m, e.elo);
      if (e.elo < min) min = e.elo;
      if (e.elo > max) max = e.elo;
    });
    // The x domain spans the current frame (min to max visible ELO) so the
    // differences between the top models stay readable.
    return { rankByModel, eloByModel, lo: min, span: max - min || 1 };
  }, [frame]);

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Pressing play on the final frame restarts the race from the beginning.
    if (idx >= lastIdx) setIdx(0);
    setPlaying(true);
  }

  function handleScrub(e: React.ChangeEvent<HTMLInputElement>) {
    setPlaying(false); // dragging the slider pauses playback
    setIdx(Number(e.target.value));
  }

  return (
    <div>
      {/* controls */}
      <div className="flex items-center gap-4 border-b border-hairline pb-4">
        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? "Pause" : "Play"}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-coral/40 bg-coral/10 text-coral transition-colors hover:bg-coral/20"
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <input
          type="range"
          min={0}
          max={lastIdx}
          step={1}
          value={idx}
          onChange={handleScrub}
          aria-label="Snapshot date"
          aria-valuetext={formatDateLong(frame.date)}
          className="h-1 min-w-0 flex-1 cursor-pointer accent-coral"
        />
        <span className="hidden w-44 shrink-0 text-right font-mono text-xs text-faint sm:inline">
          snapshot {String(idx + 1).padStart(2, "0")} of {frames.length}
        </span>
      </div>

      {/* current date */}
      <div className="flex items-baseline justify-between pb-5 pt-4">
        <span className="font-mono text-sm text-fg sm:text-base">
          {formatDateLong(frame.date)}
        </span>
        <span className="section-label">Top 15 by ELO</span>
      </div>

      {/* bars */}
      <div
        className="relative overflow-hidden"
        style={{ height: VISIBLE_ROWS * ROW_H }}
      >
        {models.map((m, i) => {
          const rank = rankByModel.get(i);
          const visible = rank !== undefined;
          const elo = eloByModel.get(i);
          const widthPct = visible
            ? 10 + 90 * (((elo as number) - lo) / span)
            : 0;
          return (
            <div
              key={m.model}
              aria-hidden={!visible}
              className="absolute inset-x-0 top-0 flex items-center gap-3"
              style={{
                height: ROW_H,
                transform: `translateY(${
                  visible ? (rank as number) * ROW_H : VISIBLE_ROWS * ROW_H + 10
                }px)`,
                opacity: visible ? 1 : 0,
                pointerEvents: "none",
                transition: `transform ${EASE}, opacity ${EASE}`,
              }}
            >
              <div className="flex w-32 shrink-0 items-center gap-2 sm:w-56">
                <ProviderLogo provider={m.provider} size={15} />
                <span className="truncate font-mono text-[11px] text-muted">
                  {m.model}
                </span>
              </div>
              <div className="flex min-w-0 flex-1 items-center gap-2.5">
                <div
                  className="shrink-0 rounded-[5px]"
                  style={{
                    height: BAR_H,
                    width: `calc((100% - 44px) * ${(widthPct / 100).toFixed(4)})`,
                    background: m.color,
                    opacity: 0.9,
                    transition: `width ${EASE}`,
                  }}
                />
                <span className="shrink-0 font-mono text-xs text-fg">
                  {elo === undefined ? "" : Math.round(elo)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

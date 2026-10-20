import { useState } from "react";
import data from "../chartData.json";
import { useReveal } from "../hooks";

type Tri = [number, number, number];
const SERIES: { key: keyof typeof data.curve; label: string; color: string; width: number }[] = [
  { key: "ragx", label: "ragX", color: "url(#ragx-line)", width: 3.2 },
  { key: "selfask", label: "Self-Ask", color: "#f59e0b", width: 1.8 },
  { key: "react", label: "ReAct", color: "#14b8a6", width: 1.8 },
  { key: "ss", label: "Single-shot (strongest)", color: "#94a3b8", width: 1.8 },
];
const SWATCH: Record<string, string> = {
  ragx: "linear-gradient(90deg,#38bdf8,#818cf8,#c084fc)",
  selfask: "#f59e0b",
  react: "#14b8a6",
  ss: "#94a3b8",
};

const W = 640;
const H = 340;
const M = { l: 44, r: 92, t: 18, b: 46 };
// end labels of series whose last points sit close together are nudged apart
const NUDGE: Record<string, number> = { react: -6, selfask: 7 };
const Y0 = 36;
const Y1 = 56;
const x = (i: number) => M.l + (i * (W - M.l - M.r)) / 4;
const y = (v: number) => M.t + ((Y1 - v) * (H - M.t - M.b)) / (Y1 - Y0);

export default function ScaleChart() {
  const { ref, shown } = useReveal<HTMLDivElement>(0.3);
  const [hover, setHover] = useState<number | null>(null);
  const tiers = data.tiers;
  const curve = data.curve as unknown as Record<string, Tri[]>;
  // the paper's gaps, from the unrounded means (the rounded points differ by 0.1 at 21M)
  const gapStart = "5.4";
  const gapEnd = "9.0";
  return (
    <div ref={ref} className="card p-5 sm:p-7">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-white">The lead grows with the corpus</div>
          <div className="text-xs text-slate-500">Multi-hop F1 · nested Wikipedia tiers · eight models · dense index · 95% bands</div>
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-slate-400">
          {SERIES.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="h-1.5 w-4 rounded-full" style={{ background: SWATCH[s.key] }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-4 w-full select-none" role="img" aria-label="F1 against corpus size">
        <defs>
          <linearGradient id="ragx-line" x1="0" x2="1">
            <stop offset="0" stopColor="#38bdf8" />
            <stop offset="0.5" stopColor="#818cf8" />
            <stop offset="1" stopColor="#c084fc" />
          </linearGradient>
          <linearGradient id="ragx-band" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#818cf8" stopOpacity="0.28" />
            <stop offset="1" stopColor="#38bdf8" stopOpacity="0.08" />
          </linearGradient>
        </defs>
        {[36, 40, 44, 48, 52, 56].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="stroke-slate-500/15" />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="fill-slate-500 text-[11px]">{v}</text>
          </g>
        ))}
        {tiers.map((t, i) => (
          <text key={t} x={x(i)} y={H - 20} textAnchor="middle" className="fill-slate-500 text-[11px]">{t}</text>
        ))}
        <text x={(W - M.r + M.l) / 2} y={H - 2} textAnchor="middle" className="fill-slate-500 text-[10px]">
          passages in the corpus
        </text>
        {SERIES.map((s) => {
          const pts = curve[s.key];
          const band =
            pts.map((p, i) => `${x(i)},${y(p[2])}`).join(" ") +
            " " +
            pts.map((p, i) => `${x(4 - i)},${y(pts[4 - i][1])}`).join(" ");
          const d = pts.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p[0])}`).join(" ");
          return (
            <g key={s.key}>
              <polygon
                points={band}
                fill={s.key === "ragx" ? "url(#ragx-band)" : s.color}
                opacity={s.key === "ragx" ? 1 : 0.1}
                className={`transition-opacity duration-1000 ${shown ? "" : "opacity-0"}`}
              />
              <path
                d={d}
                fill="none"
                stroke={s.color}
                strokeWidth={s.width}
                strokeLinecap="round"
                strokeLinejoin="round"
                pathLength={1}
                strokeDasharray="1"
                strokeDashoffset={shown ? 0 : 1}
                style={{ transition: "stroke-dashoffset 1.6s cubic-bezier(.2,.8,.2,1)" }}
              />
              {pts.map((p, i) => (
                <circle
                  key={i}
                  cx={x(i)}
                  cy={y(p[0])}
                  r={hover === i ? 5 : s.key === "ragx" ? 3.8 : 2.8}
                  fill={s.key === "ragx" ? "#a5b4fc" : s.color}
                  className={`transition-all duration-500 ${shown ? "opacity-100" : "opacity-0"}`}
                  style={{ transitionDelay: `${0.25 * i + 0.4}s` }}
                />
              ))}
              <text
                x={W - M.r + 8}
                y={y(pts[4][0]) + 4 + (NUDGE[s.key] ?? 0)}
                className={`text-[11px] font-semibold transition-opacity duration-700 ${s.key === "ragx" ? "fill-indigo-300" : ""} ${shown ? "opacity-100" : "opacity-0"}`}
                fill={s.key === "ragx" ? undefined : s.color}
                style={{ transitionDelay: "1.4s" }}
              >
                {s.label.split(" ")[0]}
              </text>
            </g>
          );
        })}
        {[
          [0, gapStart],
          [4, gapEnd],
        ].map(([i, g]) => {
          const xi = x(Number(i));
          const a = y(curve.ragx[Number(i)][0]);
          const b = y(curve.ss[Number(i)][0]);
          return (
            <g key={String(i)} className={`transition-opacity duration-700 ${shown ? "opacity-100" : "opacity-0"}`} style={{ transitionDelay: "1.6s" }}>
              <line x1={xi + (i ? -16 : 16)} x2={xi + (i ? -16 : 16)} y1={a + 6} y2={b - 6} stroke="#818cf8" strokeWidth="1.5" markerEnd="" />
              <text x={xi + (i ? -22 : 22)} y={(a + b) / 2 + 4} textAnchor={i ? "end" : "start"} className="fill-indigo-300 text-[12px] font-bold">
                +{g}
              </text>
            </g>
          );
        })}
        {tiers.map((_, i) => (
          <rect
            key={i}
            x={x(i) - (W - M.l - M.r) / 8}
            y={M.t}
            width={(W - M.l - M.r) / 4}
            height={H - M.t - M.b}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hover !== null && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} className="stroke-slate-400/40" strokeDasharray="3 3" />
            <g transform={`translate(${Math.min(x(hover) + 10, W - M.r - 150)}, ${M.t + 6})`}>
              <rect width="150" height="84" rx="8" className="fill-slate-900/90 stroke-white/10" />
              <text x="10" y="18" className="fill-white text-[11px] font-semibold">{tiers[hover]} passages</text>
              {SERIES.map((s, k) => (
                <text key={s.key} x="10" y={34 + k * 13} className="fill-slate-300 text-[10.5px]">
                  {s.label.split(" ")[0]}: {curve[s.key][hover][0].toFixed(1)} F1
                </text>
              ))}
            </g>
          </g>
        )}
      </svg>
    </div>
  );
}

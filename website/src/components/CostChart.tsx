import { useState } from "react";
import data from "../chartData.json";
import { useReveal } from "../hooks";

const W = 640;
const H = 340;
const M = { l: 44, r: 20, t: 18, b: 40 };
const X1 = 28;
const Y0 = 26;
const Y1 = 54;
const x = (v: number) => M.l + (v * (W - M.l - M.r)) / X1;
const y = (v: number) => M.t + ((Y1 - v) * (H - M.t - M.b)) / (Y1 - Y0);

// label placement (dx, dy, anchor) so no two labels collide
const PLACE: Record<string, [number, number, "start" | "end" | "middle"]> = {
  "Closed-book": [8, 4, "start"],
  "BM25@16": [8, 12, "start"],
  "Dense@48": [8, 4, "start"],
  "Dense@160 (long context)": [-8, -8, "end"],
  "Rerank@16": [8, 2, "start"],
  IRCoT: [8, 6, "start"],
  "Self-Ask": [-8, -6, "end"],
  "ReAct (8 rounds)": [-8, 14, "end"],
  "ragX-lexical": [8, 10, "start"],
  "ragX-fast": [-8, 4, "end"],
  "ragX-fused": [8, -4, "start"],
  "ragX-core": [-8, 10, "end"],
  ragX: [8, 4, "start"],
  "ragX-rerank": [8, -4, "start"],
};

export default function CostChart() {
  const { ref, shown } = useReveal<HTMLDivElement>(0.3);
  const [hover, setHover] = useState<string | null>(null);
  const pts = data.cost;
  return (
    <div ref={ref} className="card p-5 sm:p-7">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-white">Accuracy against cost at 21M passages</div>
          <div className="text-xs text-slate-500">Multi-hop F1 · eight models · three datasets · prompt tokens per question</div>
        </div>
        <div className="flex gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-gradient-to-r from-sky-400 to-violet-400" /> ragX variants
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-slate-400" /> baselines
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-4 w-full select-none" role="img" aria-label="F1 against prompt tokens">
        <defs>
          <radialGradient id="ragx-dot">
            <stop offset="0" stopColor="#e0e7ff" />
            <stop offset="0.55" stopColor="#818cf8" />
            <stop offset="1" stopColor="#c084fc" />
          </radialGradient>
          <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="3.2" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {[26, 32, 38, 44, 50].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="stroke-slate-500/15" />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="fill-slate-500 text-[11px]">{v}</text>
          </g>
        ))}
        {[0, 5, 10, 15, 20, 25].map((v) => (
          <text key={v} x={x(v)} y={H - 18} textAnchor="middle" className="fill-slate-500 text-[11px]">{v}k</text>
        ))}
        <text x={(W + M.l) / 2} y={H - 2} textAnchor="middle" className="fill-slate-500 text-[10px]">
          prompt tokens per question
        </text>
        {pts.map((p, k) => {
          const ours = p.label.startsWith("ragX");
          const [dx, dy, anchor] = PLACE[p.label] ?? [8, 4, "start"];
          const on = hover === p.label;
          return (
            <g
              key={p.label}
              onMouseEnter={() => setHover(p.label)}
              onMouseLeave={() => setHover(null)}
              className={`cursor-default transition-all duration-700 ${shown ? "opacity-100" : "opacity-0"}`}
              style={{ transitionDelay: `${0.06 * k}s` }}
            >
              <circle
                cx={x(p.ktok)}
                cy={y(p.f1)}
                r={on ? 8 : ours ? 6 : 4.5}
                fill={ours ? "url(#ragx-dot)" : "#94a3b8"}
                fillOpacity={ours ? 1 : 0.7}
                filter={ours ? "url(#glow)" : undefined}
                style={{ transition: "r .2s" }}
              />
              <text
                x={x(p.ktok) + dx}
                y={y(p.f1) + dy}
                textAnchor={anchor}
                className={`text-[10.5px] ${ours ? "fill-indigo-300 font-semibold" : "fill-slate-500"}`}
              >
                {p.label.replace(" (long context)", "").replace(" (8 rounds)", "")}
              </text>
            </g>
          );
        })}
        <g className={`transition-opacity duration-700 ${shown ? "opacity-100" : "opacity-0"}`} style={{ transitionDelay: "1s" }}>
          <path d={`M${x(19.5)} ${y(29)} H${x(27.4)}`} stroke="#818cf8" strokeWidth="1.4" markerEnd="url(#arrow)" />
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 0 L10 5 L0 10 z" fill="#818cf8" />
            </marker>
          </defs>
          <text x={x(27.4)} y={y(29) - 8} textAnchor="end" className="fill-indigo-300 text-[10.5px] font-semibold">
            DCI-Agent-Lite: 22× ragX's prompt tokens
          </text>
        </g>
        {hover && (() => {
          const p = pts.find((q) => q.label === hover)!;
          const tx = Math.min(x(p.ktok) + 12, W - 180);
          return (
            <g pointerEvents="none" transform={`translate(${tx}, ${Math.max(y(p.f1) - 60, 6)})`}>
              <rect width="168" height="44" rx="8" className="fill-slate-900/90 stroke-white/10" />
              <text x="10" y="18" className="fill-white text-[11px] font-semibold">{p.label}</text>
              <text x="10" y="34" className="fill-slate-300 text-[10.5px]">
                {p.f1.toFixed(1)} F1 · {p.ktok.toFixed(1)}k prompt tokens
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

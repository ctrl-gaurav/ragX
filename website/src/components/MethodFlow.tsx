import { useReveal } from "../hooks";

const STAGES = [
  {
    n: "01",
    title: "Forage loop",
    body: "An agent RANKs the corpus and READs whole passages for up to ten turns. Its next query is written after reading, so it can name what the question never did.",
    chips: ["RANK", "READ", "≤ 10 turns"],
    grad: "from-sky-400 to-indigo-400",
  },
  {
    n: "02",
    title: "Slot chain",
    body: "The question becomes up to four sub-questions with #N slots. Each hop is retrieved with the earlier answers filled in, a second, separate route to the bridge.",
    chips: ["#1 → #2", "4 passages / hop", "1 short call"],
    grad: "from-indigo-400 to-violet-400",
  },
  {
    n: "03",
    title: "Compose",
    body: "One call answers over the read passages, the hop passages, the other surfaced ones and the question's own top 48. Tool output stays fenced as untrusted data.",
    chips: ["1 call", "whole passages", "fenced"],
    grad: "from-violet-400 to-fuchsia-400",
  },
];

function Connector({ id, delay }: { id: string; delay: number }) {
  return (
    <svg viewBox="0 0 100 20" className="hidden h-6 w-full md:block" preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id={`g-${id}`} x1="0" x2="1">
          <stop offset="0" stopColor="#38bdf8" />
          <stop offset="1" stopColor="#c084fc" />
        </linearGradient>
      </defs>
      <path id={`p-${id}`} d="M0 10 H100" stroke={`url(#g-${id})`} strokeWidth="1.2" strokeDasharray="3 3" fill="none" opacity="0.6" />
      {[0, 0.6, 1.2].map((b) => (
        <circle key={b} r="2.2" fill="#a5b4fc">
          <animateMotion dur="1.8s" begin={`${delay + b}s`} repeatCount="indefinite">
            <mpath href={`#p-${id}`} />
          </animateMotion>
        </circle>
      ))}
    </svg>
  );
}

function ForageOrbit() {
  return (
    <svg viewBox="0 0 120 120" className="h-28 w-28" aria-hidden>
      <defs>
        <linearGradient id="orbit" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#38bdf8" />
          <stop offset="1" stopColor="#818cf8" />
        </linearGradient>
      </defs>
      <circle cx="60" cy="60" r="42" stroke="url(#orbit)" strokeOpacity="0.35" strokeWidth="1.5" fill="none" />
      <path id="orbit-path" d="M60 18 A42 42 0 1 1 59.9 18" fill="none" />
      <g className="font-mono" fontSize="10" fontWeight="700">
        <rect x="38" y="8" width="44" height="18" rx="6" className="fill-sky-500/20 stroke-sky-400/60" />
        <text x="60" y="21" textAnchor="middle" className="fill-sky-300">RANK</text>
        <rect x="38" y="94" width="44" height="18" rx="6" className="fill-emerald-500/20 stroke-emerald-400/60" />
        <text x="60" y="107" textAnchor="middle" className="fill-emerald-300">READ</text>
      </g>
      <circle r="4" fill="#e0e7ff">
        <animateMotion dur="2.4s" repeatCount="indefinite">
          <mpath href="#orbit-path" />
        </animateMotion>
      </circle>
      <text x="60" y="64" textAnchor="middle" fontSize="11" className="fill-slate-400 font-mono">×10</text>
    </svg>
  );
}

function SlotChain() {
  return (
    <div className="flex h-28 flex-col justify-center gap-2 font-mono text-[11px]">
      {["Who voices Jarvis? → Paul Bettany", "Spouse of #1? → Jennifer Connelly"].map((t, i) => (
        <div
          key={t}
          className="animate-fadeup rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 text-slate-300"
          style={{ animationDelay: `${0.3 + i * 0.4}s` }}
        >
          <span className="text-indigo-300">#{i + 1}</span> {t}
        </div>
      ))}
    </div>
  );
}

function ComposeStack() {
  return (
    <div className="relative flex h-28 items-center justify-center">
      {[0, 1, 2, 3].map((i) => (
        <div
          key={i}
          className="absolute h-14 w-28 rounded-lg border border-white/10 bg-white/[0.05]"
          style={{ transform: `translate(${i * 7 - 10}px, ${-i * 6 + 8}px)` }}
        />
      ))}
      <div className="absolute translate-x-12 translate-y-3 rounded-lg bg-gradient-to-r from-violet-500 to-fuchsia-500 px-2.5 py-1 font-mono text-[11px] font-bold text-white shadow-lg shadow-fuchsia-500/30">
        answer ✓
      </div>
    </div>
  );
}

export default function MethodFlow() {
  const { ref, shown } = useReveal<HTMLDivElement>();
  const visuals = [<ForageOrbit key="f" />, <SlotChain key="s" />, <ComposeStack key="c" />];
  return (
    <div ref={ref} className="grid items-stretch gap-4 md:grid-cols-[1fr_48px_1fr_48px_1fr]">
      {STAGES.map((s, i) => (
        <div key={s.n} className="contents">
          <div
            className={`card glow-border relative flex flex-col p-6 transition-all duration-700 ${
              shown ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"
            }`}
            style={{ transitionDelay: `${i * 150}ms` }}
          >
            <div className="flex items-center justify-between">
              <span className={`bg-gradient-to-r ${s.grad} bg-clip-text font-mono text-sm font-bold text-transparent`}>
                {s.n}
              </span>
              <span className={`h-1.5 w-12 rounded-full bg-gradient-to-r ${s.grad}`} />
            </div>
            <div className="my-3 flex justify-center">{visuals[i]}</div>
            <h3 className="text-lg font-bold text-white">{s.title}</h3>
            <p className="mt-2 flex-1 text-sm leading-relaxed text-slate-400">{s.body}</p>
            <div className="mt-4 flex flex-wrap gap-1.5">
              {s.chips.map((c) => (
                <span key={c} className="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                  {c}
                </span>
              ))}
            </div>
          </div>
          {i < STAGES.length - 1 && (
            <div className="flex items-center">
              <Connector id={`c${i}`} delay={i * 0.4} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

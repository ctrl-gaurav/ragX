import { useEffect, useState } from "react";

type Line = { kind: string; text: string; note?: string; accent?: boolean };

// The paper's running example: the question names Iron Man and Jarvis, never the bridge (Paul Bettany).
const SCRIPT: Line[] = [
  { kind: "plan", text: "Foraging for the question" },
  { kind: "rank", text: "Jarvis Iron Man voice actor" },
  { kind: "read", text: "Iron Man (2008 film):41-47" },
  { kind: "rank", text: "Paul Bettany spouse", note: "named after reading", accent: true },
  { kind: "read", text: "Paul Bettany:1-6" },
  { kind: "hop", text: "Who voices Jarvis in Iron Man? → Paul Bettany" },
  { kind: "hop", text: "Who is the spouse of #1? → Jennifer Connelly" },
  { kind: "backfill", text: "48 passages from the question's own ranking" },
  { kind: "compose", text: "one call over 20 whole passages" },
];

const COLOR: Record<string, string> = {
  plan: "text-sky-400",
  rank: "text-fuchsia-400",
  read: "text-emerald-400",
  hop: "text-cyan-300",
  backfill: "text-indigo-300",
  compose: "text-amber-300",
};

export default function Terminal() {
  const [n, setN] = useState(0);
  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setN(SCRIPT.length + 1);
      return;
    }
    const id = window.setInterval(
      () => setN((v) => (v > SCRIPT.length + 6 ? 0 : v + 1)),
      720,
    );
    return () => window.clearInterval(id);
  }, []);
  const done = n > SCRIPT.length;
  return (
    <div className="code-surface card glow-border overflow-hidden shadow-2xl shadow-indigo-500/10">
      <div className="flex items-center gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
        <span className="ml-2 font-mono text-xs text-slate-500">ragx · live trace</span>
      </div>
      <div className="min-h-[23rem] p-5 font-mono text-[12.5px] leading-relaxed">
        <div className="text-slate-400">
          <span className="text-emerald-400">$</span> ragx ask{" "}
          <span className="tok-str">"Who is the spouse of the actor who plays Jarvis in Iron Man?"</span>{" "}
          <span className="text-slate-500">-v ragx</span>
        </div>
        <div className="mt-3 space-y-1">
          {SCRIPT.slice(0, n).map((l, i) => (
            <div key={i} className="flex animate-fadeup gap-3">
              <span className={`w-16 shrink-0 ${COLOR[l.kind] ?? "text-slate-400"}`}>{l.kind}</span>
              <span className={l.accent ? "font-semibold text-white" : "text-slate-300"}>{l.text}</span>
              {l.note && (
                <span className="hidden rounded-full bg-gradient-to-r from-sky-400/20 to-violet-400/20 px-2 text-[10.5px] text-violet-300 sm:inline">
                  {l.note}
                </span>
              )}
            </div>
          ))}
        </div>
        {done && (
          <div className="mt-4 animate-fadeup rounded-xl border border-white/10 bg-white/[0.03] p-3">
            <div className="text-[11px] uppercase tracking-widest text-slate-500">answer</div>
            <div className="mt-1 text-base font-semibold text-white">Jennifer Connelly</div>
            <div className="mt-1 text-[11.5px] text-sky-400">↳ Paul Bettany:3-4 · grounded</div>
          </div>
        )}
        {!done && <span className="mt-2 inline-block h-4 w-2 animate-blink bg-slate-400" />}
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import CodeBlock from "./components/CodeBlock";
import CostChart from "./components/CostChart";
import MethodFlow from "./components/MethodFlow";
import ScaleChart from "./components/ScaleChart";
import Terminal from "./components/Terminal";
import { useCountUp, useReveal } from "./hooks";

const GH = "https://github.com/ctrl-gaurav/ragX";
const PYPI = "https://pypi.org/project/pyragx/";
const DOCS = `${import.meta.env.BASE_URL}docs/`;

function Wordmark({ className = "" }: { className?: string }) {
  return (
    <span className={className}>
      rag<span className="gradient-text gradient-anim">X</span>
    </span>
  );
}

function Reveal({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const { ref, shown } = useReveal();
  return (
    <div
      ref={ref}
      className={`${className} transition-all duration-700 ${shown ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

function SectionHead({ kicker, title, sub }: { kicker: string; title: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <Reveal className="mx-auto mb-12 max-w-3xl text-center">
      <div className="mb-3 font-mono text-xs font-semibold uppercase tracking-[0.25em] text-accent-sky">{kicker}</div>
      <h2 className="text-3xl font-extrabold tracking-tight text-white md:text-5xl">{title}</h2>
      {sub && <p className="mt-5 text-lg leading-relaxed text-slate-400">{sub}</p>}
    </Reveal>
  );
}

function ThemeToggle() {
  const [light, setLight] = useState(() => typeof document !== "undefined" && document.documentElement.classList.contains("light"));
  const toggle = () => {
    const next = !light;
    setLight(next);
    document.documentElement.classList.toggle("light", next);
    document.documentElement.classList.toggle("dark", !next);
    try {
      localStorage.setItem("ragx-theme", next ? "light" : "dark");
    } catch {
      /* storage may be unavailable */
    }
  };
  return (
    <button
      onClick={toggle}
      aria-label={light ? "Switch to dark theme" : "Switch to light theme"}
      className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition hover:bg-white/10"
    >
      {light ? (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
      ) : (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      )}
    </button>
  );
}

function GitHubIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

const NAV = [
  ["Method", "#method"],
  ["Results", "#results"],
  ["Variants", "#variants"],
  ["Quickstart", "#quickstart"],
] as const;

function Nav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return (
    <header className={`sticky top-0 z-50 border-b transition-colors ${scrolled ? "border-white/10 bg-ink-950/75 backdrop-blur-xl" : "border-transparent"}`}>
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <a href="#top" className="text-2xl font-black tracking-tight text-white">
          <Wordmark />
        </a>
        <nav className="hidden items-center gap-8 text-sm font-medium text-slate-400 md:flex">
          {NAV.map(([l, h]) => (
            <a key={h} className="transition hover:text-white" href={h}>{l}</a>
          ))}
          <a className="transition hover:text-white" href={DOCS}>Docs</a>
        </nav>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <a href={GH} className="hidden items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3.5 py-2 text-sm font-semibold text-slate-200 transition hover:bg-white/10 sm:flex">
            <GitHubIcon /> GitHub
          </a>
          <button className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/5 text-slate-300 md:hidden" onClick={() => setOpen(!open)} aria-label="Menu">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d={open ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"} /></svg>
          </button>
        </div>
      </div>
      {open && (
        <div className="border-t border-white/10 bg-ink-950/95 px-6 py-4 md:hidden">
          {[...NAV, ["Docs", DOCS], ["GitHub", GH]].map(([l, h]) => (
            <a key={l} href={h} onClick={() => setOpen(false)} className="block py-2 text-slate-300">{l}</a>
          ))}
        </div>
      )}
    </header>
  );
}

function InstallPill() {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText("pip install pyragx").then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      },
      () => undefined,
    );
  };
  return (
    <button onClick={copy} className="code-surface group flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-5 py-3 font-mono text-sm transition hover:border-white/20">
      <span className="text-emerald-400">$</span>
      <span className="text-slate-200">pip install pyragx</span>
      <span className="text-xs text-slate-500 transition group-hover:text-slate-300">{copied ? "copied ✓" : "copy"}</span>
    </button>
  );
}

function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="orb left-[-12%] top-[-18%] h-[36rem] w-[36rem] animate-aurora bg-accent-sky/25" />
      <div className="orb right-[-10%] top-[-8%] h-[30rem] w-[30rem] animate-aurora bg-accent-violet/25 [animation-delay:-6s]" />
      <div className="orb left-[35%] top-[30%] h-[22rem] w-[22rem] animate-aurora bg-accent-indigo/20 [animation-delay:-12s]" />
      <div className="grid-bg absolute inset-0" />
      <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-6 pb-20 pt-14 lg:grid-cols-[1.05fr_1fr] lg:pb-28 lg:pt-20">
        <div className="animate-fadeup">
          <a href={DOCS} className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 py-1 pl-1 pr-3 text-xs font-medium text-slate-300 transition hover:bg-white/10">
            <span className="rounded-full bg-gradient-to-r from-sky-400 to-violet-400 px-2 py-0.5 text-[10.5px] font-bold text-ink-950">v0.0.1</span>
            Open source · Apache 2.0 · read the docs →
          </a>
          <h1 className="text-5xl font-black leading-[1.02] tracking-tight text-white md:text-7xl">
            Agentic retrieval
            <br />
            <span className="gradient-text gradient-anim">at scale.</span>
          </h1>
          <p className="mt-7 max-w-xl text-lg leading-relaxed text-slate-400">
            On millions of documents, single-shot RAG misses the evidence a question does not name.{" "}
            <Wordmark className="font-semibold text-slate-200" /> names it: an agent ranks and reads passages, a chain of
            sub-questions resolves each hop, and one call answers with exact citations. No training, any LLM.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <a href={DOCS} className="btn-grad rounded-xl px-6 py-3 font-semibold text-ink-950 shadow-lg shadow-indigo-500/20">
              Get started
            </a>
            <InstallPill />
          </div>
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-slate-500">
            <a href={GH} className="flex items-center gap-1.5 transition hover:text-slate-200"><GitHubIcon /> GitHub</a>
            <a href={PYPI} className="transition hover:text-slate-200">PyPI</a>
            <span className="flex items-center gap-1.5">
              Paper <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10.5px]">coming soon</span>
            </span>
          </div>
        </div>
        <div className="animate-fadeup [animation-delay:0.15s]">
          <Terminal />
          <p className="mt-3 text-center text-xs text-slate-500">An illustrative trace of the paper's running example.</p>
        </div>
      </div>
    </section>
  );
}

const STATS = [
  { value: 8.1, fmt: (v: number) => `+${v.toFixed(1)}`, unit: "F1", label: "over the strongest single-shot baseline", sub: "on all of Wikipedia, 21M passages, eight models" },
  { value: 9.0, fmt: (v: number) => `+5.4 → +${v.toFixed(1)}`, unit: "F1", label: "a lead that grows with the corpus", sub: "from 240K to 21M passages, dense index" },
  { value: 22, fmt: (v: number) => `${Math.round(v)}×`, unit: "", label: "fewer prompt tokens", sub: "than DCI-Agent-Lite, an agent that greps the raw corpus" },
  { value: 27, fmt: (v: number) => `${Math.round(v)}`, unit: "models", label: "benchmarked on five QA datasets", sub: "leads single-shot retrieval on 17 of 21 open-weight models" },
];

function Stat({ s, i }: { s: (typeof STATS)[number]; i: number }) {
  const { ref, shown } = useReveal<HTMLDivElement>();
  const v = useCountUp(s.value, shown, 1300);
  return (
    <div ref={ref} className="bg-ink-950/60 p-7 transition hover:bg-white/[0.03]" style={{ transitionDelay: `${i * 80}ms` }}>
      <div className="flex items-baseline gap-1.5">
        <span className="gradient-text text-4xl font-black tracking-tight">{s.fmt(v)}</span>
        {s.unit && <span className="text-sm font-semibold text-slate-400">{s.unit}</span>}
      </div>
      <div className="mt-2 text-sm font-semibold text-white">{s.label}</div>
      <div className="mt-1.5 text-xs leading-relaxed text-slate-500">{s.sub}</div>
    </div>
  );
}

function Stats() {
  return (
    <section className="relative mx-auto max-w-7xl px-6">
      <div className="grid gap-px overflow-hidden rounded-2xl border border-white/10 bg-white/5 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s, i) => (
          <Stat key={s.label} s={s} i={i} />
        ))}
      </div>
    </section>
  );
}

function Problem() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-28">
      <div className="grid items-center gap-14 lg:grid-cols-2">
        <Reveal>
          <div className="mb-3 font-mono text-xs font-semibold uppercase tracking-[0.25em] text-accent-sky">The problem</div>
          <h2 className="text-3xl font-extrabold tracking-tight text-white md:text-5xl">
            Scale hides the evidence a question <span className="gradient-text">does not name</span>.
          </h2>
          <p className="mt-6 text-lg leading-relaxed text-slate-400">
            Single-shot retrieval sends the question itself as the query. That finds what the question names. On a
            growing corpus it misses the rest: from 240K to 21M passages, it loses unnamed evidence four times faster
            than named evidence. A larger embedder or a deep reranker does not change this; a query that names the
            missing entity does.
          </p>
        </Reveal>
        <Reveal delay={150}>
          <div className="card glow-border p-7">
            <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">question</div>
            <p className="mt-2 text-xl font-semibold leading-snug text-white">
              Who is the spouse of the actor who plays <span className="rounded bg-sky-400/15 px-1 text-sky-300">Jarvis</span> in{" "}
              <span className="rounded bg-sky-400/15 px-1 text-sky-300">Iron Man</span>?
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                <div className="text-xs font-semibold text-sky-300">named → easy to rank</div>
                <div className="mt-1 text-sm text-slate-400">Iron Man, Jarvis</div>
              </div>
              <div className="rounded-xl border border-fuchsia-400/30 bg-fuchsia-400/[0.06] p-4">
                <div className="text-xs font-semibold text-fuchsia-300">unnamed → the bridge</div>
                <div className="mt-1 text-sm text-slate-300">Paul Bettany, whose article holds the answer</div>
              </div>
            </div>
            <div className="mt-6 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className="font-mono text-2xl font-black text-slate-200">146,602</div>
              <div className="text-sm text-slate-400">
                the answer passage's rank for the question itself on all of Wikipedia. No query copied from the
                question can reach it.
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function Method() {
  return (
    <section id="method" className="relative border-y border-white/5 bg-white/[0.015] py-28">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHead
          kicker="How it works"
          title={
            <>
              Two routes to the bridge, <span className="gradient-text">one answer</span>
            </>
          }
          sub="ragX names missing evidence through two separate routes, a search-and-read agent and a chain of sub-questions, and composes once over everything they found."
        />
        <MethodFlow />
      </div>
    </section>
  );
}

const FINDINGS = [
  ["BM25 alone is enough to win", "ragX over a plain BM25 index beats every dense single-shot baseline at 21M passages, with no embedding index at all."],
  ["Any model, no training", "One config string switches between open-weight and hosted models; a ragX variant leads single-shot retrieval on 19 of 21 open-weight models."],
  ["Cited and fenced", "Every answer carries path:line citations checked against the source, and tool output is fenced as untrusted data against planted instructions."],
];

function Results() {
  return (
    <section id="results" className="mx-auto max-w-7xl px-6 py-28">
      <SectionHead
        kicker="Results"
        title={
          <>
            Built for the corpus you <span className="gradient-text">actually serve</span>
          </>
        }
        sub="Benchmarked on nested samples of Wikipedia from 240K to 21M passages, with 27 language models, against single-shot retrieval, ReAct, Self-Ask, IRCoT, DCI-Agent-Lite and Search-R1."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <Reveal>
          <ScaleChart />
        </Reveal>
        <Reveal delay={120}>
          <CostChart />
        </Reveal>
      </div>
      <div className="mt-6 grid gap-6 md:grid-cols-3">
        {FINDINGS.map(([t, b], i) => (
          <Reveal key={t} delay={i * 100}>
            <div className="card card-hover h-full p-6">
              <div className="text-base font-bold text-white">{t}</div>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{b}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

const VARIANTS = [
  { name: "ragx", tag: "full method", idx: "dense", text: "Forage loop, slot chain and a 48-passage backfill. The most complete pipeline.", extra: "dense" },
  { name: "ragx-rerank", tag: "most accurate", idx: "cross-encoder", text: "The dense top 100 reordered by bge-reranker-v2-m3.", extra: "dense" },
  { name: "ragx-lexical", tag: "default", idx: "BM25", text: "No embedding index, no GPU. Install and ask.", extra: "" },
  { name: "ragx-fused", tag: "hybrid", idx: "BM25 + dense", text: "Both rankings fused by reciprocal rank.", extra: "dense" },
  { name: "ragx-core", tag: "lean", idx: "dense", text: "Forage loop and slot chain without the backfill.", extra: "dense" },
  { name: "ragx-fast", tag: "cheapest", idx: "dense", text: "The forage loop alone, one compose call.", extra: "dense" },
  { name: "ragx-think", tag: "reasoning", idx: "dense", text: "ragx with thinking on in every call.", extra: "dense" },
];

function Variants() {
  return (
    <section id="variants" className="border-y border-white/5 bg-white/[0.015] py-28">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHead
          kicker="Variants"
          title={
            <>
              One method, <span className="gradient-text">seven settings</span>
            </>
          }
          sub="Pick the index you have and the budget you want; every variant shares the prompts and the frozen settings of the paper."
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {VARIANTS.map((v, i) => (
            <Reveal key={v.name} delay={i * 60}>
              <div className={`card card-hover glow-border h-full p-6 ${v.tag === "default" ? "ring-1 ring-indigo-400/40" : ""}`}>
                <div className="flex items-center justify-between">
                  <code className="font-mono text-sm font-bold text-white">{v.name}</code>
                  <span className={`rounded-full px-2 py-0.5 text-[10.5px] font-semibold ${v.tag === "default" ? "bg-gradient-to-r from-sky-400 to-violet-400 text-ink-950" : "border border-white/10 text-slate-400"}`}>
                    {v.tag}
                  </span>
                </div>
                <div className="mt-3 font-mono text-[11px] text-accent-sky">{v.idx}</div>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{v.text}</p>
                <div className="mt-4 font-mono text-[11px] text-slate-500">pip install {v.extra ? `"ragx[${v.extra}]"` : "ragx"}</div>
              </div>
            </Reveal>
          ))}
          <Reveal delay={420}>
            <a href={`${DOCS}guide/variants`} className="card card-hover flex h-full flex-col justify-center p-6 text-center">
              <div className="text-sm font-semibold text-white">Choosing a variant →</div>
              <div className="mt-1 text-xs text-slate-500">accuracy, cost and index trade-offs</div>
            </a>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

const SNIPPETS = {
  Python: {
    lang: "python" as const,
    title: "quickstart.py",
    code: `from ragx import RagX

rag = RagX("./docs")                      # ragx-lexical: BM25, no embedding index
answer = rag.ask("Who founded the university the author of Harbor Light attended?")

print(answer.text)                        # Thomas Wren
print(answer.sources)                     # ['ashford_university.md:1-5', 'thomas_wren.md:1-5']

# the full method, with a dense index and the backfill
rag = RagX("./docs", variant="ragx", model="openai:gpt-4o-mini")`,
  },
  CLI: {
    lang: "bash" as const,
    title: "terminal",
    code: `pip install pyragx                 # or: pip install "pyragx[dense]" for every variant
export OPENAI_API_KEY=...        # or any provider; local models work too

ragx ask "Who founded the university the author of Harbor Light attended?" -c ./docs
ragx ask "..." -c ./docs --variant ragx-rerank
ragx search "Ashford University founder" -c ./docs     # retrieval only, no LLM
ragx variants                                          # list every variant`,
  },
  Server: {
    lang: "bash" as const,
    title: "OpenAI-compatible server",
    code: `pip install "pyragx[server]"
ragx serve ./docs --variant ragx-lexical --port 8000

curl localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Who founded Ashford University?"}]}'`,
  },
};

function Quickstart() {
  const [tab, setTab] = useState<keyof typeof SNIPPETS>("Python");
  const s = SNIPPETS[tab];
  return (
    <section id="quickstart" className="mx-auto max-w-7xl px-6 py-28">
      <div className="grid items-start gap-12 lg:grid-cols-[0.9fr_1.1fr]">
        <Reveal>
          <div className="mb-3 font-mono text-xs font-semibold uppercase tracking-[0.25em] text-accent-sky">Quickstart</div>
          <h2 className="text-3xl font-extrabold tracking-tight text-white md:text-5xl">
            Point it at a folder. <span className="gradient-text">Ask.</span>
          </h2>
          <p className="mt-6 text-lg leading-relaxed text-slate-400">
            ragX ingests text, Markdown, HTML, PDF, Word, CSV and source files into a line-numbered mirror, cuts them into
            passages and answers with <code className="text-accent-sky">path:line</code> citations. Use it from Python, the
            command line, or as an OpenAI-compatible endpoint.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {["OpenAI", "Anthropic", "Google", "Groq", "Together", "Cerebras", "Fireworks", "vLLM", "Hugging Face"].map((p) => (
              <span key={p} className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300">
                {p}
              </span>
            ))}
          </div>
          <a href={DOCS} className="mt-8 inline-flex items-center gap-2 text-sm font-semibold text-accent-sky hover:underline">
            Read the full documentation →
          </a>
        </Reveal>
        <Reveal delay={120}>
          <div className="mb-3 flex gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1">
            {(Object.keys(SNIPPETS) as (keyof typeof SNIPPETS)[]).map((k) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition ${tab === k ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"}`}
              >
                {k}
              </button>
            ))}
          </div>
          <CodeBlock code={s.code} lang={s.lang} title={s.title} />
        </Reveal>
      </div>
    </section>
  );
}

function CTA() {
  return (
    <section className="mx-auto max-w-7xl px-6 pb-28">
      <Reveal>
        <div className="relative overflow-hidden rounded-3xl border border-white/10 p-10 text-center md:p-16">
          <div className="orb left-[-10%] top-[-40%] h-[26rem] w-[26rem] animate-aurora bg-accent-sky/20" />
          <div className="orb right-[-10%] bottom-[-50%] h-[26rem] w-[26rem] animate-aurora bg-accent-violet/20 [animation-delay:-8s]" />
          <div className="relative">
            <h2 className="text-3xl font-black tracking-tight text-white md:text-5xl">
              Evaluate at the scale you <span className="gradient-text gradient-anim">serve</span>.
            </h2>
            <p className="mx-auto mt-5 max-w-2xl text-lg text-slate-400">
              Small-corpus rankings can reverse at scale. ragX is built for the corpus size where finding what the question
              leaves unnamed decides what you can answer.
            </p>
            <div className="mt-9 flex flex-wrap justify-center gap-3">
              <a href={DOCS} className="btn-grad rounded-xl px-6 py-3 font-semibold text-ink-950">Read the docs</a>
              <a href={GH} className="flex items-center gap-2 rounded-xl border border-white/15 px-6 py-3 font-semibold text-slate-200 transition hover:bg-white/5">
                <GitHubIcon /> Star on GitHub
              </a>
            </div>
          </div>
        </div>
      </Reveal>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/5">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-10 text-sm text-slate-500 md:flex-row">
        <div className="flex items-center gap-3">
          <span className="text-lg font-black text-slate-200"><Wordmark /></span>
          <span>Apache 2.0 · © 2026 Gaurav Srivastava</span>
        </div>
        <div className="flex gap-6">
          <a className="hover:text-slate-200" href={DOCS}>Docs</a>
          <a className="hover:text-slate-200" href={GH}>GitHub</a>
          <a className="hover:text-slate-200" href={PYPI}>PyPI</a>
          <span>Paper · coming soon</span>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main>
        <Hero />
        <Stats />
        <Problem />
        <Method />
        <Results />
        <Variants />
        <Quickstart />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}

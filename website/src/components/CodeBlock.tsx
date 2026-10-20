import { useState } from "react";

type Lang = "python" | "bash";

type Spec = { type: string; re: RegExp };

// Sticky-regex token specs, tried in order at each scan position.
const SPECS: Record<Lang, Spec[]> = {
  python: [
    { type: "com", re: /#.*/y },
    { type: "str", re: /[frbFRB]{0,2}"""[\s\S]*?"""|[frbFRB]{0,2}'''[\s\S]*?'''|[frbFRB]{0,2}"(?:[^"\\]|\\.)*"|[frbFRB]{0,2}'(?:[^'\\]|\\.)*'/y },
    { type: "kw", re: /(?:import|from|def|class|return|for|in|if|elif|else|while|with|as|None|True|False|and|or|not|lambda|pass|yield|raise|try|except|finally|async|await|global)\b/y },
    { type: "builtin", re: /(?:print|len|str|int|float|list|dict|set|range|open|enumerate|zip|RagX|Settings)\b/y },
    { type: "num", re: /\d+\.?\d*/y },
    { type: "fn", re: /[A-Za-z_]\w*(?=\s*\()/y },
    { type: "name", re: /[A-Za-z_]\w*/y },
    { type: "punct", re: /[(){}[\].,:=+\-*/%<>!|&]/y },
    { type: "ws", re: /\s+/y },
  ],
  bash: [
    { type: "com", re: /#.*/y },
    { type: "str", re: /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/y },
    { type: "builtin", re: /(?:pip|ragx|cd|export|curl|python|uvicorn|git|npm|node|echo)\b/y },
    { type: "kw", re: /(?:install|ask|ingest|serve|run|build)\b/y },
    { type: "num", re: /\b\d+\b/y },
    { type: "punct", re: /--?[A-Za-z][\w-]*/y },
    { type: "name", re: /[A-Za-z_][\w./-]*/y },
    { type: "ws", re: /\s+/y },
  ],
};

const CLASS: Record<string, string> = {
  com: "tok-com",
  str: "tok-str",
  kw: "tok-kw",
  builtin: "tok-builtin",
  num: "tok-num",
  fn: "tok-fn",
  punct: "tok-punct",
};

function tokenize(code: string, lang: Lang) {
  const specs = SPECS[lang];
  const out: { t: string; type: string }[] = [];
  let i = 0;
  while (i < code.length) {
    let matched = false;
    for (const { type, re } of specs) {
      re.lastIndex = i;
      const m = re.exec(code);
      if (m && m.index === i && m[0].length > 0) {
        out.push({ t: m[0], type });
        i += m[0].length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      out.push({ t: code[i], type: "name" });
      i += 1;
    }
  }
  return out;
}

export default function CodeBlock({
  code,
  lang = "python",
  title,
  lineNumbers = false,
}: {
  code: string;
  lang?: Lang;
  title?: string;
  lineNumbers?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const tokens = tokenize(code, lang);
  const lineCount = code.split("\n").length;
  const copy = async () => {
    let ok = false;
    try {
      // Async Clipboard API — only available in secure contexts (https/localhost).
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code);
        ok = true;
      }
    } catch {
      ok = false;
    }
    if (!ok) {
      // Fallback for http/file:// or older browsers where the Clipboard API is blocked.
      try {
        const ta = document.createElement("textarea");
        ta.value = code;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "0";
        ta.style.left = "0";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch {
        ok = false;
      }
    }
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }
  };
  return (
    <div className="code-surface card glow-border overflow-hidden">
      <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.02] px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
          {title && <span className="ml-2 font-mono text-xs text-slate-500">{title}</span>}
        </div>
        <button
          onClick={copy}
          className="rounded-md px-2 py-1 font-mono text-[11px] text-slate-400 transition hover:bg-white/10 hover:text-white"
        >
          {copied ? "copied ✓" : "copy"}
        </button>
      </div>
      <div className="flex overflow-x-auto">
        {lineNumbers && (
          <pre
            aria-hidden
            className="select-none border-r border-white/5 px-3 py-5 text-right font-mono text-[13px] leading-relaxed text-slate-600"
          >
            {Array.from({ length: lineCount }, (_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </pre>
        )}
        <pre className="flex-1 p-5 font-mono text-[13px] leading-relaxed">
          <code>
            {tokens.map((tok, idx) =>
              CLASS[tok.type] ? (
                <span key={idx} className={CLASS[tok.type]}>
                  {tok.t}
                </span>
              ) : (
                <span key={idx} className="text-slate-200">
                  {tok.t}
                </span>
              ),
            )}
          </code>
        </pre>
      </div>
    </div>
  );
}

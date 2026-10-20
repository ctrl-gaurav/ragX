"""The toolbelt: fast, CPU-only search primitives the agent uses to forage the corpus.

Everything operates on the line-numbered text mirror, so every result carries a stable
``path:line`` the agent (and the grounding gate) can trust. No embeddings, no GPU.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest import Corpus

_WORD = re.compile(r"[A-Za-z0-9]+")
_HAS_RG = shutil.which("rg") is not None
_HAS_UGREP = shutil.which("ugrep") is not None

# --- lexical distillation helpers (no LLM, no vectors) --------------------------------------
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or",
    "is", "was", "were", "are", "be", "been", "did", "does", "do", "who", "what",
    "which", "where", "when", "whose", "whom", "how", "why", "that", "this", "it",
    "its", "as", "from",
}  # fmt: skip
# question relation-words whose evidence sentences use a different surface form — the one
# lexical gap prefix-matching can't bridge
_REL_SYN: dict[str, set[str]] = {
    "spouse": {"married", "marriage", "wife", "husband"},
    "wife": {"married", "spouse"},
    "husband": {"married", "spouse"},
    "death": {"died", "dead"},
    "die": {"death", "dead"},
    "birth": {"born"},
    "born": {"birth"},
    "nationality": {"born", "citizen"},
    "child": {"son", "daughter"},
    "father": {"son", "child"},
    "mother": {"son", "daughter", "child"},
}


def _qtokens(s: str) -> set[str]:
    return {t for t in _WORD.findall(s.lower()) if t not in _STOP}


def _overlap(qt: set[str], st: set[str]) -> int:
    """Morphology-tolerant overlap: exact match, or a shared prefix of >=4 chars
    (die/died, direct/directed/director)."""
    n = 0
    for q in qt:
        if (
            q in st
            or len(q) >= 3
            and any(
                len(s) >= 4
                and (s.startswith(q[:4]) if len(q) >= 4 else s.startswith(q))
                and q[0] == s[0]
                for s in st
            )
        ):
            n += 1
    return n


@dataclass
class Toolbelt:
    corpus: Corpus
    context_lines: int = 1
    max_lines: int = 40
    max_read_chars: int = 4000
    seen_files: set[str] = field(default_factory=set)  # source paths surfaced as evidence
    reads: list[tuple[str, int, int, str]] = field(default_factory=list)  # (src, start, end, body)
    cache: dict[str, str] | None = None  # memoize pure search/rank results across calls
    chain_notes: list[str] = field(
        default_factory=list
    )  # resolved sub-question chain (route="deep")
    hop_evidence: dict[str, str] = field(
        default_factory=dict
    )  # route="deep": file -> excerpt distilled against the sub-question that surfaced it

    def __post_init__(self) -> None:
        self._src2mirror = {f.source_path: f.mirror_path for f in self.corpus.files}
        self._mirror2src = {f.mirror_path: f.source_path for f in self.corpus.files}
        # lazily-built semantic index (only if semantic_search is ever called)
        self._sem_model = None
        self._sem_rows: list[tuple[str, int, str]] = []
        self._sem_mat = None

    # ----------------------------------------------------------------- search
    def _cached(self, key: str, fn) -> str:
        if self.cache is None:
            return fn()
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def search(self, query: str, regex: bool = False, glob: str | None = None) -> str:
        return self._cached(
            f"search\x00{regex}\x00{glob}\x00{query}", lambda: self._search(query, regex, glob)
        )

    def _search(self, query: str, regex: bool = False, glob: str | None = None) -> str:
        if not _HAS_RG:
            return self._py_search(query)
        terms = [query] if regex else (shlex.split(query) or [query])
        cmd = ["rg", "-i", "-n", f"-C{self.context_lines}", "--no-heading"]
        if not regex:
            cmd.append("-F")
        if glob:
            cmd += ["-g", glob]
        for t in terms[:8]:
            cmd += ["-e", t]
        cmd.append(str(self.corpus.mirror_dir))
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
        except Exception as e:  # noqa: BLE001
            return f"(search error: {e})"
        return self._format_hits(out)

    def fuzzy_search(self, query: str, max_edits: int = 2) -> str:
        return self._cached(
            f"fuzzy\x00{max_edits}\x00{query}", lambda: self._fuzzy(query, max_edits)
        )

    def _fuzzy(self, query: str, max_edits: int = 2) -> str:
        if not _HAS_UGREP:
            return self._search(query)  # graceful fallback to exact search
        cmd = [
            "ugrep",
            "-i",
            "-n",
            f"-Z{max_edits}",
            f"-C{self.context_lines}",
            "--no-heading",
            "-e",
            query,
            str(self.corpus.mirror_dir),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
        except Exception as e:  # noqa: BLE001
            return f"(fuzzy error: {e})"
        return self._format_hits(out)

    # ----------------------------------------------------------------- rank (df-BM25)
    def _rank_scored(self, terms: str) -> list[tuple[float, object]]:
        """Index-free BM25 over the mirror document frequencies. No vectors."""
        import math

        q = [t.lower() for t in _WORD.findall(terms)]
        n = max(1, self.corpus.n_docs)
        avgdl = max(1.0, sum(f.n_lines for f in self.corpus.files) / n)
        k1, b = 1.5, 0.75
        scored = []
        for f in self.corpus.files:
            mfile = self.corpus.mirror_dir / f.mirror_path
            try:
                text = mfile.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            toks = _WORD.findall(text)
            if not toks:
                continue
            tf: dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            dl = len(toks)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                dfreq = self.corpus.df.get(term, 1)
                idf = math.log(1 + (n - dfreq + 0.5) / (dfreq + 0.5))
                freq = tf[term]
                score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def rank(self, terms: str, top: int = 8) -> str:
        return self._cached(f"rank\x00{top}\x00{terms}", lambda: self._rank_fmt(terms, top))

    def _rank_fmt(self, terms: str, top: int) -> str:
        scored = self._rank_scored(terms)
        if not scored:
            return "(no documents matched)"
        lines = ["Top documents by relevance (df-BM25, no vectors):"]
        for sc, f in scored[:top]:
            lines.append(f"  {sc:5.2f}  {f.source_path}  — {f.title}")  # type: ignore[attr-defined]
        return "\n".join(lines)

    def rank_files(self, terms: str, top: int = 4) -> list[str]:
        """Top source paths by df-BM25 — used to pre-seed/route the agent. No vectors."""
        return [f.source_path for _, f in self._rank_scored(terms)[:top]]  # type: ignore[attr-defined]

    # ----------------------------------------------------------------- semantic (opt-in)
    def semantic_search(self, query: str, top: int = 8) -> str:
        """Optional embedding-based fallback (``pyragx[semantic]``).

        Built lazily over the mirror on first use; returns citable ``path:line`` hits. This is
        a fallback for when lexical search misses — the core method is still embedding-free.
        """
        if not self._build_semantic_index():
            return (
                "(semantic_search unavailable — install the extra: pip install 'pyragx[semantic]'. "
                "Use SEARCH / RANK / FUZZY instead.)"
            )
        try:
            import numpy as np

            qv = self._sem_model.encode([query], normalize_embeddings=True)[0]
            sims = self._sem_mat @ qv
            order = np.argsort(-sims)[:top]
        except Exception as e:  # noqa: BLE001
            return f"(semantic error: {e})"
        lines = ["Top semantic matches (embedding fallback):"]
        for i in order:
            src, lineno, text = self._sem_rows[int(i)]
            self.seen_files.add(src)
            lines.append(f"  {sims[int(i)]:.2f}  {src}:{lineno}: {text[:160]}")
        return "\n".join(lines)

    def _build_semantic_index(self) -> bool:
        if self._sem_mat is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return False
        rows: list[tuple[str, int, str]] = []
        for f in self.corpus.files:
            mfile = self.corpus.mirror_dir / f.mirror_path
            try:
                text = mfile.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, ln in enumerate(text.splitlines(), 1):
                s = ln.strip()
                if len(s) >= 8:  # skip trivial lines
                    rows.append((f.source_path, i, s))
        if not rows:
            return False
        import os as _os

        model_name = _os.environ.get("RAGX_SEMANTIC_MODEL", "all-MiniLM-L6-v2")
        self._sem_model = SentenceTransformer(model_name, device="cpu")
        self._sem_mat = self._sem_model.encode(
            [r[2] for r in rows], normalize_embeddings=True, show_progress_bar=False
        )
        self._sem_rows = rows
        return True

    # ----------------------------------------------------------------- navigation
    def list_files(self, glob: str | None = None) -> str:
        files = self.corpus.files
        if glob:
            from fnmatch import fnmatch

            files = [f for f in files if fnmatch(f.source_path, glob)]
        if not files:
            return "(no files)"
        lines = [f"{len(files)} file(s):"]
        for f in files[:200]:
            lines.append(f"  {f.source_path}  ({f.n_lines} lines)  — {f.title}")
        return "\n".join(lines)

    def outline(self) -> str:
        hs = self._read_headings()
        if not hs:
            return "(no headings detected; use list_files + search)"
        lines = ["Corpus outline:"]
        for h in hs[:120]:
            lines.append("  " + "  " * (h["level"] - 1) + f"{h['path']}:{h['line']}  {h['text']}")
        return "\n".join(lines)

    def toc(self, path: str) -> str:
        src = self._resolve_src(path)
        hs = [h for h in self._read_headings() if h["path"] == src]
        if not hs:
            return f"(no headings in {path})"
        return "\n".join(f"  {h['line']:>5}: " + "  " * (h["level"] - 1) + h["text"] for h in hs)

    # ----------------------------------------------------------------- read
    def read(self, path: str, start: int = 1, end: int | None = None) -> str:
        src = self._resolve_src(path)
        mirror_rel = self._src2mirror.get(src)
        if not mirror_rel:
            return f"(no such file '{path}'. Use list_files or search to find paths.)"
        mfile = self.corpus.mirror_dir / mirror_rel
        try:
            all_lines = mfile.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return f"(read error: {e})"
        start = max(1, start)
        end = len(all_lines) if end is None else min(end, len(all_lines))
        chunk = all_lines[start - 1 : end]
        body = "\n".join(f"{start + i:>5}: {ln}" for i, ln in enumerate(chunk))
        body = body[: self.max_read_chars]
        self.seen_files.add(src)
        self.reads.append((src, start, end, "\n".join(chunk)[: self.max_read_chars]))
        return f"--- {src} (lines {start}-{end}) ---\n{body}"

    def surfaced_evidence(
        self,
        extra_cap: int = 8,
        head_chars: int = 1200,
        rank_query: str | None = None,
        distill_query: str | None = None,
        priority: dict[str, str] | None = None,
    ) -> list[tuple[str, int, int, str]]:
        """The read spans, plus the head of up to ``extra_cap`` surfaced-but-unread files — so the
        synthesizer composes over what the agent *found*, not only what it explicitly READ. On
        multi-hop QA the agent often surfaces the right documents (high support-recall) but
        under-reads them; routing the surfaced set to the answer step recovers answer-F1 (+16 on
        HotpotQA, +7.5 on MuSiQue). The ``extra_cap`` matters: adding *all* surfaced docs dilutes
        cleanly-read comparison questions (2WikiMultiHop), so the extras are capped (cap=8 is
        parity-or-better on answer-F1 on both HotpotQA and 2WikiMultiHop in our sweep).

        When ``rank_query`` is given (normally the question), the capped extras are the
        **most df-BM25-relevant** surfaced files rather than an arbitrary set order — measured
        +1.1 to +1.9 answer-F1 over first-come extras, and it makes the selection deterministic.

        ``priority`` (route="deep") maps file → pre-distilled excerpt for the docs that resolved
        each hop; they enter the bundle FIRST. Ranking extras against the question would drop
        exactly these files — a later-hop doc shares no vocabulary with the question by
        construction (that is why the deep route had to resolve the hop to find it).
        """
        bundle = list(self.reads)
        read_srcs = {r[0] for r in self.reads}
        added = 0
        for src, body in (priority or {}).items():
            if src in read_srcs or added >= extra_cap:
                continue
            bundle.append((src, 1, 40, body))
            read_srcs.add(src)  # do not re-add below
            added += 1
        candidates: list[str] = sorted(self.seen_files)
        if rank_query and candidates:
            score = {
                f.source_path: sc  # type: ignore[attr-defined]
                for sc, f in self._rank_scored(rank_query)
            }
            candidates.sort(key=lambda src: -score.get(src, 0.0))
        for src in candidates:
            if src in read_srcs or added >= extra_cap:
                continue
            mirror_rel = self._src2mirror.get(src)
            if not mirror_rel:
                continue
            try:
                text = (self.corpus.mirror_dir / mirror_rel).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            n_lines = text.count("\n") + 1
            body = self.distill(distill_query, text) if distill_query else text[:head_chars]
            bundle.append((src, 1, min(n_lines, 40), body))
            added += 1
        return bundle

    def peek(self, path: str, line: int, window: int = 5) -> str:
        return self.read(path, line - window, line + window)

    # ------------------------------------------------------- distillation (no LLM, no vectors)
    def file_text(self, path: str) -> str:
        """The full mirror text of a file (no read-tracking side effects), '' if missing."""
        mirror_rel = self._src2mirror.get(self._resolve_src(path))
        if not mirror_rel:
            return ""
        try:
            return (self.corpus.mirror_dir / mirror_rel).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return ""

    def distill(self, question: str, text: str, keep: int = 3, budget: int = 420) -> str:
        """Sentence-level lexical distillation: the first sentence (title/definition) plus the
        sentences with the highest content-word overlap with the question, in original order.
        Morphology-tolerant (shared >=4-char prefixes) with a tiny relation lexicon
        (spouse<->married, death<->died, ...) for the QA relation words whose evidence uses a
        different surface form. Lets a composer see many documents in a small token budget."""
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            return text[:budget]
        qt = _qtokens(question)
        for t in list(qt):
            qt |= _REL_SYN.get(t, set())
        order = sorted(range(1, len(sents)), key=lambda i: -_overlap(qt, _qtokens(sents[i])))
        pick = {0} | set(order[: max(0, keep - 1)])
        return " ".join(sents[i] for i in sorted(pick))[:budget]

    # ----------------------------------------------------------------- helpers
    def _format_hits(self, out: str) -> str:
        if not out.strip():
            return "(no matches — try synonyms, broader terms, fuzzy_search, or list_files)"
        shown: list[str] = []
        for ln in out.splitlines()[: self.max_lines]:
            m = re.match(r"^(.*?\.txt)[:-](\d+)[:-]?(.*)$", ln)
            if m:
                mirror_rel = (
                    Path(m.group(1)).relative_to(self.corpus.mirror_dir).as_posix()
                    if str(self.corpus.mirror_dir) in m.group(1)
                    else os.path.basename(m.group(1))
                )
                src = self._mirror2src.get(mirror_rel) or self._mirror_basename_to_src(m.group(1))
                if src:
                    self.seen_files.add(src)
                shown.append(f"{src or m.group(1)}:{m.group(2)}: {m.group(3).strip()[:160]}")
            else:
                shown.append(ln[:160])
        more = (
            ""
            if len(out.splitlines()) <= self.max_lines
            else f"\n… ({len(out.splitlines())} total hits)"
        )
        return "\n".join(shown) + more

    def _mirror_basename_to_src(self, full: str) -> str | None:
        try:
            rel = Path(full).resolve().relative_to(self.corpus.mirror_dir.resolve()).as_posix()
        except Exception:  # noqa: BLE001
            return None
        return self._mirror2src.get(rel)

    def _resolve_src(self, path: str) -> str:
        if path in self._src2mirror:
            return path
        # tolerate mirror paths or basenames
        if path in self._mirror2src:
            return self._mirror2src[path]
        base = os.path.basename(path)
        for src in self._src2mirror:
            if os.path.basename(src) == base or src.endswith(path):
                return src
        return path

    def _read_headings(self) -> list[dict]:
        import json

        f = self.corpus.cache_dir / "index" / "headings.jsonl"
        if not f.exists():
            return []
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

    def _py_search(self, query: str) -> str:
        """Pure-python fallback when ripgrep is unavailable."""
        terms = [t.lower() for t in (shlex.split(query) or [query])]
        shown: list[str] = []
        for f in self.corpus.files:
            mfile = self.corpus.mirror_dir / f.mirror_path
            try:
                lines = mfile.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, ln in enumerate(lines, 1):
                low = ln.lower()
                if any(t in low for t in terms):
                    self.seen_files.add(f.source_path)
                    shown.append(f"{f.source_path}:{i}: {ln.strip()[:160]}")
                    if len(shown) >= self.max_lines:
                        return "\n".join(shown)
        return "\n".join(shown) if shown else "(no matches)"


__all__ = ["Toolbelt"]

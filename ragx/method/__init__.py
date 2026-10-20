"""The ragX method: a forage loop and a slot chain feeding one compose call.

1. **Forage loop.** An agent issues up to ``steps`` actions: ``RANK`` (the ranker's top
   passages), ``READ`` (a passage in full) and ``ANSWER``. Queries written after reading name
   evidence the question itself does not name.
2. **Slot chain.** The question is decomposed into at most ``max_hops`` sub-questions with
   ``#N`` slots; each hop retrieves ``hop_k`` passages for its sub-question with the earlier
   answers filled in, and answers it in one short call.
3. **Compose.** One call answers over the read passages, the passages that resolved each hop,
   the other surfaced passages ranked against the question and, in ``ragx`` and ``ragx-think``,
   the question's own top ``backfill`` passages.

The defaults are the frozen settings of the paper (ten turns, eight titles per RANK, four
passages per hop, eight reads and twelve extras composed, whole passages, the fence on).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..passages import Passage, PassageStore
from ..synthesize import answer_type_hint, hint_kind
from ..types import Event
from . import prompts as P

_ACTION = re.compile(r"(RANK|SEARCH|READ|ANSWER)\s*:\s*(.*)", re.S | re.I)
_SPLIT_SENT = re.compile(r"(?<=[.!?])\s+")
_TOKRE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "did",
        "does",
        "do",
        "who",
        "what",
        "which",
        "where",
        "when",
        "whose",
        "whom",
        "how",
        "why",
        "that",
        "this",
        "it",
        "its",
        "as",
        "from",
    ]
)
_REL_SYN = {
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

CHARS_PER_TOK = 3.5
ANSWER_RESERVE = 512
PROMPT_RESERVE = 768
_MIN_KEEP_TOK = 64


@dataclass
class MethodConfig:
    """Settings of one run of the method (see :mod:`ragx.variants` for the named presets)."""

    route: str = "deep"  # "deep" (forage loop + slot chain) or "fast" (forage loop only)
    steps: int = 10  # forager turns (T)
    rank_k: int = 8  # passages listed per RANK (k_R)
    hop_k: int = 4  # passages retrieved per hop (k_H)
    max_hops: int = 4  # sub-questions at most
    read_cap: int = 8  # read passages composed (c_R)
    extra_cap: int = 12  # extra passages composed, hop passages first (c_E)
    fast_cap: int = 16  # extras composed by the fast route
    backfill: int = 0  # the question's own top passages appended (K_b)
    tools: tuple[str, ...] = ("RANK", "READ")
    fence: bool = True  # tool results and evidence fenced as untrusted data
    typed: bool = True  # answer-type hint in the compose prompt
    reasoning: bool | str = False  # a reasoning pass before the compose (True / "auto")
    abstain: bool = False  # composer may reply "I don't know"
    read_chars: int = 1500  # characters of a READ shown to the forager
    rank_chars: int = 0  # characters of each RANK result shown inline (0 = titles only)


@dataclass
class RunResult:
    answer: str
    composed: list[int] = field(default_factory=list)  # passage ids the composer saw, in order
    reads: list[int] = field(default_factory=list)
    surfaced: list[int] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    n_calls: int = 0
    latency_s: float = 0.0


# --------------------------------------------------------------------------------- helpers
def _qtokens(s: str) -> set[str]:
    return {t for t in _TOKRE.findall(s.lower()) if t not in _STOP}


def _overlap(qt: set[str], st: set[str]) -> int:
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


def distill(question: str, text: str, keep: int = 3, budget: int = 420) -> str:
    """First sentence plus the sentences with the most question overlap, in original order."""
    sents = [s.strip() for s in _SPLIT_SENT.split(text) if s.strip()]
    if not sents:
        return text[:budget]
    qt = _qtokens(question)
    for t in list(qt):
        qt |= _REL_SYN.get(t, set())
    order = sorted(range(1, len(sents)), key=lambda i: -_overlap(qt, _qtokens(sents[i])))
    pick = {0} | set(order[: max(0, keep - 1)])
    return " ".join(sents[i] for i in sorted(pick))[:budget]


def _parse_lines(text: str, max_n: int = 4, require_q: bool = False) -> list[str]:
    out = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln).strip() for ln in (text or "").splitlines()]
    return [s for s in out if s and (("?" in s) if require_q else True)][:max_n]


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOK) + 1


def fit_parts(
    parts: list[str], budget: int | None, first_to_go=()
) -> tuple[list[str], list[int] | None]:
    """Fit evidence parts into ``budget`` tokens: drop or cut from the tail, the indices in
    ``first_to_go`` (the backfill) first. Returns the kept parts and their indices (``None``
    when nothing changed)."""
    parts = list(parts)
    if budget is None or not parts:
        return parts, None
    sizes = [est_tokens(p) + 1 for p in parts]
    total = sum(sizes)
    if total <= budget:
        return parts, None
    over = total - budget
    alive = [True] * len(parts)
    go = [i for i in sorted(set(first_to_go), reverse=True) if 0 <= i < len(parts)]
    go += [i for i in reversed(range(len(parts))) if i not in set(go)]
    for i in go:
        if over <= 0:
            break
        if sizes[i] - over >= _MIN_KEEP_TOK:
            keep = sizes[i] - over
            parts[i] = parts[i][: max(1, int(len(parts[i]) * keep / sizes[i]) - 8)]
            over = 0
        else:
            alive[i] = False
            over -= sizes[i]
    kept = [i for i in range(len(parts)) if alive[i]]
    return [parts[i] for i in kept], kept


def _part(p: Passage, text: str | None = None) -> str:
    return f"[{p.name} — {p.title}]\n{p.text if text is None else text}"


def _ask(llm, prompt: str, sys: str, max_tokens: int) -> str:
    return (
        llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        or ""
    )


# ---------------------------------------------------------------------------------- engine
class Method:
    """One configured ragX pipeline over a passage store and a ranker."""

    def __init__(
        self,
        store: PassageStore,
        ranker,
        cfg: MethodConfig,
        budget: Callable[[str], int | None] | None = None,
    ):
        self.store, self.ranker, self.cfg = store, ranker, cfg
        self.budget = budget or (lambda used: None)

    # ------------------------------------------------------------------ public
    def run(self, question: str, llm, answer_llm=None) -> Iterator[Event]:
        """Stream the run's events; the generator's return value is a :class:`RunResult`."""
        t0 = time.time()
        cfg = self.cfg
        cllm = answer_llm or llm
        yield Event("plan", f"Foraging for: {question}")
        st = yield from self._forage(question, llm)
        ncalls = st["ncalls"]
        reads = list(dict.fromkeys(st["reads"]))
        seen = list(st["seen"])
        chain: list[str] = []
        hop_ev: dict[int, str] = {}
        if cfg.route == "deep":
            hops = yield from self._chain(question, llm, seen)
            ncalls += hops["ncalls"]
            chain, hop_ev = hops["notes"], hops["hop_ev"]
        parts, composed, backfill_ix = self._evidence(question, reads, seen, st["shown"], hop_ev)
        if cfg.backfill:
            yield Event("backfill", f"{len(backfill_ix)} passages from the question's own ranking")
        used = question
        parts, kept = fit_parts(parts, self.budget(used), first_to_go=backfill_ix)
        if kept is not None:
            composed = [composed[i] for i in kept]
        answer = st["draft"]
        if parts:
            yield Event("verify", f"Composing over {len(parts)} passages")
            answer, n = self._compose(cllm, question, "\n\n".join(parts))
            ncalls += n
        if not answer:
            answer = _ask(
                llm,
                f"Question: {question}\n(Answer from what you found.)\nAnswer:",
                P.ANSWER_SYS,
                256,
            ).strip()
            ncalls += 1
        yield Event("answer", answer)
        return RunResult(
            answer=answer,
            composed=composed,
            reads=reads,
            surfaced=seen,
            chain=chain,
            n_calls=ncalls,
            latency_s=round(time.time() - t0, 3),
        )

    # ------------------------------------------------------------------ forage loop
    def _forage(self, question: str, llm):
        cfg, store = self.cfg, self.store
        allowed = {t.upper() for t in cfg.tools}
        sys = P.agent_prompt(getattr(self.ranker, "name", "bm25"), allowed)
        if cfg.fence:
            sys = sys + "\n" + P.UNTRUSTED_NOTE
        msgs = [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": f"Question: {question}\nThe corpus has {len(store)} text files. Begin.",
            },
        ]
        seen: list[int] = []
        reads: list[int] = []
        shown: list[int] = []
        ncalls = 0
        draft = ""

        def see(pid: int) -> None:
            if pid not in seen:
                seen.append(pid)

        for turn in range(1, cfg.steps + 1):
            reply = llm.chat(msgs, max_tokens=400) or ""
            ncalls += 1
            msgs.append({"role": "assistant", "content": reply})
            m = _ACTION.search(reply)
            if not m:
                obs = "Invalid action. Use RANK:/SEARCH:/READ:/ANSWER:."
                yield Event("thought", "invalid action", step=turn)
            else:
                act = m.group(1).upper()
                arg = m.group(2).strip().splitlines()[0].strip() if m.group(2).strip() else ""
                if act == "ANSWER":
                    draft = arg
                    yield Event("thought", f"forager answered: {arg}", step=turn)
                    break
                if act not in allowed:
                    obs = "Tool not available."
                elif act == "RANK":
                    yield Event("rank", arg, step=turn)
                    lines = ["Top files by relevance:"]
                    for pos, pid in enumerate(self.ranker.search(arg, cfg.rank_k)):
                        p = store[pid]
                        see(pid)
                        lines.append(f"  {p.name}  — {p.title}")
                        if pos < cfg.rank_chars:
                            lines.append(f"      {p.text[:300]}")
                            if pid not in reads and pid not in shown:
                                shown.append(pid)
                    obs = "\n".join(lines) if len(lines) > 1 else "(no files matched)"
                elif act == "SEARCH":
                    yield Event("search", arg, step=turn)
                    obs, hits = self._search(arg)
                    for pid in hits:
                        see(pid)
                else:  # READ
                    p = store.resolve(arg)
                    if p is None:
                        obs = f"No such file '{arg}'. RANK first to find filenames."
                    else:
                        yield Event("read", p.name, step=turn)
                        obs = f"--- {p.name} ---\n{p.numbered()[: cfg.read_chars]}"
                        see(p.pid)
                        reads.append(p.pid)
            if cfg.fence:
                obs = P.quarantine(obs)
            msgs.append({"role": "user", "content": f"Observation:\n{obs}"})
        return {"draft": draft, "seen": seen, "reads": reads, "shown": shown, "ncalls": ncalls}

    def _search(self, query: str, max_hits: int = 40) -> tuple[str, list[int]]:
        """Case-insensitive OR search over passage lines (``SEARCH``, off in the frozen method)."""
        terms = [t.lower() for t in query.split() if t.strip()]
        out, hits = [], []
        for p in self.store.passages:
            for i, ln in enumerate(p.text.splitlines()):
                low = ln.lower()
                if any(t in low for t in terms):
                    out.append(f"{p.path}:{p.start + i}: {ln.strip()[:160]}")
                    if p.pid not in hits:
                        hits.append(p.pid)
                    if len(out) >= max_hits:
                        return "\n".join(out), hits
        return ("\n".join(out) if out else "(no matches)"), hits

    # ------------------------------------------------------------------ slot chain
    def _chain(self, question: str, llm, seen: list[int]):
        cfg, store = self.cfg, self.store
        subqs = _parse_lines(
            _ask(llm, f"Question: {question}\n\nSub-questions:", P.ITER_DECOMP_SYS, 160),
            max_n=cfg.max_hops,
            require_q=True,
        )
        ncalls = 1
        answers: list[str] = []
        notes: list[str] = []
        hop_ev: dict[int, str] = {}
        for tmpl in subqs:
            sq = tmpl
            for j, a in enumerate(answers):
                sq = sq.replace(f"#{j + 1}", a)
            pids = self.ranker.search(sq, cfg.hop_k)
            ev = "\n\n".join(f"[{store[i].title}]\n{distill(sq, store[i].text)}" for i in pids)
            lines = _ask(llm, f"Question: {sq}\n\nExcerpts:\n{ev}\n\nAnswer:", P.SUBQ_SYS, 48)
            ncalls += 1
            first = lines.strip().splitlines()
            a = first[0][:80] if first else ""
            answers.append(a)
            notes.append(f"- {sq} -> {a}")
            for i in pids[:2]:
                hop_ev.setdefault(i, distill(sq, store[i].text))
            for i in pids:
                if i not in seen:
                    seen.append(i)
            yield Event("hop", f"{sq} → {a}")
        return {"notes": notes, "hop_ev": hop_ev, "ncalls": ncalls}

    # ------------------------------------------------------------------ evidence
    def _evidence(self, question, reads, seen, shown, hop_ev):
        """Compose order: reads, then hop and shown passages (priority), then the other
        surfaced passages ranked against the question, then the backfill."""
        cfg, store = self.cfg, self.store
        read_set = set(reads)
        reads = reads[: cfg.read_cap]
        parts = [_part(store[i]) for i in reads]
        composed = list(reads)
        extras: list[int] = []
        if cfg.route == "fast":
            order = {pid: r for r, pid in enumerate(self.ranker.search(question, 64))}
            rest = [i for i in shown if i not in read_set]
            rest += sorted(
                (i for i in seen if i not in read_set and i not in rest),
                key=lambda i: order.get(i, 10**6),
            )
            extras = rest[: cfg.fast_cap]
        else:
            for i in list(hop_ev) + shown:
                if i not in read_set and i not in extras and len(extras) < cfg.extra_cap:
                    extras.append(i)
            order = {pid: r for r, pid in enumerate(self.ranker.search(question, 64))}
            rest = sorted(
                (i for i in seen if i not in read_set and i not in extras),
                key=lambda i: order.get(i, 10**6),
            )
            extras += rest[: max(0, cfg.extra_cap - len(extras))]
        parts += [_part(store[i]) for i in extras]
        composed += extras
        backfill_ix: list[int] = []
        if cfg.backfill:
            have = set(composed)
            for i in self.ranker.search(question, cfg.backfill):
                if i not in have:
                    backfill_ix.append(len(parts))
                    parts.append(_part(store[i]))
                    composed.append(i)
                    have.add(i)
        return parts, composed, backfill_ix

    # ------------------------------------------------------------------ compose
    def _compose(self, llm, question: str, evidence: str) -> tuple[str, int]:
        cfg = self.cfg
        if cfg.fence:
            evidence = P.quarantine(evidence)
            question = f"{question}\n({P.UNTRUSTED_NOTE})"
        sys = P.SYNTH_SYS_ABSTAIN if cfg.abstain else P.SYNTH_SYS
        hint = answer_type_hint(question) if cfg.typed else None
        hint_line = f"\n({hint})" if hint else ""
        body = f"Question: {question}\n\nExcerpts:\n{evidence}{hint_line}"
        use_cot = cfg.reasoning is True or (
            cfg.reasoning == "auto" and hint_kind(question) in ("number", "comparison", "date")
        )
        if use_cot:
            reasoning = _ask(llm, body + "\n\nReasoning:", P.COT_SYS, 400).strip()
            ans = _ask(
                llm,
                f"Question: {question}\n\nAnalysis of the source documents:\n{reasoning}{hint_line}\n"
                "(Even if the analysis is inconclusive, commit to the single most likely "
                "specific answer — never output 'cannot be determined' or 'not provided'.)"
                "\n\nAnswer:",
                sys,
                128,
            )
            return ans.strip(), 2
        return _ask(llm, body + "\n\nAnswer:", sys, 128).strip(), 1


__all__ = ["Method", "MethodConfig", "RunResult", "distill", "fit_parts"]

"""The public entry point: :class:`RagX`."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import Settings
from .ingest import Corpus, ingest
from .llm import LLM
from .method import Method, MethodConfig, RunResult, est_tokens
from .passages import Passage, PassageStore, build_passages
from .pricing import estimate_cost
from .types import Answer, Citation, Event
from .variants import Variant, get_variant

_WORD = re.compile(r"[A-Za-z0-9]+")
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
        "into",
        "over",
        "under",
        "than",
        "then",
    ]
)
_LOCAL_CONTEXT = 32768  # context assumed for vllm:/hf: readers unless max_context_tokens is set


def _usage_with_cost(llm: LLM) -> dict[str, Any]:
    """LLM usage dict enriched with an estimated USD cost (``None`` if model is unpriced/local)."""
    d = llm.usage.as_dict()
    d["cost_usd"] = estimate_cost(llm.spec, d["prompt_tokens"], d["completion_tokens"])
    return d


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s)}


def _cite(p: Passage) -> Citation:
    quote = next((ln.strip() for ln in p.text.splitlines() if ln.strip()), "")[:160]
    return Citation(p.path, p.start, p.end, quote)


def ground(text: str, passages: list[Passage], verify: bool = True) -> tuple[list[Citation], bool]:
    """Citations for an answer: the composed passages that contain it (verbatim, or at least 80%
    of its content words). ``grounded`` is True only when one passage does."""
    ans_tok = _tokens(text) - _STOP
    ans_norm = " ".join(_WORD.findall(text.lower()))
    cites: list[Citation] = []
    for p in passages:
        body_norm = " ".join(_WORD.findall(p.text.lower()))
        hit = bool(ans_norm) and ans_norm in body_norm
        covered = bool(ans_tok) and len(ans_tok & _tokens(p.text)) >= 0.8 * len(ans_tok)
        if hit or covered:
            cites.append(_cite(p))
    if cites:
        return cites, True
    if not verify:  # no check: cite what the composer saw
        return [_cite(p) for p in passages[:8]], bool(passages)
    return [_cite(p) for p in passages[:4]], False


class RagX:
    """Agentic retrieval over a folder of documents.

    Parameters
    ----------
    corpus:
        Path to a file or directory. Ingested incrementally into a line-numbered text mirror
        (cached in ``<corpus>/.ragx``).
    model:
        ``provider:model`` string. Defaults to whatever API key is present, else a local model.
    variant:
        One of ``ragx``, ``ragx-core``, ``ragx-lexical`` (default), ``ragx-fused``,
        ``ragx-rerank``, ``ragx-fast``, ``ragx-think`` or ``classic``; see :mod:`ragx.variants`.
    **kwargs:
        Any field of :class:`ragx.config.Settings` (``steps``, ``backfill``, ``embed_model``,
        ``answer_model``, ``think``, ...).
    """

    def __init__(
        self,
        corpus: str | Path,
        model: str | None = None,
        variant: str | None = None,
        **kwargs: Any,
    ):
        if model is not None:
            kwargs["model"] = model
        if variant is not None:
            kwargs["variant"] = variant
        self.settings = Settings(**kwargs)
        self.variant: Variant = get_variant(self.settings.variant)
        self.corpus: Corpus = ingest(corpus, self.settings)
        self._llms: dict[str, LLM] = {}
        self._store: PassageStore | None = None
        self._ranker = None
        self._response_cache: dict[str, str] | None = {} if self.settings.cache else None
        self._tool_cache: dict[str, str] | None = {} if self.settings.cache else None

    # ------------------------------------------------------------------ components
    @property
    def think(self) -> bool:
        return self.variant.think if self.settings.think is None else self.settings.think

    @property
    def retriever(self) -> str:
        return self.settings.retriever or self.variant.retriever

    @property
    def store(self) -> PassageStore:
        """The corpus cut into passages (built on first use)."""
        if self._store is None:
            self._store = build_passages(self.corpus, self.settings.passage_words)
        return self._store

    @property
    def ranker(self):
        """The variant's ranker over :attr:`store` (built on first use; dense embeddings are
        cached under ``.ragx/embeddings``)."""
        if self._ranker is None:
            from .rankers import make_ranker

            self._ranker = make_ranker(
                self.retriever,
                self.store,
                self.settings,
                cache_dir=self.corpus.cache_dir / "embeddings",
            )
        return self._ranker

    def _llm(self, spec: str) -> LLM:
        if spec not in self._llms:
            self._llms[spec] = LLM(
                spec,
                temperature=self.settings.temperature,
                response_cache=self._response_cache,
                think=self.think,
                think_budget=self.settings.think_budget,
            )
        return self._llms[spec]

    def _budget(self, llm: LLM):
        ctx = self.settings.max_context_tokens
        if ctx is None and llm.provider in ("vllm", "hf"):
            ctx = _LOCAL_CONTEXT
        if ctx is None:
            return lambda used: None
        think = self.settings.think_budget if self.think else 0

        def budget(used: str) -> int:
            return max(256, ctx - 512 - 768 - think - est_tokens(used))

        return budget

    def method_config(self) -> MethodConfig:
        s, v = self.settings, self.variant
        return MethodConfig(
            route=v.route,
            steps=s.steps,
            rank_k=s.rank_k,
            hop_k=s.hop_k,
            max_hops=s.max_hops,
            read_cap=s.read_cap,
            extra_cap=s.extra_cap,
            backfill=v.backfill if s.backfill is None else s.backfill,
            tools=tuple(t.upper() for t in s.tools),
            fence=s.fence,
            typed=s.typed_hints,
            reasoning=s.compose_reasoning,
            abstain=s.abstain_on_weak_evidence,
        )

    # ------------------------------------------------------------------ public api
    def reingest(self) -> Corpus:
        """Re-scan the corpus (incremental); call after files change."""
        self.corpus = ingest(self.corpus.root, self.settings)
        self._store = None
        self._ranker = None
        return self.corpus

    def ask(self, question: str) -> Answer:
        """Answer a question with a grounded, cited response."""
        gen = self.ask_stream(question)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]

    def search(self, query: str, top: int = 8) -> list[str]:
        """Retrieval only, no LLM call: the top passages (``path:start-end``) for a query."""
        if self.variant.route == "classic":
            from .tools import Toolbelt

            return Toolbelt(corpus=self.corpus, cache=self._tool_cache).rank_files(query, top=top)
        return [self.store[i].name for i in self.ranker.search(query, top)]

    def ask_stream(self, question: str) -> Iterator[Event]:
        """Stream the agent's steps; the generator's return value is the final :class:`Answer`."""
        if not question or not question.strip():
            return Answer(text="", grounded=False, confidence=0.0)
        if self.variant.route == "classic":
            return (yield from self._classic_stream(question))
        scout = self._llm(self.settings.scout())
        answerer = self._llm(self.settings.answerer())
        method = Method(self.store, self.ranker, self.method_config(), self._budget(answerer))
        events: list[Event] = []
        gen = method.run(question, scout, answerer)
        while True:
            try:
                ev = next(gen)
            except StopIteration as stop:
                result: RunResult = stop.value
                break
            events.append(ev)
            yield ev
        passages = [self.store[i] for i in result.composed]
        cites, grounded = ground(result.answer, passages, self.settings.verify_citations)
        text = result.answer
        if self.settings.abstain_on_weak_evidence and not grounded:
            text = "I couldn't find enough evidence in the corpus to answer that confidently."
        ans = Answer(text=text, citations=cites[:8], grounded=grounded, trace=events)
        ans.usage = self._usage(scout, answerer)
        ans.usage["llm_calls"] = result.n_calls
        ans.usage["variant"] = self.variant.name
        return ans

    def _usage(self, scout: LLM, answerer: LLM) -> dict[str, Any]:
        usage: dict[str, Any] = {scout.spec: _usage_with_cost(scout)}
        if answerer.spec != scout.spec:
            usage[answerer.spec] = _usage_with_cost(answerer)
        total = sum(
            u["cost_usd"] for u in usage.values() if isinstance(u, dict) and u.get("cost_usd")
        )
        usage["total_cost_usd"] = round(total, 6) or None
        return usage

    # ------------------------------------------------------------------ classic agent
    def _classic_stream(self, question: str) -> Iterator[Event]:
        """ragX's original file-level agent: the grep toolbelt (ripgrep, fuzzy search, outline,
        line-range reads) over the text mirror, with the routes of ``Settings.route``."""
        from .agent import Agent
        from .synthesize import answers_agree, synthesize
        from .tools import _HAS_UGREP, Toolbelt

        tools = Toolbelt(
            corpus=self.corpus,
            context_lines=self.settings.context_lines,
            max_lines=self.settings.max_result_lines,
            max_read_chars=self.settings.max_read_chars,
            cache=self._tool_cache,
        )
        scout = self._llm(self.settings.scout())
        answerer = self._llm(self.settings.answerer())
        events: list[Event] = []
        deep = self.settings.route in ("deep", "both")
        max_steps, fast_cap = self.settings.budget_for()

        def _synth(draft: str) -> Answer:
            ans = synthesize(
                question,
                draft,
                tools,
                answer_llm=answerer,
                verify=self.settings.verify_citations,
                abstain=self.settings.abstain_on_weak_evidence,
                compose_surfaced=self.settings.compose_surfaced,
                surf_cap=self.settings.deep_surf_cap if deep else fast_cap,
                typed_hints=self.settings.typed_hints,
                distill_extras=deep,
                chain_notes=deep and self.settings.deep_chain_notes,
                compose_reasoning=self.settings.compose_reasoning,
            )
            ans.trace = events
            ans.usage = self._usage(scout, answerer)
            ans.usage["variant"] = "classic"
            return ans

        if self.settings.route == "adaptive":
            draft = ""
            for ev in self._single_shot(question, tools, scout):
                if ev.kind == "answer":
                    draft = ev.text
                events.append(ev)
                yield ev
            answer = _synth(draft)
            if answer.grounded and tools.reads:
                return answer
            yield Event("thought", "weak single-shot evidence — escalating to full foraging")

        prerank = tools.rank_files(question, top=6) if self.settings.prerank else []
        agent = Agent(
            scout,
            tools,
            max_steps=max_steps,
            fuzzy_available=_HAS_UGREP and self.settings.fuzzy,
            semantic_available=self.settings.semantic_fallback,
            prerank_files=prerank,
            sandbox=self.settings.sandbox_observations,
        )
        for ev in agent.stream(question):
            events.append(ev)
            yield ev
        if deep:
            from .agent.resolve import resolve_chain

            for ev in resolve_chain(
                question,
                tools,
                scout,
                k=self.settings.single_shot_reads,
                max_hops=self.settings.deep_hops,
            ):
                events.append(ev)
                yield ev
        notes_alt: str | None = None
        if self.settings.route == "both":
            notes_alt = synthesize(
                question,
                agent.answer_draft,
                tools,
                answer_llm=answerer,
                verify=False,
                compose_surfaced=self.settings.compose_surfaced,
                surf_cap=self.settings.deep_surf_cap,
                typed_hints=self.settings.typed_hints,
                distill_extras=True,
                chain_notes=True,
            ).text
        yield Event("verify", "Grounding answer in cited evidence", step=agent.max_steps)
        answer = _synth(agent.answer_draft)
        if notes_alt is not None:
            agree = answers_agree(answer.text, notes_alt)
            answer.confidence = 0.9 if agree else 0.5
            check_ev = Event(
                "verify",
                f"note-robustness cross-check: {'agree' if agree else 'disagree'}",
                detail={"alternative": notes_alt} if not agree else {},
            )
            events.append(check_ev)
            yield check_ev
        return answer

    def _single_shot(self, question: str, tools, scout: LLM) -> Iterator[Event]:
        from .synthesize import _SYNTH_SYS

        files = tools.rank_files(question, top=self.settings.single_shot_reads)
        yield Event("rank", question)
        if not files:
            yield Event("answer", "")
            return
        for path in files:
            tools.read(path, 1)
            yield Event("read", f"{path} (top-ranked)")
        evidence = "\n\n".join(f"[{s}:{a}-{b}]\n{body}" for s, a, b, body in tools.reads)
        draft = scout.chat(
            [
                {"role": "system", "content": _SYNTH_SYS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nExcerpts:\n{evidence}\n\nAnswer:",
                },
            ],
            max_tokens=256,
        ).strip()
        yield Event("answer", draft)

    # ------------------------------------------------------------------ niceties
    def stats(self) -> dict:
        return {
            "corpus": str(self.corpus.root),
            **self.corpus.stats(),
            "model": self.settings.model,
            "variant": self.variant.name,
            "retriever": self.retriever,
        }

    def __repr__(self) -> str:
        s = self.corpus.stats()
        return (
            f"RagX(corpus={self.corpus.root.name!r}, files={s['files']}, "
            f"variant={self.variant.name!r}, model={self.settings.model!r})"
        )

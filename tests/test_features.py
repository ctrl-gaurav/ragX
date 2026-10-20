"""Tests for serialization, the semantic fallback wiring, and agent loop guards."""

import ragx.core as core
from ragx.agent import Agent
from ragx.config import Settings
from ragx.ingest import ingest
from ragx.tools import Toolbelt
from ragx.types import Answer, Citation, Event


def _belt(corpus_dir):
    return Toolbelt(corpus=ingest(corpus_dir, Settings(model="fake:m")))


def test_answer_to_dict_roundtrips():
    ans = Answer(
        text="Dana Whitfield founded Acme in 2014.",
        citations=[Citation("company.md", 1, 4, "Acme Robotics was founded in 2014")],
        grounded=True,
    )
    d = ans.to_dict()
    assert d["text"].startswith("Dana")
    assert d["sources"] == ["company.md:1-4"]
    assert d["citations"][0]["ref"] == "company.md:1-4"
    assert "trace" not in d
    assert "trace" in ans.to_dict(include_trace=True)


def test_event_and_citation_to_dict():
    assert Event("search", "Acme", step=2).to_dict()["kind"] == "search"
    assert str(Citation("a.md", 5, 5)) == "a.md:5"
    assert Citation("a.md", 5, 7).to_dict()["ref"] == "a.md:5-7"


def test_semantic_search_graceful_when_extra_missing(corpus_dir):
    t = _belt(corpus_dir)
    out = t.semantic_search("who founded the company")
    # sentence-transformers is not installed in the lean test env → friendly message
    assert "pyragx[semantic]" in out or out.startswith("Top semantic matches")


def test_agent_exposes_semantic_action_when_enabled(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    agent = Agent(
        FakeLLMCls(script=["SEMANTIC: founder", "ANSWER: x"]),
        t,
        max_steps=4,
        semantic_available=True,
    )
    kinds = [e.kind for e in agent.stream("who founded Acme?")]
    assert "semantic" in kinds


def test_semantic_action_hidden_by_default(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    llm = FakeLLMCls(script=["SEARCH: Acme", "ANSWER: x"])
    list(Agent(llm, t, max_steps=4).stream("q"))
    sys_prompt = llm.calls[0][0]["content"]
    assert "SEMANTIC" not in sys_prompt


def test_repeated_action_is_nudged(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    llm = FakeLLMCls(script=["SEARCH: Acme founder", "SEARCH: Acme founder", "ANSWER: done"])
    agent = Agent(llm, t, max_steps=5)
    list(agent.stream("who founded Acme?"))
    # the nudge is fed back as an observation to a later turn
    all_text = " ".join(m["content"] for call in llm.calls for m in call)
    assert "already tried this exact query" in all_text


def test_settings_semantic_flag_wires_agent(corpus_dir, monkeypatch, FakeLLMCls):
    script = ["SEMANTIC: founder", "ANSWER: x", "synthesized"]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(
        corpus_dir, model="fake:m", variant="classic", semantic_fallback=True, max_steps=4
    )
    ans = rag.ask("who founded Acme?")
    assert isinstance(ans.to_dict(), dict)


# --------------------------------------------------------------- P4: cost/latency
def test_tool_cache_and_rank_files(corpus_dir):
    cache: dict = {}
    t = Toolbelt(corpus=ingest(corpus_dir, Settings(model="fake:m")), cache=cache)
    out1 = t.search("Dana Whitfield")
    assert cache  # populated
    out2 = t.search("Dana Whitfield")
    assert out1 == out2
    assert "company.md" in t.rank_files("Acme founder Dana", top=2)


def test_surfaced_evidence_includes_unread_surfaced_files(corpus_dir):
    """The composer must see ALL surfaced evidence, not just explicit reads (the MuSiQue +F1 fix)."""
    t = _belt(corpus_dir)
    t.search("Acme")  # surfaces company.md via a search hit (not a READ)
    assert t.seen_files  # something was surfaced
    assert not t.reads  # but nothing was explicitly READ yet
    bundle = t.surfaced_evidence()
    assert bundle and any("company" in src for src, *_ in bundle)  # surfaced file is composed over


def test_surfaced_evidence_ranked_extras(corpus_dir):
    """With rank_query, the capped extras are df-BM25-ordered (deterministic, most-relevant first)."""
    t = _belt(corpus_dir)
    t.search("2021")  # surfaces both company.md and mars.txt without reading either
    assert len(t.seen_files) >= 2 and not t.reads
    bundle = t.surfaced_evidence(extra_cap=1, rank_query="Perseverance rover Jezero Crater")
    assert len(bundle) == 1
    assert "mars" in bundle[0][0]  # the query-relevant file wins the single extra slot


def test_answer_type_hint_rules():
    from ragx.synthesize import answer_type_hint

    assert "yes" in answer_type_hint("Were both Peter Duffell and Fred Niblo actors?")
    assert "ONE of the options" in answer_type_hint(
        "Which film was released more recently, The Vein or Now Barabbas?"
    )
    # "Is A or B ..." is an option question, not yes/no
    assert "ONE of the options" in answer_type_hint("Is Marsilea or Brabejum the larger genus?")
    assert "PLACE" in answer_type_hint("Where was the director of Blood Father born?")
    assert "date or year" in answer_type_hint("In what year was the drummer born?")
    assert "number" in answer_type_hint("How many times did the plague occur there?")
    assert "person" in answer_type_hint("Who is the spouse of the director of Dharam Karam?")
    assert answer_type_hint("Describe the plot of the film.") is None


def test_synthesize_typed_hint_reaches_prompt(corpus_dir, FakeLLMCls):
    from ragx.synthesize import synthesize

    t = _belt(corpus_dir)
    t.read("company.md", 1, 4)
    llm = FakeLLMCls(script=["Portland"])
    synthesize("Where was Acme Robotics founded?", "draft", t, answer_llm=llm)
    user_msg = llm.calls[0][-1]["content"]
    assert "PLACE" in user_msg
    llm2 = FakeLLMCls(script=["Portland"])
    synthesize("Where was Acme founded?", "draft", t, answer_llm=llm2, typed_hints=False)
    assert "PLACE" not in llm2.calls[0][-1]["content"]


def test_synthesize_compose_reasoning_two_calls(corpus_dir, FakeLLMCls):
    from ragx.synthesize import synthesize

    t = _belt(corpus_dir)
    t.read("company.md", 1, 4)
    llm = FakeLLMCls(script=["Founded in Portland in 1969; asked for the city.", "Portland"])
    ans = synthesize(
        "Where was Acme Robotics founded?", "draft", t, answer_llm=llm, compose_reasoning=True
    )
    assert ans.text == "Portland"
    assert len(llm.calls) == 2
    # call 1: free-form reasoning over the evidence
    assert "Reasoning:" in llm.calls[0][-1]["content"]
    assert llm.calls[0][0]["content"].startswith("You analyze a question")
    # call 2: strict compose over the question + the reasoning, not the raw excerpts
    second = llm.calls[1][-1]["content"]
    assert "Analysis of the source documents:" in second
    assert "Founded in Portland in 1969" in second
    # off by default: single call
    llm2 = FakeLLMCls(script=["Portland"])
    synthesize("Where was Acme Robotics founded?", "draft", t, answer_llm=llm2)
    assert len(llm2.calls) == 1


def test_llm_response_cache(monkeypatch):
    from ragx.llm import LLM

    monkeypatch.setattr(LLM, "_init_backend", lambda self: None)
    calls = {"n": 0}

    def fake(self, messages, mt, temp):
        calls["n"] += 1
        return ("hi", 0, 0)

    monkeypatch.setattr(LLM, "_chat_openai", fake)
    cache: dict = {}
    llm = LLM("openai:x", response_cache=cache)
    a = llm.chat([{"role": "user", "content": "q"}])
    b = llm.chat([{"role": "user", "content": "q"}])
    assert a == b == "hi"
    assert calls["n"] == 1  # second served from cache
    assert llm.usage.cache_hits == 1


def test_retry_recovers_from_transient_error(monkeypatch):
    """A rate-limit/overload blip is retried with backoff; the call then succeeds."""
    import ragx.llm as llm_mod
    from ragx.llm import LLM

    monkeypatch.setattr(LLM, "_init_backend", lambda self: None)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)  # don't actually wait
    attempts = {"n": 0}

    def flaky(self, messages, mt, temp):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("429 rate limit exceeded, please retry")
        return ("ok", 1, 1)

    monkeypatch.setattr(
        LLM, "_chat_openai", lambda self, m, mt, t: LLM._retry(self, lambda: flaky(self, m, mt, t))
    )
    llm = LLM("openai:x", max_retries=5)
    assert llm.chat([{"role": "user", "content": "q"}]) == "ok"
    assert attempts["n"] == 3  # failed twice, succeeded on the third


def test_retry_gives_up_on_non_retryable(monkeypatch):
    """A non-transient error (e.g. bad request) is surfaced immediately as LLMError."""
    import ragx.llm as llm_mod
    from ragx.llm import LLM, LLMError

    monkeypatch.setattr(LLM, "_init_backend", lambda self: None)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("400 invalid request: bad schema")

    llm = LLM("openai:x", max_retries=5)
    import pytest

    with pytest.raises(LLMError):
        llm._retry(boom)
    assert calls["n"] == 1  # not retried


def test_backoff_honors_retry_after(monkeypatch):
    from ragx.llm import _backoff_seconds

    class E(Exception):
        retry_after = 7

    assert _backoff_seconds(0, E()) == 7.0  # honored, not the exponential default


def test_prerank_hint_in_first_message(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    llm = FakeLLMCls(script=["ANSWER: x"])
    list(Agent(llm, t, max_steps=3, prerank_files=["company.md"]).stream("q"))
    first_user = llm.calls[0][1]["content"]
    assert "df-BM25 pre-rank" in first_user and "company.md" in first_user


def test_pricing_estimate_and_local_free():
    from ragx import pricing

    # hosted model: cost scales with tokens
    c = pricing.estimate_cost("openai:gpt-4o-mini", 1_000_000, 1_000_000)
    assert c == round(0.15 + 0.60, 6)
    # local models are free
    assert pricing.estimate_cost("hf:Qwen/Qwen3.5-4B", 1000, 1000) == 0.0
    # unknown model -> None (cost simply not shown)
    assert pricing.estimate_cost("mystery:model", 100, 100) is None
    # overridable at runtime; most-specific match wins
    pricing.set_price("mystery:model", 1.0, 2.0)
    assert pricing.estimate_cost("mystery:model", 1_000_000, 0) == 1.0


def test_usage_includes_cost(corpus_dir, monkeypatch, FakeLLMCls):
    llm = FakeLLMCls(spec="openai:gpt-4o-mini", script=["SEARCH: Acme", "ANSWER: x", "synthesized"])
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(corpus_dir, model="openai:gpt-4o-mini", variant="classic", max_steps=4)
    ans = rag.ask("who founded Acme?")
    assert "total_cost_usd" in ans.usage
    assert "cost_usd" in ans.usage["openai:gpt-4o-mini"]


def test_empty_question_short_circuits(corpus_dir, monkeypatch, FakeLLMCls):
    llm = FakeLLMCls(spec="fake:m", script=["should-not-run"])
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(corpus_dir, model="fake:m", variant="classic")
    ans = rag.ask("   ")
    assert ans.text == "" and ans.grounded is False
    assert llm.usage.calls == 0  # no LLM call wasted


def test_search_retrieval_only(corpus_dir, monkeypatch, FakeLLMCls):
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: FakeLLMCls(spec=spec))
    rag = core.RagX(corpus_dir, model="fake:m", variant="classic")
    files = rag.search("Acme founder Dana", top=2)
    assert "company.md" in files  # ranked without any LLM call


def test_adaptive_route_single_shot(corpus_dir, monkeypatch, FakeLLMCls):
    # single-shot draft, then synthesis pass — both ground in company.md
    script = [
        "Dana Whitfield founded Acme in 2014.",
        "Dana Whitfield founded Acme Robotics in 2014.",
    ]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(corpus_dir, model="fake:m", variant="classic", route="adaptive")
    ans = rag.ask("who founded Acme and when?")
    kinds = [e.kind for e in ans.trace]
    assert "rank" in kinds and "read" in kinds  # took the cheap path
    assert ans.grounded and "2014" in ans.text


def test_distill_picks_relevant_sentences(corpus_dir):
    t = _belt(corpus_dir)
    text = (
        "John Smith is a film director. He was born in Ohio in 1900. He liked cats. "
        "He directed many films. He died in Los Angeles in 1970."
    )
    out = t.distill("Where did the director die?", text)
    assert "died in Los Angeles" in out and "cats" not in out
    # relation lexicon bridges spouse<->married
    text2 = (
        "Dharam Karam is a 1975 film. The film is directed by Randhir Kapoor. "
        "Music was by Burman. Kapoor married actress Babita in 1971."
    )
    assert "married actress Babita" in t.distill("Who is the spouse of the director?", text2)


def test_resolve_chain_substitutes_entities(corpus_dir, FakeLLMCls):
    from ragx.agent.resolve import resolve_chain

    t = _belt(corpus_dir)
    llm = FakeLLMCls(
        script=[
            "Who founded Acme Robotics?\nWhere was #1 working in 2021?",  # decomposition
            "Dana Whitfield",  # hop 1 answer
            "Portland",  # hop 2 answer
        ]
    )
    events = list(resolve_chain("Where was the founder of Acme working in 2021?", t, llm, k=2))
    assert len(events) == 2 and all(e.kind == "hop" for e in events)
    # hop-2 question had the hop-1 entity substituted in
    hop2_prompt = llm.calls[2][-1]["content"]
    assert "Dana Whitfield" in hop2_prompt and "#1" not in hop2_prompt
    assert t.chain_notes and "-> Portland" in t.chain_notes[-1]
    assert t.seen_files  # per-hop ranked files joined the surfaced set
    # top hop docs are recorded for the composer (priority evidence): question-ranked extras
    # would drop them — a later-hop doc shares no vocabulary with the question by construction
    assert t.hop_evidence and set(t.hop_evidence) <= t.seen_files
    bundle = t.surfaced_evidence(extra_cap=4, priority=t.hop_evidence)
    bundled_srcs = [src for src, _a, _b, _body in bundle]
    assert all(src in bundled_srcs for src in t.hop_evidence)


def test_deep_route_end_to_end(corpus_dir, monkeypatch, FakeLLMCls):
    script = [
        "SEARCH: Acme founder",  # agent turn 1
        "ANSWER: Dana Whitfield",  # agent finishes
        "Who founded Acme Robotics?",  # decomposition (single hop)
        "Dana Whitfield",  # hop answer
        "Dana Whitfield founded Acme Robotics in 2014.",  # final synthesis
    ]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(corpus_dir, model="fake:m", variant="classic", route="deep")
    ans = rag.ask("Who founded Acme Robotics?")
    kinds = [e.kind for e in ans.trace]
    assert "hop" in kinds
    assert "2014" in ans.text
    # chain NOTES stay out of the synthesis prompt by default (deep_chain_notes=False) —
    # wrong hop answers mislead the composer; the hop documents still reach it as evidence
    assert not any("Resolved sub-questions" in m[-1]["content"] for m in llm.calls)


def test_deep_route_chain_notes_opt_in(corpus_dir, monkeypatch, FakeLLMCls):
    script = [
        "SEARCH: Acme founder",
        "ANSWER: Dana Whitfield",
        "Who founded Acme Robotics?",
        "Dana Whitfield",
        "Dana Whitfield founded Acme Robotics in 2014.",
    ]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(
        corpus_dir, model="fake:m", variant="classic", route="deep", deep_chain_notes=True
    )
    rag.ask("Who founded Acme Robotics?")
    assert any("Resolved sub-questions" in m[-1]["content"] for m in llm.calls)


def test_route_both_agreement_confidence(corpus_dir, monkeypatch, FakeLLMCls):
    # notes-injected compose and deep compose return the SAME answer -> note-robust -> high confidence
    script = [
        "SEARCH: Acme founder",
        "ANSWER: Dana Whitfield",
        "Who founded Acme Robotics?",
        "Dana Whitfield",
        "Dana Whitfield",  # notes-injected compose (route="both" extra call)
        "Dana Whitfield",  # deep compose
    ]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(corpus_dir, model="fake:m", variant="classic", route="both")
    ans = rag.ask("Who founded Acme Robotics?")
    assert ans.confidence == 0.9

    # disagreeing answers -> low confidence, and the alternative is surfaced in the trace
    script2 = [
        "SEARCH: Acme founder",
        "ANSWER: Dana Whitfield",
        "Who founded Acme Robotics?",
        "Dana Whitfield",
        "Marcus Bell",  # notes-injected compose disagrees
        "Dana Whitfield",  # deep compose
    ]
    llm2 = FakeLLMCls(spec="fake:m", script=script2)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm2)
    rag2 = core.RagX(corpus_dir, model="fake:m", variant="classic", route="both")
    ans2 = rag2.ask("Who founded Acme Robotics?")
    assert ans2.confidence == 0.5
    assert any(e.kind == "verify" and "disagree" in e.text for e in ans2.trace)

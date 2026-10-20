"""Phase-2 package features: grounding semantics, size-aware budgets, CoT auto-gate, sandboxing."""

import ragx.core as core
from ragx.agent import Agent
from ragx.config import Settings
from ragx.ingest import ingest
from ragx.synthesize import hint_kind, synthesize
from ragx.tools import Toolbelt


def _belt(corpus_dir):
    return Toolbelt(corpus=ingest(corpus_dir, Settings(model="fake:m")))


def test_grounded_means_answer_span_in_cited_excerpt(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    t.read("company.md", 1, 4)
    # answer occurs verbatim in the read excerpt → grounded, cited
    ans = synthesize(
        "Who founded Acme?", "draft", t, answer_llm=FakeLLMCls(script=["Dana Whitfield"])
    )
    assert ans.grounded and ans.citations and ans.citations[0].path == "company.md"
    # answer NOT in any excerpt → not grounded (citations still point at what was read)
    ans2 = synthesize(
        "Who founded Acme?", "draft", t, answer_llm=FakeLLMCls(script=["Zorblax Prime"])
    )
    assert not ans2.grounded


def test_citations_come_from_surfaced_extras_too(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    t.read("company.md", 1, 1)  # read only the heading line
    t.seen_files.add("mars.txt")  # surfaced but never read
    ans = synthesize(
        "When did Ingenuity fly?", "draft", t, answer_llm=FakeLLMCls(script=["April 19, 2021"])
    )
    assert ans.grounded
    assert any(c.path == "mars.txt" for c in ans.citations)


def test_budget_for_sizes():
    s = Settings(model="hf:Qwen/Qwen3.5-2B")
    assert s.budget_for() == (2, 8)
    s4 = Settings(model="hf:Qwen/Qwen3.5-4B")
    assert s4.budget_for() == (5, 16)
    s9 = Settings(model="hf:Qwen/Qwen3.5-9B", surf_cap=20)
    assert s9.budget_for() == (5, 20)
    off = Settings(model="hf:Qwen/Qwen3.5-2B", auto_budget=False)
    assert off.budget_for() == (5, 16)
    api = Settings(model="cerebras:gpt-oss-120b")
    assert api.budget_for() == (5, 16)  # 120b parses as 120 → large → explicit settings
    assert Settings(model="openai:gpt-4o-mini").budget_for() == (5, 16)


def test_hint_kind_and_cot_auto(corpus_dir, FakeLLMCls):
    assert hint_kind("How many years between the two events?") == "number"
    assert hint_kind("Which is older, Acme or Zeta?") == "comparison"
    assert hint_kind("Who founded Acme?") == "person"
    t = _belt(corpus_dir)
    t.read("company.md", 1, 4)
    # numeric question → reasoning pass (2 calls)
    llm = FakeLLMCls(script=["2021-2014=7", "7"])
    synthesize(
        "How many years after founding did Acme open Austin?",
        "d",
        t,
        answer_llm=llm,
        compose_reasoning="auto",
    )
    assert len(llm.calls) == 2
    # entity question → single strict compose
    llm2 = FakeLLMCls(script=["Dana Whitfield"])
    synthesize("Who founded Acme?", "d", t, answer_llm=llm2, compose_reasoning="auto")
    assert len(llm2.calls) == 1


def test_sandbox_wraps_observations(corpus_dir, FakeLLMCls):
    t = _belt(corpus_dir)
    llm = FakeLLMCls(script=["SEARCH: Acme", "ANSWER: x"])
    list(Agent(llm, t, max_steps=3, sandbox=True).stream("who founded Acme?"))
    assert "Security:" in llm.calls[0][0]["content"]
    obs_msgs = [
        m["content"] for call in llm.calls for m in call if m["content"].startswith("Observation:")
    ]
    assert obs_msgs and "<file_content>" in obs_msgs[0]
    llm2 = FakeLLMCls(script=["SEARCH: Acme", "ANSWER: x"])
    list(Agent(llm2, t, max_steps=3).stream("who founded Acme?"))
    assert "<file_content>" not in " ".join(m["content"] for call in llm2.calls for m in call)


def test_core_applies_auto_budget_and_sandbox(corpus_dir, monkeypatch, FakeLLMCls):
    llm = FakeLLMCls(
        spec="hf:Qwen/Qwen3.5-2B", script=["SEARCH: Acme", "SEARCH: Dana", "ANSWER: x", "x"]
    )
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)
    rag = core.RagX(
        corpus_dir,
        model="hf:Qwen/Qwen3.5-2B",
        variant="classic",
        sandbox_observations=True,
        max_steps=5,
    )
    rag.ask("who founded Acme?")
    # 2B → 2 foraging turns: the third scripted action is never requested as an agent turn
    turns = [
        c
        for c in llm.calls
        if c[0]["content"].startswith("You answer a question using ONLY a corpus")
    ]
    assert len(turns) <= 3  # 2 turns + the forced out-of-steps answer
    assert "Security:" in llm.calls[0][0]["content"]

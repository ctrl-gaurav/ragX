import ragx.core as core
from ragx.agent import Agent
from ragx.config import Settings
from ragx.ingest import ingest
from ragx.tools import Toolbelt


def test_agent_loop_and_citations(corpus_dir, FakeLLMCls):
    c = ingest(corpus_dir, Settings(model="fake:m"))
    tools = Toolbelt(corpus=c)
    script = [
        "THOUGHT: find the founder\nSEARCH: Acme Robotics founder",
        "READ: company.md:1-4",
        "ANSWER: Dana Whitfield",
    ]
    agent = Agent(FakeLLMCls(script=script), tools, max_steps=6)
    events = list(agent.stream("Who founded Acme Robotics?"))
    kinds = [e.kind for e in events]
    assert "search" in kinds and "read" in kinds and kinds[-1] == "answer"
    assert agent.answer_draft == "Dana Whitfield"
    assert any(r[0] == "company.md" for r in tools.reads)


def test_ragx_end_to_end_with_fake_llm(corpus_dir, monkeypatch, FakeLLMCls):
    script = [
        "SEARCH: Ingenuity first flight",
        "READ: mars.txt:1-2",
        "ANSWER: April 19, 2021",
        "Ingenuity made its first powered flight on April 19, 2021.",  # synthesis pass
    ]
    llm = FakeLLMCls(spec="fake:m", script=script)
    monkeypatch.setattr(core, "LLM", lambda spec, **kw: llm)

    rag = core.RagX(corpus_dir, model="fake:m", variant="classic", max_steps=6)
    ans = rag.ask("When did Ingenuity make its first flight?")
    assert "2021" in ans.text
    assert ans.citations and ans.citations[0].path == "mars.txt"
    assert ans.grounded

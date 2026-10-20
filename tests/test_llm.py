"""The provider client: effort suffixes, the thinking switch, length retries, custom endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from ragx.llm import LLM


class FakeCompletions:
    def __init__(self, replies, reject_template=False):
        self.replies = list(replies)
        self.kwargs = []
        self.reject_template = reject_template

    def create(self, **kw):
        self.kwargs.append(kw)
        if self.reject_template and "extra_body" in kw:
            raise ValueError("chat_template_kwargs not supported")
        text, finish = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=SimpleNamespace(completion_tokens=5, prompt_tokens=7),
        )


def _llm(spec, completions, monkeypatch, **kw):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = LLM(spec, max_retries=1, **kw)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm


MSG = [{"role": "user", "content": "hi"}]


def test_effort_suffix_is_parsed(monkeypatch):
    c = FakeCompletions([("ok", "stop")])
    llm = _llm("openai:o4-mini@high", c, monkeypatch)
    assert llm.model == "o4-mini" and llm.effort == "high"
    llm.chat(MSG)
    assert c.kwargs[0]["reasoning_effort"] == "high"
    assert LLM.__new__(LLM) is not None  # class is importable without a network


def test_length_cutoff_is_retried_with_a_larger_budget(monkeypatch):
    c = FakeCompletions([("B", "length"), ("Bristol", "stop")])
    llm = _llm("openai:gpt-oss-120b", c, monkeypatch)
    assert llm.chat(MSG, max_tokens=128) == "Bristol"
    assert [k["max_tokens"] for k in c.kwargs] == [128, 512]
    assert llm.usage.completion_tokens == 10 and llm.usage.prompt_tokens == 14


def test_vllm_thinking_switch_and_fallback(monkeypatch):
    c = FakeCompletions([("<think>hmm</think>Paris", "stop")])
    llm = _llm("vllm:Qwen/Qwen3.5-4B", c, monkeypatch, think=True, think_budget=100)
    assert llm.chat(MSG, max_tokens=50) == "Paris"
    assert c.kwargs[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert c.kwargs[0]["max_tokens"] == 150
    # a template that rejects the switch is asked again without it, once and for all
    c2 = FakeCompletions([("x", "stop"), ("y", "stop")], reject_template=True)
    llm2 = _llm("vllm:mistral", c2, monkeypatch)
    assert llm2.chat(MSG) == "x" and llm2.chat(MSG) == "y"
    assert "extra_body" not in c2.kwargs[-1]


def test_empty_thinking_answer_falls_back_to_no_thinking(monkeypatch):
    c = FakeCompletions([("<think>long</think>", "stop"), ("42", "stop")])
    llm = _llm("vllm:Qwen/Qwen3.5-4B", c, monkeypatch, think=True)
    assert llm.chat(MSG) == "42"
    assert c.kwargs[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_base_url_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    llm = LLM("openai:some-model")
    assert str(llm._client.base_url).startswith("https://example.invalid/v1")


def test_hosted_think_adds_no_budget_and_rejected_retry_keeps_answer(monkeypatch):
    class Capped(FakeCompletions):
        def create(self, **kw):
            if kw["max_tokens"] > 600:
                self.kwargs.append(kw)
                raise ValueError("413 non-streaming request asked for too many tokens")
            return super().create(**kw)

    c = Capped([("Bris", "length"), ("Bristol", "length")])
    llm = _llm("openai:gpt-oss-120b", c, monkeypatch, think=True, think_budget=8192)
    assert llm.chat(MSG, max_tokens=128) == "Bristol"
    assert [k["max_tokens"] for k in c.kwargs] == [128, 512, 2048]

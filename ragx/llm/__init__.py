"""Provider-agnostic LLM client.

One interface, many backends. API providers are OpenAI-compatible (Cerebras, Groq,
Together, Fireworks, OpenAI) or have thin adapters (Google Gemini, Anthropic). Local
models run via in-process transformers (``hf:``) or a vLLM OpenAI server (``vllm:``).

Keys are read from the environment by name only — never logged.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

# provider -> (base_url, env_key)
_OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
}

# substrings (matched against repr(exc).lower()) that mark a transient, worth-retrying failure
_RETRYABLE = (
    "rate",
    "429",
    "overload",
    "timeout",
    "timed out",
    "503",
    "502",
    "500",
    "connection",
    "limit",
    "529",
    "temporarily",
    "unavailable",
    "econnreset",
)

Message = dict[str, str]
T = TypeVar("T")
_EFFORTS = ("minimal", "low", "medium", "high")
_MAX_RETRY_BUDGET = 8000
_THINK_TAGS = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


def _is_retryable(exc: Exception) -> bool:
    """A transient error (rate limit, overload, network blip) worth backing off and retrying."""
    m = repr(exc).lower()
    if "413" in m or "context length" in m or "maximum context" in m:
        return False  # the request itself is too large; retrying cannot help
    return any(k in m for k in _RETRYABLE)


def _backoff_seconds(attempt: int, exc: Exception | None = None) -> float:
    """Exponential backoff with full jitter, honoring a ``Retry-After`` hint when present."""
    if exc is not None:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            hdrs = getattr(getattr(exc, "response", None), "headers", None)
            if hdrs is not None:
                retry_after = hdrs.get("retry-after") or hdrs.get("Retry-After")
        try:
            if retry_after is not None:
                return min(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
    return min(1.5 * (2**attempt), 45.0) + random.uniform(0.0, 0.75)


@dataclass
class Usage:
    calls: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    latency_s: float = 0.0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "latency_s": round(self.latency_s, 3),
            "cache_hits": self.cache_hits,
        }


@dataclass
class LLM:
    """A chat model identified by ``provider:model``."""

    spec: str
    temperature: float = 0.0
    max_tokens: int = 1024
    usage: Usage = field(default_factory=Usage)
    response_cache: dict[str, str] | None = None  # set to a dict to memoize identical calls
    max_retries: int = field(default_factory=lambda: int(os.environ.get("RAGX_MAX_RETRIES", "8")))
    think: bool = False  # thinking mode (vLLM chat templates, Anthropic, local hf models)
    think_budget: int = 8192  # extra tokens a thinking call may spend

    def __post_init__(self) -> None:
        if ":" not in self.spec:
            raise ValueError(f"model spec must be 'provider:model', got {self.spec!r}")
        self.provider, self.model = self.spec.split(":", 1)
        # ``provider:model@effort`` sets the reasoning effort of hosted reasoning models
        self.effort: str | None = None
        if "@" in self.model and self.model.rsplit("@", 1)[1] in _EFFORTS:
            self.model, self.effort = self.model.rsplit("@", 1)
        self._template_kwargs_ok = True
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._client: Any = None
        self._hf = None
        self._tok = None
        self._init_backend()

    # ------------------------------------------------------------------ backends
    def _init_backend(self) -> None:
        p = self.provider
        if p == "vllm":
            from openai import OpenAI

            base = os.environ.get("RAGX_VLLM_URL", "http://localhost:8001/v1")
            key = os.environ.get("RAGX_VLLM_API_KEY", "EMPTY")
            self._client = OpenAI(base_url=base, api_key=key, timeout=300, max_retries=0)
        elif p in _OPENAI_COMPATIBLE:
            from openai import OpenAI

            base, key = _OPENAI_COMPATIBLE[p]
            if p == "openai":  # any OpenAI-compatible endpoint that takes an OpenAI-style key
                base = os.environ.get("OPENAI_BASE_URL", base)
            if not os.environ.get(key):
                raise LLMError(
                    f"missing {key} for provider '{p}'. Set it in your environment/.env."
                )
            self._client = OpenAI(
                base_url=base, api_key=os.environ[key], timeout=120, max_retries=0
            )
        elif p == "google":
            self._init_google()
        elif p == "anthropic":
            self._init_anthropic()
        elif p == "hf":
            self._init_hf()
        else:
            raise ValueError(f"unknown provider '{p}' in spec {self.spec!r}")

    def _init_google(self) -> None:
        import google.generativeai as genai  # type: ignore

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("missing GOOGLE_API_KEY for provider 'google'.")
        genai.configure(api_key=key)
        self._client = genai

    def _init_anthropic(self) -> None:
        import anthropic  # type: ignore

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("missing ANTHROPIC_API_KEY for provider 'anthropic'.")
        self._client = anthropic.Anthropic()

    def _init_hf(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise LLMError(
                "local 'hf:' models need transformers + torch. Install with: "
                "pip install 'pyragx[local]'"
            ) from e

        device = os.environ.get("RAGX_HF_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model)
        self._hf = AutoModelForCausalLM.from_pretrained(
            self.model, torch_dtype=torch.bfloat16, device_map=device
        )
        self._hf.eval()

    # ------------------------------------------------------------------ public
    def chat(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        mt = max_tokens or self.max_tokens
        temp = self.temperature if temperature is None else temperature
        if self.response_cache is not None:
            import hashlib
            import json as _json

            ckey = hashlib.sha1(
                _json.dumps([self.spec, messages, mt, temp], sort_keys=True).encode()
            ).hexdigest()
            if ckey in self.response_cache:
                with self._lock:
                    self.usage.cache_hits += 1
                return self.response_cache[ckey]
        else:
            ckey = None
        t0 = time.time()
        try:
            if self.provider in _OPENAI_COMPATIBLE or self.provider == "vllm":
                out, ct, pt = self._chat_openai(messages, mt, temp)
            elif self.provider == "google":
                out, ct, pt = self._chat_google(messages, mt, temp)
            elif self.provider == "anthropic":
                out, ct, pt = self._chat_anthropic(messages, mt, temp)
            else:  # hf
                out, ct, pt = self._chat_hf(messages, mt, temp), 0, 0
        finally:
            dt = time.time() - t0
            with self._lock:
                self.usage.calls += 1
                self.usage.latency_s += dt
        with self._lock:
            self.usage.completion_tokens += ct
            self.usage.prompt_tokens += pt
        if ckey is not None:
            self.response_cache[ckey] = out
        return out

    # ------------------------------------------------------------------ retries
    def _retry(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` with exponential backoff + jitter on transient API failures.

        Retries rate-limit / overload / network errors (honoring ``Retry-After`` when the
        provider sends it); re-raises everything else immediately as an :class:`LLMError`.
        """
        last: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last = e
                if _is_retryable(e) and attempt < self.max_retries - 1:
                    time.sleep(_backoff_seconds(attempt, e))
                    continue
                raise LLMError(str(e)) from e
        raise LLMError(str(last))  # pragma: no cover - loop always returns or raises

    # ------------------------------------------------------------------ impls
    def _chat_openai(self, messages, mt, temp) -> tuple[str, int, int]:
        # thinking is a chat-template switch on a vLLM server; hosted reasoning models think on
        # their own (set ``@effort`` on the model string), so only vLLM gets the extra budget
        template_think = self.provider == "vllm"

        def _call(think: bool, budget: int) -> tuple[str, int, int, str]:
            kw: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": budget + (self.think_budget if think and template_think else 0),
                "temperature": temp,
            }
            if self.effort:
                kw["reasoning_effort"] = self.effort
            # chat templates that think by default (Qwen3 family) are told explicitly; a server
            # whose template rejects the switch is asked again without it
            if self.provider == "vllm" and self._template_kwargs_ok:
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think}}
            try:
                r = self._client.chat.completions.create(**kw)
            except Exception as e:  # noqa: BLE001
                if "extra_body" in kw and "chat_template" in repr(e).lower():
                    self._template_kwargs_ok = False
                    kw.pop("extra_body")
                    r = self._client.chat.completions.create(**kw)
                else:
                    raise
            choice = r.choices[0]
            content = _THINK_TAGS.sub("", choice.message.content or "").strip()
            ct = getattr(r.usage, "completion_tokens", 0) if r.usage else 0
            pt = getattr(r.usage, "prompt_tokens", 0) if r.usage else 0
            return content, ct, pt, getattr(choice, "finish_reason", "") or ""

        # hosted reasoning models spend max_tokens on hidden reasoning first, so a reply cut off
        # by the length limit is asked again with a larger budget (up to 8,000 tokens)
        out, c, p, finish = self._retry(lambda: _call(self.think, mt))
        ct, pt, budget = c, p, mt
        while finish == "length" and budget < _MAX_RETRY_BUDGET:
            budget = min(_MAX_RETRY_BUDGET, budget * 4)
            try:
                out2, c, p, finish = self._retry(lambda b=budget: _call(self.think, b))
            except LLMError:  # the provider caps output tokens below this budget: keep what we have
                break
            out, ct, pt = out2, ct + c, pt + p
        if not out and self.think and template_think:
            # a thinking call that spent its whole budget thinking: answer once more without it
            out, c, p, _ = self._retry(lambda: _call(False, mt))
            ct, pt = ct + c, pt + p
        return out, ct, pt

    def _chat_google(self, messages, mt, temp) -> tuple[str, int, int]:
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        hist = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
        )

        def _call() -> tuple[str, int, int]:
            model = self._client.GenerativeModel(self.model, system_instruction=sys or None)
            r = model.generate_content(
                hist, generation_config={"temperature": temp, "max_output_tokens": mt}
            )
            return (r.text or "").strip(), 0, 0

        return self._retry(_call)

    def _chat_anthropic(self, messages, mt, temp) -> tuple[str, int, int]:
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]

        def _call() -> tuple[str, int, int]:
            kw: dict[str, Any] = {"model": self.model, "messages": conv, "max_tokens": mt}
            if sys:
                kw["system"] = sys
            if self.think:  # extended thinking requires the default temperature
                kw["thinking"] = {"type": "enabled", "budget_tokens": self.think_budget}
                kw["max_tokens"] = mt + self.think_budget
            else:
                kw["temperature"] = temp
            r = self._client.messages.create(**kw)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            u = getattr(r, "usage", None)
            return text.strip(), getattr(u, "output_tokens", 0), getattr(u, "input_tokens", 0)

        return self._retry(_call)

    def _hf_prompt(self, messages: list[Message]) -> str:
        """Render the chat template. Disable Qwen3+ "thinking" mode; fold a system role into
        the first user turn for templates that reject it (e.g. Mistral)."""
        for kw in ({"enable_thinking": self.think}, {}):
            try:
                return self._tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kw
                )
            except TypeError:
                continue
            except Exception:  # noqa: BLE001
                break
        merged: list[Message] = []
        sys_txt = ""
        for m in messages:
            if m["role"] == "system":
                sys_txt += m["content"] + "\n\n"
            elif m["role"] == "user" and sys_txt:
                merged.append({"role": "user", "content": sys_txt + m["content"]})
                sys_txt = ""
            else:
                merged.append(m)
        if sys_txt and merged:
            merged[0] = {"role": "user", "content": sys_txt + merged[0]["content"]}
        return self._tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def _chat_hf(self, messages, mt, temp) -> str:
        import re

        import torch  # type: ignore

        prompt = self._hf_prompt(messages)
        ins = self._tok(prompt, return_tensors="pt").to(self._hf.device)
        kw: dict[str, Any] = {
            "max_new_tokens": mt + (self.think_budget if self.think else 0),
            "pad_token_id": self._tok.eos_token_id,
        }
        if temp and temp > 0:
            kw.update(do_sample=True, temperature=temp)
        else:
            kw.update(do_sample=False)
        with self._gen_lock, torch.no_grad():
            out = self._hf.generate(**ins, **kw)
        text = self._tok.decode(out[0][ins.input_ids.shape[1] :], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


__all__ = ["LLM", "LLMError", "Usage"]

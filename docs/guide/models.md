# Models and providers

A model is named `provider:model`:

```python
RagX("./docs", model="openai:gpt-4o-mini")
RagX("./docs", model="anthropic:claude-3-5-sonnet-latest")
RagX("./docs", model="vllm:Qwen/Qwen3.5-9B")
```

## Providers

| Provider | Example | Key | Notes |
|---|---|---|---|
| `openai` | `openai:gpt-4o-mini` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` points it at any OpenAI-compatible endpoint that takes a key |
| `anthropic` | `anthropic:claude-3-5-sonnet-latest` | `ANTHROPIC_API_KEY` | needs `pyragx[anthropic]` |
| `google` | `google:gemini-1.5-flash` | `GOOGLE_API_KEY` | needs `pyragx[google]` |
| `groq` | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` | |
| `together` | `together:Qwen/Qwen2.5-7B-Instruct-Turbo` | `TOGETHER_API_KEY` | |
| `cerebras` | `cerebras:gpt-oss-120b` | `CEREBRAS_API_KEY` | |
| `fireworks` | `fireworks:accounts/fireworks/models/...` | `FIREWORKS_API_KEY` | |
| `vllm` | `vllm:Qwen/Qwen3.5-9B` | `RAGX_VLLM_API_KEY` (optional) | any OpenAI-compatible server at `RAGX_VLLM_URL` (default `http://localhost:8001/v1`) |
| `hf` | `hf:Qwen/Qwen3.5-4B` | — | runs in-process with transformers; needs `pyragx[local]`; uses a GPU when present |

Without `model=`, ragX uses the first provider whose key is set, in the order OpenAI, Anthropic,
Google, Groq, Together, Cerebras, and otherwise `RAGX_MODEL` or `hf:Qwen/Qwen3.5-4B`.

## A separate answer model

The forager and the slot chain make many short calls; the compose call makes one long one. You can
give them different models:

```python
RagX(
    "./docs",
    scout_model="groq:llama-3.3-70b-versatile",   # forage loop and slot chain
    answer_model="openai:gpt-4o",                  # the compose call
)
```

## Reasoning effort

Append `@minimal`, `@low`, `@medium` or `@high` to a hosted reasoning model to set its reasoning
effort:

```python
RagX("./docs", model="openai:o4-mini@low")
```

Hosted reasoning models spend part of their output budget on hidden reasoning. When a reply is cut
off by the length limit, ragX asks again with a larger budget, up to 8,000 tokens or the provider's
limit.

## Thinking

`think=True` (or the `ragx-think` variant) turns on a model's thinking mode in every call:

- **vLLM servers:** ragX sets the chat template's `enable_thinking` switch (used by the Qwen3
  family) and adds `think_budget` tokens (8,192 by default) to each call. Without `think=True` it
  sets the switch to off, so models that think by default answer directly. Servers whose template
  rejects the switch are detected and asked without it.
- **Local `hf:` models:** the same switch is passed to the chat template.
- **Anthropic:** extended thinking is enabled with `think_budget` budget tokens.
- **Other hosted models** decide for themselves; use `@effort` instead.

If a thinking call spends its whole budget and returns no answer, ragX asks once more with
thinking off.

## Local models

With a vLLM server:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8001
```

```python
RagX("./docs", model="vllm:Qwen/Qwen3.5-9B")
```

In-process (no server):

```python
RagX("./docs", model="hf:Qwen/Qwen3.5-4B")   # pip install "pyragx[local]"
```

ragX assumes a 32,768-token context for `vllm:` and `hf:` models and trims the composed evidence to
fit; set `max_context_tokens=` to match your server. Very small models (below about 4B parameters)
often fail to follow the action format of the forage loop.

## Usage and cost

`answer.usage` records calls, prompt and completion tokens and latency per model, an estimated cost
in USD for priced hosted models, and the number of LLM calls the run made.

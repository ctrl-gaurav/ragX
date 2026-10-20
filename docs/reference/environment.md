# Environment variables

ragX loads a `.env` file from the working directory or any parent before reading these. Keys are
read by name and never logged.

## Provider keys

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `openai:` |
| `OPENAI_BASE_URL` | `openai:` against any OpenAI-compatible endpoint that takes a key |
| `ANTHROPIC_API_KEY` | `anthropic:` |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | `google:` |
| `GROQ_API_KEY` | `groq:` |
| `TOGETHER_API_KEY` | `together:` |
| `CEREBRAS_API_KEY` | `cerebras:` |
| `FIREWORKS_API_KEY` | `fireworks:` |

## Local models

| Variable | Default | Meaning |
|---|---|---|
| `RAGX_VLLM_URL` | `http://localhost:8001/v1` | the vLLM (or other OpenAI-compatible) server for `vllm:` |
| `RAGX_VLLM_API_KEY` | `EMPTY` | its key, when the server requires one |
| `RAGX_HF_DEVICE` | `cuda:0` when available, else `cpu` | device for `hf:` models |
| `RAGX_MODEL` | `hf:Qwen/Qwen3.5-4B` | the default model when no provider key is set |

## Behaviour

| Variable | Default | Meaning |
|---|---|---|
| `RAGX_MAX_RETRIES` | 8 | retries on rate limits, overloads and network errors, with exponential backoff |
| `RAGX_SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | embedder of the classic agent's `SEMANTIC` tool |

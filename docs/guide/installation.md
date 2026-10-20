# Installation

ragX needs Python 3.10 or newer.

```bash
pip install pyragx
```

The distribution is named `pyragx` on PyPI; the package you import and the command you run are
both `ragx`.

The base install runs the default variant, `ragx-lexical`, which ranks with BM25 and needs no
embedding model, no vector database and no GPU.

## Optional extras

Install extras for the parts you use:

| Extra | Install | What it adds |
|---|---|---|
| `dense` | `pip install "pyragx[dense]"` | sentence-transformers and NumPy, for the dense, fused and rerank rankers (`ragx`, `ragx-core`, `ragx-fused`, `ragx-rerank`, `ragx-fast`, `ragx-think`) |
| `anthropic` | `pip install "pyragx[anthropic]"` | the Anthropic SDK, for `anthropic:` models |
| `google` | `pip install "pyragx[google]"` | the Google Generative AI SDK, for `google:` models |
| `local` | `pip install "pyragx[local]"` | transformers, PyTorch and Accelerate, to run `hf:` models in-process |
| `server` | `pip install "pyragx[server]"` | FastAPI and Uvicorn, for `ragx serve` |
| `ui` | `pip install "pyragx[ui]"` | Textual, for the full-screen terminal view |
| `all` | `pip install "pyragx[all]"` | everything above |

OpenAI-compatible providers (OpenAI, Groq, Together, Cerebras, Fireworks and any vLLM server) work
with the base install.

## Set a model

ragX reads provider keys from the environment, or from a `.env` file in the working directory or
any parent. It never logs them.

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ...
```

Without `model=`, ragX picks a default from the first key it finds (see
[Models and providers](./models)). With no key at all it falls back to a small local model, which
needs the `local` extra.

## Optional tools

The [classic agent](./classic) uses [ripgrep](https://github.com/BurntSushi/ripgrep) for exact
search when it is on the `PATH` (and falls back to a pure-Python search when it is not), and
[ugrep](https://github.com/Genivia/ugrep) for fuzzy search. The ragX variants need neither.

## Verify

```bash
ragx --version
ragx variants
```

# Changelog

## 0.0.1

The first public release.

- **The ragX method:** the forage loop (`RANK`, `READ`), the slot chain and one compose call, with
  the paper's prompts and frozen settings.
- **Seven variants:** `ragx`, `ragx-core`, `ragx-lexical` (default), `ragx-fused`, `ragx-rerank`,
  `ragx-fast` and `ragx-think`, plus the `classic` file-level agent.
- **Rankers:** BM25, dense (any sentence-transformers model, cached embeddings), reciprocal-rank
  fusion, and a cross-encoder reranker.
- **Passages** with `path:start-end` names, and citations grounded in the passages that contain the
  answer.
- **Providers:** OpenAI (and any OpenAI-compatible endpoint), Anthropic, Google, Groq, Together,
  Cerebras, Fireworks, vLLM and local Hugging Face models; reasoning effort and thinking mode.
- **Interfaces:** Python API with streaming, the `ragx` command line, and an OpenAI-compatible
  server.
- **Security:** tool output and evidence fenced as untrusted data by default.

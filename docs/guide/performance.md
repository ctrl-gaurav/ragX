# Cost and speed

A ragX run costs more than one retrieval call, and buys its accuracy with that. This page shows where
the calls and tokens go and how to trade them.

## Where the calls go

| Stage | Calls | Prompt size |
|---|---|---|
| Forage loop | up to 10 (one per turn) | grows with the conversation; each `READ` adds up to 1,500 characters |
| Slot chain | 1 decomposition + 1 per hop (at most 4) | short: a sub-question and four distilled passages |
| Compose | 1 | the largest: up to 20 whole passages, plus the backfill in `ragx` |

In the paper's benchmark on all of Wikipedia, a ragX run averaged 12.6 calls per multi-hop
question and, across eight models, about 17 thousand prompt tokens: fewer tokens than a 160-passage single-shot context (about 26
thousand), and 22 times fewer than DCI-Agent-Lite.

## Making it cheaper

- **`ragx-fast`** drops the slot chain: no decomposition or hop calls.
- **No backfill.** `ragx-core` and `ragx-lexical` compose without the 48 backfill passages, the
  largest part of the `ragx` compose prompt.
- **Fewer turns.** `steps=5` halves the forage loop's maximum.
- **A cheaper scout.** Run the many short calls on a small, fast model and only the compose call on
  a strong one:

  ```python
  RagX("./docs", scout_model="groq:llama-3.1-8b-instant", answer_model="openai:gpt-4o")
  ```

- **Caching.** `cache=True` memoises identical model calls and tool results across questions in the
  same process, which helps when similar questions repeat.

## Making it faster

- Serve local models with vLLM rather than in-process `hf:`; the forage loop is sequential, so
  per-call latency matters more than throughput.
- Run the embedder and reranker on a GPU (`device="cuda"`); on a CPU the cross-encoder of
  `ragx-rerank` dominates the time of each `RANK`.
- Pre-build the mirror and embeddings with `ragx ingest` before serving.

## Watching cost

`answer.usage` reports, per model, the calls, prompt and completion tokens, latency and an estimated
cost in USD for priced hosted models, plus `llm_calls` for the run and `total_cost_usd`.

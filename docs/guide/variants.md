# Variants

All variants share the prompts and settings of [the method](./how-it-works). They differ in the
ranker behind `RANK`, whether the slot chain runs, whether the backfill is added, and whether the
model thinks.

| Variant | Ranker | Stages | Backfill | Needs |
|---|---|---|---|---|
| `ragx` | dense | forage loop + slot chain | 48 | `pyragx[dense]` |
| `ragx-core` | dense | forage loop + slot chain | 0 | `pyragx[dense]` |
| `ragx-lexical` *(default)* | BM25 | forage loop + slot chain | 0 | nothing extra |
| `ragx-fused` | BM25 + dense (RRF) | forage loop + slot chain | 0 | `pyragx[dense]` |
| `ragx-rerank` | dense top 100 → cross-encoder | forage loop + slot chain | 0 | `pyragx[dense]` |
| `ragx-fast` | dense | forage loop only | 0 | `pyragx[dense]` |
| `ragx-think` | dense | forage loop + slot chain | 48 | `pyragx[dense]`, a thinking model |
| `classic` | file-level BM25 + ripgrep | [the classic agent](./classic) | — | nothing extra |

Select one with `variant=`:

```python
from ragx import RagX

rag = RagX("./docs", variant="ragx-rerank")
```

or on the command line with `--variant` / `-v`. `ragx-lex` and `lexical` are accepted as aliases of
`ragx-lexical`.

## Choosing a variant

**Start with `ragx-lexical`.** It needs no embedding model, ingests instantly and, in the paper's
benchmark, beats every dense single-shot baseline on all of Wikipedia.

**Use `ragx-rerank` for the best accuracy.** It was the most accurate variant in the benchmark, at
the cost of running a cross-encoder over 100 candidates per `RANK`. On a CPU that is slow for long
passages; a GPU helps.

**Use `ragx` for the most complete pipeline.** It adds the 48-passage backfill, which pays off on
large corpora where the question's own top passages hold part of the evidence, and costs more
prompt tokens per question.

**Use `ragx-fused`** when your documents mix exact identifiers (codes, names, numbers) that BM25
matches well with paraphrased text that a dense index matches well.

**Use `ragx-fast`** when cost matters most: it skips the slot chain and makes the fewest calls.

**Use `ragx-think`** with a model that has a thinking mode served through vLLM (for example the
Qwen3 family) or with Anthropic models; see [thinking](./models#thinking).

## Mixing settings

A variant sets defaults; any setting can override them:

```python
# the full method with BM25 instead of the dense index
rag = RagX("./docs", variant="ragx", retriever="bm25")

# ragx-core with a smaller backfill
rag = RagX("./docs", variant="ragx-core", backfill=16)
```

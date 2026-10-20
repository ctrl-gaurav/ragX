# Retrievers

The ranker answers every `RANK`, every hop of the slot chain and the backfill. Each variant picks
one; `retriever=` overrides it.

## BM25

```python
RagX("./docs", retriever="bm25")
```

Okapi BM25 over each passage's title and text, from an in-memory inverted index built when the
first question is asked. It needs no extra dependencies and no GPU, and handles corpora of tens of
thousands of passages comfortably.

## Dense

```python
RagX("./docs", retriever="dense", embed_model="Qwen/Qwen3-Embedding-0.6B", device="cuda")
```

Exact inner-product search over normalised passage embeddings from any
[sentence-transformers](https://www.sbert.net/) model. The default is the paper's embedder,
`Qwen/Qwen3-Embedding-0.6B`. Queries use the model's `query` prompt when the model defines one.

Embeddings are cached in `<corpus>/.ragx/embeddings/` keyed by passage content, so the corpus is
embedded once and, after a change, only new or edited passages are embedded again.

Requires `pip install "pyragx[dense]"`. Set `device="cpu"` or `device="cuda"` to choose where the
embedder runs; by default sentence-transformers picks a GPU when one is available.

## Fused

```python
RagX("./docs", retriever="fused")
```

Reciprocal-rank fusion of the BM25 and dense rankings: each passage scores `1 / (60 + rank)` in
each of the two top-100 lists, and the sums are ranked. It combines exact-term matching with
semantic matching.

## Rerank

```python
RagX("./docs", retriever="rerank", rerank_model="BAAI/bge-reranker-v2-m3", rerank_depth=100)
```

The dense top `rerank_depth` passages (100 by default) are scored by a cross-encoder, default
`BAAI/bge-reranker-v2-m3`, and reordered. The cross-encoder's input is capped at the model's
position limit, so models with 512 positions work as well as long-context ones.

## Retrieval without a model

`RagX.search` runs the ranker alone and makes no LLM call, which is useful for checking retrieval:

```python
rag = RagX("./docs", variant="ragx-fused")
rag.search("Ashford University founder", top=5)
# ['ashford_university.md:1-5', 'thomas_wren.md:1-5', ...]
```

```bash
ragx search "Ashford University founder" -c ./docs -v ragx-fused -k 5
```

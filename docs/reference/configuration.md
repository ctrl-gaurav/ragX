# Configuration

Every setting is a keyword argument of `RagX` (or a field of `ragx.Settings`):

```python
RagX("./docs", variant="ragx", steps=8, backfill=24, embed_model="BAAI/bge-m3", device="cuda")
```

## Models

| Setting | Default | Meaning |
|---|---|---|
| `model` | from the environment | `provider:model`; see [Models and providers](../guide/models) |
| `scout_model` | `model` | model for the forage loop and the slot chain |
| `answer_model` | `model` | model for the compose call |
| `think` | the variant's (`ragx-think`: on) | thinking mode in every call |
| `think_budget` | 8192 | extra tokens a thinking call may spend |
| `max_context_tokens` | 32768 for `vllm:`/`hf:`, none otherwise | the reader's context; the evidence is trimmed to fit |
| `temperature` | 0.0 | sampling temperature |
| `cache` | `False` | memoise identical model calls and tool results within the process |

## The method

| Setting | Default | Meaning |
|---|---|---|
| `variant` | `ragx-lexical` | one of `ragx`, `ragx-core`, `ragx-lexical`, `ragx-fused`, `ragx-rerank`, `ragx-fast`, `ragx-think`, `classic` |
| `retriever` | the variant's | override the ranker: `bm25`, `dense`, `fused`, `rerank` |
| `backfill` | the variant's (48 or 0) | the question's own top passages appended at compose |
| `steps` | 10 | forager turns |
| `rank_k` | 8 | passages listed per `RANK` |
| `hop_k` | 4 | passages retrieved per hop |
| `max_hops` | 4 | sub-questions at most |
| `read_cap` | 8 | read passages composed |
| `extra_cap` | 12 | extra passages composed (hop passages first) |
| `tools` | `["RANK", "READ"]` | forager tools; add `"SEARCH"` for line search |
| `fence` | `True` | fence tool output and evidence as untrusted data |
| `compose_reasoning` | `False` | a reasoning pass before the compose: `True`, or `"auto"` for number, date and comparison questions |
| `typed_hints` | `True` | the answer-type hint in the compose prompt |

## Retrieval

| Setting | Default | Meaning |
|---|---|---|
| `embed_model` | `Qwen/Qwen3-Embedding-0.6B` | sentence-transformers embedder for dense ranking |
| `rerank_model` | `BAAI/bge-reranker-v2-m3` | cross-encoder for `rerank` |
| `rerank_depth` | 100 | dense candidates the reranker (and fusion) consider |
| `device` | automatic | `"cpu"`, `"cuda"`, ... for the embedder and reranker |
| `passage_words` | 120 | passage size when files are cut |

## Answers

| Setting | Default | Meaning |
|---|---|---|
| `verify_citations` | `True` | cite only passages that contain the answer ([grounding](../guide/citations)) |
| `abstain_on_weak_evidence` | `False` | allow "I don't know" and replace ungrounded answers with a refusal |

## Ingestion

| Setting | Default | Meaning |
|---|---|---|
| `cache_dir` | `<corpus>/.ragx` | where the mirror, index and embeddings live |
| `include` | `[]` (everything supported) | glob allowlist |
| `exclude` | `[".git/**", "node_modules/**", ".ragx/**"]` | glob blocklist |

## Classic agent

Used only with `variant="classic"`; see [The classic agent](../guide/classic).

| Setting | Default | Meaning |
|---|---|---|
| `route` | `agentic` | `agentic`, `adaptive`, `deep` or `both` |
| `max_steps` | 5 | tool-loop turns |
| `auto_budget` | `True` | fewer turns and extras for scouts of 2.5B parameters or less |
| `max_result_lines` | 40 | search lines shown per call |
| `max_read_chars` | 4000 | characters per read |
| `prerank` | `True` | seed the loop with a BM25 ranking of likely files |
| `single_shot_reads` | 4 | files the `adaptive` route reads first; files per hop in `deep` |
| `deep_hops` | 4 | sub-questions in `deep` |
| `deep_surf_cap` | 12 | surfaced files composed in `deep` |
| `deep_chain_notes` | `False` | include the resolved chain in the compose prompt |
| `surf_cap` | 16 | surfaced files composed in `agentic` |
| `compose_surfaced` | `True` | compose over surfaced files, not only reads |
| `semantic_fallback` | `False` | offer the `SEMANTIC` tool (`pyragx[semantic]`) |
| `fuzzy` | `True` | offer `FUZZY` when ugrep is installed |
| `context_lines` | 1 | lines of context around each search hit |
| `sandbox_observations` | `False` | fence tool output as quoted data |

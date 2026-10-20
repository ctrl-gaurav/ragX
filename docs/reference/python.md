# Python API

```python
from ragx import RagX, Answer, Citation, Event, Settings, VARIANTS, get_variant
```

## `RagX`

```python
RagX(corpus, model=None, variant=None, **settings)
```

| Parameter | Type | Meaning |
|---|---|---|
| `corpus` | `str \| Path` | a file or a folder; ingested on construction |
| `model` | `str \| None` | `provider:model`; defaults from the environment ([Models](../guide/models)) |
| `variant` | `str \| None` | a [variant](../guide/variants); default `ragx-lexical` |
| `**settings` | | any field of [`Settings`](./configuration) |

Construction ingests the corpus into its text mirror. The passage store and the ranker (including
any embedding) are built on first use.

### Methods

#### `ask(question) -> Answer`

Answers a question and returns an [`Answer`](#answer). An empty question returns an empty,
ungrounded answer without calling a model.

#### `ask_stream(question) -> Iterator[Event]`

Yields [`Event`](#event)s as the run proceeds; the generator's return value (`StopIteration.value`)
is the `Answer`. See [Streaming the trace](../guide/streaming).

#### `search(query, top=8) -> list[str]`

Retrieval only, no LLM call: the names (`path:start-end`) of the ranker's top passages. With the
classic variant, the top files.

#### `reingest() -> Corpus`

Re-scans the corpus after files change (incrementally) and rebuilds the passages and ranker on next
use.

#### `stats() -> dict`

Corpus statistics (files, lines, vocabulary), the model, the variant and the retriever.

### Properties

| Property | Meaning |
|---|---|
| `settings` | the resolved `Settings` |
| `variant` | the resolved `Variant` (`name`, `retriever`, `route`, `backfill`, `think`, `summary`) |
| `retriever` | the ranker in use: `bm25`, `dense`, `fused` or `rerank` |
| `think` | whether thinking mode is on |
| `store` | the `PassageStore` (built on first access) |
| `ranker` | the ranker (built on first access) |
| `corpus` | the ingested `Corpus` |

## `Answer`

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | the answer |
| `citations` | `list[Citation]` | passages that contain the answer ([grounding](../guide/citations)) |
| `sources` | `list[str]` | the citations as `path:start-end` strings |
| `grounded` | `bool` | whether a cited passage contains the answer |
| `confidence` | `float \| None` | set by the classic agent's `both` route |
| `trace` | `list[Event]` | every event of the run |
| `usage` | `dict` | per-model calls, tokens, latency, cost; `llm_calls`; `variant`; `total_cost_usd` |

`str(answer)` is the text; `answer.to_dict(include_trace=False)` is JSON-ready.

## `Citation`

| Field | Meaning |
|---|---|
| `path` | source path, relative to the corpus |
| `line_start`, `line_end` | the line range |
| `quote` | the first non-empty line of the passage |

`str(citation)` is `path:start-end`.

## `Event`

| Field | Meaning |
|---|---|
| `kind` | `plan`, `rank`, `read`, `search`, `thought`, `hop`, `backfill`, `verify`, `answer`, ... |
| `text` | what happened |
| `step` | the forager turn, where there is one |
| `detail` | extra data, kind-specific |

## Variants

```python
from ragx import VARIANTS, get_variant

get_variant("ragx-lex")       # Variant(name='ragx-lexical', retriever='bm25', route='deep', backfill=0, ...)
list(VARIANTS)                # ['ragx', 'ragx-core', 'ragx-lexical', ...]
```

## Lower-level building blocks

For custom pipelines, the pieces are importable on their own:

```python
from ragx.passages import build_passages            # cut a Corpus into passages
from ragx.rankers import BM25Ranker, DenseRanker, FusedRanker, RerankRanker
from ragx.method import Method, MethodConfig         # the forage loop, slot chain and compose
from ragx.llm import LLM                             # the provider client
```

`Method(store, ranker, MethodConfig(...)).run(question, llm)` is a generator of events whose return
value is a `RunResult` with the answer, the composed passage ids, the reads and the resolved chain.

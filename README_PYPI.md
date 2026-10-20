<p align="center">
  <img src="https://raw.githubusercontent.com/ctrl-gaurav/ragX/main/assets/banner.png" alt="ragX: agentic retrieval at scale" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/pyragx/"><img src="https://img.shields.io/pypi/v/pyragx.svg?color=818cf8&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyragx/"><img src="https://img.shields.io/pypi/pyversions/pyragx.svg?color=38bdf8" alt="Python versions"></a>
  <a href="https://github.com/ctrl-gaurav/ragX/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-22c55e.svg" alt="License: Apache 2.0"></a>
  <a href="https://ctrl-gaurav.github.io/ragX/docs/"><img src="https://img.shields.io/badge/docs-ragX-c084fc.svg" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://ctrl-gaurav.github.io/ragX/"><b>Website</b></a> ·
  <a href="https://ctrl-gaurav.github.io/ragX/docs/"><b>Documentation</b></a> ·
  <a href="https://pypi.org/project/pyragx/"><b>PyPI</b></a> ·
  <b>Paper (coming soon)</b>
</p>

**ragX** is an open-source, training-free agentic retrieval framework. On millions of documents,
single-shot RAG misses the evidence a question does not name. ragX names it: an agent ranks and
reads passages, a chain of sub-questions resolves each hop, and one call answers with exact
`path:line` citations. It works with any LLM.

```python
from ragx import RagX

rag = RagX("./docs")
answer = rag.ask("Who founded the university the author of Harbor Light attended?")

print(answer.text)      # Thomas Wren
print(answer.sources)   # ['ashford_university.md:1-5', 'thomas_wren.md:1-5']
```

## News

- **2026-10-20** · ragX 0.0.1 is on PyPI: `pip install pyragx`. All seven variants of the paper's method,
  BM25 and dense, fused and reranked ranking, a command line and an OpenAI-compatible server.
- **2026-10-20** · The [website](https://ctrl-gaurav.github.io/ragX/) and the
  [documentation](https://ctrl-gaurav.github.io/ragX/docs/) are live.
- **Coming soon** · The paper, *ragX: Agentic Retrieval at Scale*, on arXiv.

## Why ragX

A question such as *"Who is the spouse of the actor who plays Jarvis in Iron Man?"* names Iron Man
and Jarvis, but not Paul Bettany, whose article holds the answer. On all of Wikipedia that passage
ranks 146,602th for the question itself, so single-shot retrieval cannot reach it. As a corpus
grows, single-shot retrieval loses such unnamed evidence about four times faster than named
evidence.

ragX reaches it through two separate routes, then composes once:

<p align="center">
    <img src="https://raw.githubusercontent.com/ctrl-gaurav/ragX/main/assets/architecture.png" alt="ragX: the forage loop and the slot chain feed one compose call" width="100%">
</p>

1. **Forage loop.** An agent `RANK`s the corpus and `READ`s passages for up to ten turns. It writes
   each query after reading, so a query can name what the question never did.
2. **Slot chain.** The question becomes up to four sub-questions with `#N` slots; each hop is
   retrieved with the earlier answers filled in.
3. **Compose.** One call answers over the read passages, the hop passages, the other surfaced ones
   and, in the full method, the question's own top 48.

## Results

Benchmarked on nested samples of Wikipedia from 240K to 21M passages, with 27 language models on
five QA datasets, against single-shot retrieval, ReAct, Self-Ask, IRCoT, DCI-Agent-Lite and
Search-R1:

| | |
|---|---|
| **+8.1 F1** | over the strongest single-shot baseline on all of Wikipedia (21M passages, eight models) |
| **+5.4 → +9.0 F1** | the lead over single-shot retrieval grows as the corpus grows from 240K to 21M passages |
| **22× fewer** | prompt tokens than DCI-Agent-Lite |
| **17 of 21** | open-weight models on which ragX leads single-shot retrieval (a ragX variant leads on 19) |

With BM25 alone, ragX beats every dense single-shot baseline at 21M passages. The paper is coming
soon.

## Install

```bash
pip install pyragx                 # ragx-lexical: BM25, no embedding index, no GPU
pip install "pyragx[dense]"        # every variant (dense, fused and reranked ranking)
```

The PyPI name is `pyragx`; you `import ragx` and run the `ragx` command.

Set a key for any provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`GROQ_API_KEY`, ...), point `vllm:` at your own server, or run a local Hugging Face model with
`pyragx[local]`.

## Variants

| Variant | Ranker | What it is |
|---|---|---|
| `ragx` | dense | the full method: forage loop, slot chain, 48-passage backfill |
| `ragx-rerank` | dense top 100 → cross-encoder | the most accurate in the benchmark |
| `ragx-lexical` *(default)* | BM25 | no embedding index, no GPU |
| `ragx-fused` | BM25 + dense (RRF) | exact terms and paraphrase |
| `ragx-core` | dense | forage loop and slot chain, no backfill |
| `ragx-fast` | dense | forage loop only, the cheapest |
| `ragx-think` | dense | `ragx` with thinking on |

```python
rag = RagX("./docs", variant="ragx-rerank", model="openai:gpt-4o-mini")
```

## Command line and server

```bash
ragx ask "Whom did the founder of Ashford University marry?" -c ./docs -v ragx-fused
ragx search "Ashford University founder" -c ./docs        # retrieval only, no LLM
ragx serve ./docs --port 8000                              # OpenAI-compatible /v1/chat/completions
```

## Documentation

The full documentation is at **[ctrl-gaurav.github.io/ragX/docs](https://ctrl-gaurav.github.io/ragX/docs/)**:
[how ragX works](https://ctrl-gaurav.github.io/ragX/docs/guide/how-it-works),
[variants](https://ctrl-gaurav.github.io/ragX/docs/guide/variants),
[models and providers](https://ctrl-gaurav.github.io/ragX/docs/guide/models),
[the Python API](https://ctrl-gaurav.github.io/ragX/docs/reference/python) and
[every configuration option](https://ctrl-gaurav.github.io/ragX/docs/reference/configuration).

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/ctrl-gaurav/ragX/blob/main/CONTRIBUTING.md).

## Citation

The paper, *ragX: Agentic Retrieval at Scale*, is coming soon. Until then, please cite the software:

```bibtex
@software{srivastava2026ragx,
  author  = {Srivastava, Gaurav},
  title   = {ragX: Agentic Retrieval at Scale},
  year    = {2026},
  url     = {https://github.com/ctrl-gaurav/ragX},
  version = {0.0.1}
}
```

## License

[Apache 2.0](https://github.com/ctrl-gaurav/ragX/blob/main/LICENSE)

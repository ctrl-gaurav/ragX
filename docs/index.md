---
title: What is ragX?
---

# What is ragX?

**ragX** is an open-source, training-free agentic retrieval framework. Point it at a folder of
documents and ask a question: an agent ranks and reads passages, a chain of sub-questions resolves
each hop of a multi-hop question, and one call composes the answer with exact `path:line`
citations.

<div class="hero-actions">
  <a class="hero-btn primary" href="./guide/quickstart">Quickstart →</a>
  <a class="hero-btn" href="./guide/how-it-works">How it works</a>
  <a class="hero-btn" href="https://github.com/ctrl-gaurav/ragX">GitHub</a>
  <a class="hero-btn" href="https://pypi.org/project/pyragx/">PyPI</a>
</div>

```python
from ragx import RagX

rag = RagX("./docs")
answer = rag.ask("Who founded the university the author of Harbor Light attended?")
print(answer.text)      # Thomas Wren
print(answer.sources)   # ['ashford_university.md:1-5', 'thomas_wren.md:1-5']
```

## Why ragX

Single-shot retrieval-augmented generation (RAG) sends the question itself as the search query.
That retrieves what the question names. On a large corpus it misses the evidence the question
does **not** name: in the example above, the question names *Harbor Light* but never the author or
her university, and those are the documents that hold the answer.

ragX names that missing evidence through two separate routes:

- **The forage loop.** An agent ranks the corpus and reads passages for up to ten turns. Because
  it writes each query after reading, a query can name an entity the question never mentioned.
- **The slot chain.** The question is broken into sub-questions with `#1`, `#2` slots, and each hop
  is retrieved with the earlier answers filled in.

One compose call then answers over everything both routes found. See
[How ragX works](./guide/how-it-works) for the full pipeline.

## Results in brief

On nested samples of Wikipedia from 240K to 21M passages, benchmarked with 27 language models on
five QA datasets:

| | |
|---|---|
| **+8.1 F1** | over the strongest single-shot baseline on all of Wikipedia (21M passages, eight models) |
| **+5.4 → +9.0 F1** | the lead over single-shot retrieval grows as the corpus grows from 240K to 21M passages |
| **22× fewer** | prompt tokens than DCI-Agent-Lite, an agent that searches the raw corpus with shell tools |
| **17 of 21** | open-weight models on which ragX leads single-shot retrieval (a ragX variant leads on 19) |

With a plain BM25 index, ragX also beats every dense single-shot baseline at 21M passages. The
paper with the full study is coming soon.

## What you get

<div class="card-grid">
  <div class="c"><b>Seven variants</b><span>From BM25-only <code>ragx-lexical</code> to the full <code>ragx</code> with a dense index and backfill.</span></div>
  <div class="c"><b>Any model</b><span>OpenAI, Anthropic, Google, Groq, Together, Cerebras, Fireworks, any vLLM server, or a local Hugging Face model.</span></div>
  <div class="c"><b>Exact citations</b><span>Every answer cites the passages that contain it, as <code>path:start-end</code>.</span></div>
  <div class="c"><b>Fenced by default</b><span>Tool output is wrapped as untrusted data, so planted instructions are not followed.</span></div>
  <div class="c"><b>Three interfaces</b><span>A Python API, a command line with a live trace, and an OpenAI-compatible server.</span></div>
  <div class="c"><b>Incremental ingestion</b><span>Text, Markdown, HTML, PDF, Word, CSV, JSON and source files; only changed files are re-processed.</span></div>
</div>

## Next steps

- [Install ragX](./guide/installation) and its optional extras.
- Follow the [Quickstart](./guide/quickstart) to answer your first question.
- [Choose a variant](./guide/variants) for your index and budget.

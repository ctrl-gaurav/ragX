# Quickstart

This page answers a multi-hop question over a small folder of documents, first from Python and then
from the command line.

## 1. Install and set a key

```bash
pip install pyragx
export OPENAI_API_KEY=...
```

Any [supported provider](./models) works; the examples use `openai:gpt-4o-mini`.

## 2. Get a corpus

The repository ships a small example corpus of invented articles, so a model cannot answer from
memory:

```bash
git clone https://github.com/ctrl-gaurav/ragX
cd ragX/examples/multihop_corpus
ls
# ashford_university.md  daniel_ames.md  harbor_light.md  mira_okafor.md
# northfield_college.md  penrose_prize.md  thomas_wren.md
```

Any folder of your own documents works the same way.

## 3. Ask from Python

```python
from ragx import RagX

rag = RagX("examples/multihop_corpus", model="openai:gpt-4o-mini")

answer = rag.ask("In which city is the university attended by the author of Harbor Light?")
print(answer.text)        # Bristol
print(answer.sources)     # ['ashford_university.md:1-5']
print(answer.grounded)    # True: a cited passage contains the answer
```

The question names *Harbor Light* but not its author (Mira Okafor) or her university (Ashford
University). ragX reads the novel's article, ranks again with the author's name, and resolves the
chain hop by hop.

## 4. Watch the trace

`ask_stream` yields each step as it happens and returns the answer at the end:

```python
for event in rag.ask_stream("Who founded the university where the author of Harbor Light studied?"):
    print(event)
# [plan] Foraging for: Who founded the university where ...
# [rank] Harbor Light author
# [read] harbor_light.md:1-5
# [rank] Mira Okafor university
# [read] mira_okafor.md:1-6
# [hop] Who wrote Harbor Light? → Mira Okafor
# ...
```

See [Streaming the trace](./streaming) for how to collect the final `Answer`.

## 5. Use the command line

```bash
ragx ask "Whom did the founder of Ashford University marry?" -c examples/multihop_corpus
ragx search "Ashford University founder" -c examples/multihop_corpus    # retrieval only
```

## 6. Choose a variant

The default variant, `ragx-lexical`, ranks with BM25. For the full method with a dense index and
the backfill, install the `dense` extra:

```bash
pip install "pyragx[dense]"
```

```python
rag = RagX("examples/multihop_corpus", variant="ragx", model="openai:gpt-4o-mini")
```

The first run embeds the corpus with `Qwen/Qwen3-Embedding-0.6B` and caches the vectors, so later
runs start immediately. See [Variants](./variants) for the trade-offs.

# Corpora and passages

## Supported files

ragX ingests a single file or a folder, recursively:

| Kind | Extensions |
|---|---|
| Text | `.txt` `.md` `.markdown` `.rst` `.log` `.tex` |
| Web | `.html` `.htm` `.xhtml` (converted to Markdown) |
| Documents | `.pdf` `.docx` |
| Data | `.csv` `.tsv` `.json` `.jsonl` |
| Source code | common programming-language extensions (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, ...) |

`.git/`, `node_modules/` and `.ragx/` are skipped. Use `include=` and `exclude=` glob lists to choose
files:

```python
RagX("./repo", include=["docs/**", "*.md"], exclude=[".git/**", "build/**"])
```

## The text mirror

Every file is converted once into a line-numbered text mirror under `<corpus>/.ragx/` (or
`cache_dir=`). Ingestion is incremental: a file is converted again only when its content changes,
and mirrors of deleted files are removed. After you change files in a running program, call:

```python
rag.reingest()
```

## Passages

The ragX variants rank, read and cite **passages**: each file is cut along line boundaries into
pieces of about `passage_words` words (120 by default), and a Markdown heading always starts a new
passage. A passage is named by its source path and line range:

```text
ashford_university.md:1-5
handbook/onboarding.md:42-57
```

Its title is the file's title (its first heading, or its file name), followed by the section
heading when there is one. The agent sees these names in `RANK` results, reads by them, and the
answer cites them.

Smaller passages make ranking more precise; larger ones give the agent more context per `READ`. The
paper's corpus used passages of about 100 words.

```python
RagX("./docs", passage_words=200)
```

## Large corpora

BM25 and dense search are exact and run in memory. For dense variants, embedding is the one-off
cost: a GPU embeds hundreds of passages per second, a CPU far fewer. The embeddings are cached, so
the cost is paid once. Prepare a corpus ahead of time with:

```bash
ragx ingest ./docs --variant ragx
```

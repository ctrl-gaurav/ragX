# Command line

Installing ragX adds the `ragx` command.

## `ragx ask`

Answer a question over a corpus, with a live trace of the agent:

```bash
ragx ask "Who founded Ashford University?" --corpus ./docs
```

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--corpus` | `.` | file or folder to search |
| `-m`, `--model` | from the environment | `provider:model` |
| `-v`, `--variant` | `ragx-lexical` | any [variant](./variants) |
| `--steps` | 10 | forager turns |
| `--think` / `--no-think` | the variant's | thinking mode |
| `--live` / `--no-live` | `--live` | show the live trace |
| `--json` | off | print the answer as JSON (implies `--no-live`) |

`--json` prints the text, citations, `grounded`, confidence and usage, which is convenient for
scripts:

```bash
ragx ask "Who founded Ashford University?" -c ./docs --json | jq .text
```

## `ragx search`

Retrieval only, with no LLM call: the top passages for a query.

```bash
ragx search "Ashford University founder" -c ./docs -v ragx-fused -k 5
```

## `ragx ingest`

Build the text mirror and, for dense variants, the embeddings ahead of time:

```bash
ragx ingest ./docs --variant ragx
```

## `ragx serve`

Start an [OpenAI-compatible server](./server):

```bash
ragx serve ./docs --variant ragx-lexical --port 8000
```

## `ragx variants`

List every variant with its ranker, stages and backfill.

## `ragx version`

Print the installed version (also `ragx --version`).

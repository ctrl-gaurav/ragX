# The classic agent

`variant="classic"` runs ragX's original agent. Instead of ranking passages, it works on whole files
with a toolbelt of shell-style tools, and it is useful for code bases and document folders where
exact search and section navigation matter more than multi-hop chains.

```python
rag = RagX("./repo", variant="classic")
```

## Tools

| Action | What it does |
|---|---|
| `SEARCH: <keywords>` | case-insensitive search with ripgrep (or a Python fallback); returns `path:line` snippets |
| `RANK: <keywords>` | ranks whole files with BM25 over document frequencies |
| `FUZZY: <phrase>` | spelling-tolerant search with ugrep, when installed |
| `LIST [glob]` | lists files |
| `OUTLINE` | the corpus's heading outline |
| `TOC <path>` | one file's headings |
| `READ <path>:<start>-<end>` | exact lines |
| `PEEK <path>:<line>` | a small window around a line |
| `SEMANTIC: <phrase>` | embedding search, only with `semantic_fallback=True` and `pyragx[semantic]` |
| `ANSWER: <answer>` | finish |

## Routes

`route=` chooses how the classic agent runs:

| Route | Behaviour |
|---|---|
| `agentic` (default) | the tool loop, then a compose over what it read and surfaced |
| `adaptive` | one ranked read and answer first; escalates to the loop when that answer is not grounded |
| `deep` | the loop, then a file-level slot chain |
| `both` | `deep`, plus a second compose with the chain's notes; `answer.confidence` is 0.9 when the two agree and 0.5 when they do not |

The classic agent's settings (`max_steps`, `max_result_lines`, `max_read_chars`, `prerank`,
`deep_hops`, `surf_cap`, `sandbox_observations` and others) are listed in
[Configuration](../reference/configuration#classic-agent).

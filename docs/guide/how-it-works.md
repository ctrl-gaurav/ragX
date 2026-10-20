# How ragX works

ragX answers in three stages: a **forage loop**, a **slot chain** and a single **compose** call. The
first two are separate routes to the same goal, a query that names evidence the question does not
name. The compose call answers over everything both routes found.

## The problem it solves

Single-shot RAG retrieves with the question as the query. A question such as

> Who is the spouse of the actor who plays Jarvis in Iron Man?

names *Iron Man* and *Jarvis*, but not *Paul Bettany*, whose article holds the answer. On all of
Wikipedia that answer passage ranks 146,602th for the question itself, so no query copied from the
question reaches it. As a corpus grows, single-shot retrieval loses such unnamed evidence about
four times faster than named evidence, and neither a larger embedder nor a deep reranker changes
that. A query that names the missing entity does.

## 1. The forage loop

An agent explores the corpus with two tools, one action per turn:

| Action | What it does |
|---|---|
| `RANK: <keywords>` | ranks every passage with the variant's ranker and lists the top 8 by name and title |
| `READ: <passage>` | prints one passage in full, with line numbers |
| `ANSWER: <short answer>` | ends the loop with a draft answer |

The loop runs for at most **ten turns**. Because the agent writes each query after reading, a
later `RANK` can name an entity it found in an earlier `READ`: in the example, reading the *Iron
Man* article leads to `RANK: Paul Bettany spouse`.

Every observation is [fenced](./security) as untrusted data before the agent sees it.

## 2. The slot chain

Separately, the question is decomposed into at most **four** ordered sub-questions, with `#N`
slots for earlier answers:

```text
Who plays Jarvis in Iron Man?
Who is the spouse of #1?
```

Each hop is then resolved in turn: the sub-question, with the earlier answers filled into its slots,
retrieves **four** passages; one short call answers it from their most relevant sentences; and that
answer fills the next slot. The two top passages of each hop are kept for the compose step, because
a hop's passage often shares no words with the original question.

The `ragx-fast` variant skips this stage.

## 3. Compose

One call answers the question over, in order:

1. up to **8** passages the agent read,
2. up to **12** extra passages: first those that resolved a hop, then the other passages the agent
   surfaced, ranked against the question,
3. in `ragx` and `ragx-think` only, the **backfill**: the question's own top **48** passages that are
   not already included.

Passages are composed whole. If the evidence would exceed the model's context window, ragX trims
from the end, removing backfill passages first. The compose prompt asks for the shortest exact
answer and adds a one-line hint about the expected answer type (a date, a place, a person, yes or
no, one of the named options).

The answer is then [grounded](./citations): it is cited to the composed passages that contain it.

## Calls per question

A run makes one call per forager turn, one decomposition call, one call per hop and one compose
call, so typically 8 to 15 calls. `ragx-fast` makes only the forager calls and the compose call.
See [Cost and speed](./performance).

## The frozen settings

The defaults are the settings of the paper, and every variant shares them:

| Setting | Default | Meaning |
|---|---|---|
| `steps` | 10 | forager turns |
| `rank_k` | 8 | passages listed per `RANK` |
| `hop_k` | 4 | passages retrieved per hop |
| `max_hops` | 4 | sub-questions at most |
| `read_cap` | 8 | read passages composed |
| `extra_cap` | 12 | extra passages composed |
| `backfill` | 48 or 0 | the question's own top passages appended (48 in `ragx` and `ragx-think`) |
| `tools` | `RANK`, `READ` | forager tools (`SEARCH` can be added) |
| `fence` | on | tool output fenced as untrusted data |

All of them can be changed; see [Configuration](../reference/configuration).

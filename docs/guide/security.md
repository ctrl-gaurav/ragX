# Security

An agent that reads documents also reads whatever instructions someone planted in them. ragX is
built to treat retrieved text as data.

## The fence

With `fence=True` (the default):

- The forager's system prompt states that files and excerpts are untrusted data and that only the
  question defines the task.
- Every tool result is wrapped in `<untrusted_data>` tags before the agent sees it.
- Lines inside retrieved text that imitate the action grammar (`ANSWER:`, `READ:`, `RANK:`,
  `SEARCH:`) are neutralised, so a document cannot inject an action.
- The compose call gets the same treatment: its evidence is fenced and its question carries the
  same note.

In the paper's injection tests (planted instructions inside retrieved passages), fencing kept ragX
within 1.5 F1 of its clean accuracy on average.

Turn it off only for trusted corpora where you need every token of context:

```python
RagX("./docs", fence=False)
```

## Keys

Provider keys are read from the environment or a `.env` file by name and are never written to logs,
traces or answers.

## The server

`ragx serve` has no authentication. Bind it to `127.0.0.1` (the default) or put it behind a gateway
that authenticates requests.

## Reporting a vulnerability

Please report security issues privately as described in
[SECURITY.md](https://github.com/ctrl-gaurav/ragX/blob/main/SECURITY.md), not in a public issue.

# Streaming the trace

`ask_stream` is a generator: it yields an `Event` for each step as it happens, and its return value
is the final `Answer`.

```python
gen = rag.ask_stream("Who founded the university where the author of Harbor Light studied?")
while True:
    try:
        event = next(gen)
    except StopIteration as stop:
        answer = stop.value
        break
    print(f"{event.kind:9} {event.text}")

print(answer.text, answer.sources)
```

`rag.ask(question)` does exactly this and returns the answer; `answer.trace` keeps every event.

## Event kinds

| Kind | When | `text` |
|---|---|---|
| `plan` | the run starts | the question |
| `rank` | the agent ran `RANK` | the query |
| `read` | the agent ran `READ` | the passage name |
| `search` | the agent ran `SEARCH` (only when enabled) | the query |
| `thought` | the agent answered early, or wrote an invalid action | a short note |
| `hop` | a sub-question was resolved | `sub-question → answer` |
| `backfill` | the backfill was added | how many passages |
| `verify` | the compose call starts | how many passages are composed |
| `answer` | the compose call returned | the answer |

The [classic agent](./classic) adds its own kinds (`fuzzy`, `list`, `toc`, `peek`, `semantic`).

Each `Event` has `kind`, `text`, `step` (the forager turn, where there is one) and `detail`, and
`event.to_dict()` gives a JSON-ready view.

## In a web app

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
rag = RagX("./docs")

@app.get("/ask")
def ask(q: str):
    def events():
        gen = rag.ask_stream(q)
        while True:
            try:
                yield json.dumps(next(gen).to_dict()) + "\n"
            except StopIteration as stop:
                yield json.dumps({"kind": "final", **stop.value.to_dict()}) + "\n"
                return
    return StreamingResponse(events(), media_type="application/x-ndjson")
```

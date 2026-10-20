# OpenAI-compatible server

`ragx serve` exposes a corpus as an OpenAI-style chat endpoint, so any tool that speaks the OpenAI
API can use ragX as its model.

```bash
pip install "pyragx[server]"
ragx serve ./docs --variant ragx-lexical --model openai:gpt-4o-mini --port 8000
```

## Endpoints

### `POST /v1/chat/completions`

The last user message is the question. The reply is a standard chat completion whose message holds
the answer followed by its sources, plus a `ragx` field with the full answer object:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Who founded Ashford University?"}]}'
```

```json
{
  "object": "chat.completion",
  "model": "openai:gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Thomas Wren\n\nSources:\n- ashford_university.md:1-5"
    },
    "finish_reason": "stop"
  }],
  "ragx": {
    "text": "Thomas Wren",
    "citations": [{"path": "ashford_university.md", "line_start": 1, "line_end": 5, "quote": "# Ashford University", "ref": "ashford_university.md:1-5"}],
    "sources": ["ashford_university.md:1-5"],
    "grounded": true,
    "confidence": null,
    "usage": {"...": "..."}
  }
}
```

### `GET /health`

Returns `{"status": "ok"}` with corpus statistics, the model and the variant.

## From the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="ragx",
    messages=[{"role": "user", "content": "Who founded Ashford University?"}],
)
print(reply.choices[0].message.content)
```

## In your own app

`create_app` returns the FastAPI application, so you can mount it or add middleware:

```python
from ragx.server import create_app

app = create_app("./docs", model="openai:gpt-4o-mini", variant="ragx-fused")
```

Each request runs in a worker thread. The server has no authentication of its own; put it behind
your usual gateway before exposing it.

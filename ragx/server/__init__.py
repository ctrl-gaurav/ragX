"""Optional OpenAI-compatible server (``pyragx[server]``).

Exposes ``POST /v1/chat/completions``: the last user message is treated as the question;
ragX answers it over the configured corpus and returns an OpenAI-style response whose
message includes the grounded answer plus citations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def create_app(corpus: str | Path, model: str | None = None, **kwargs: Any):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from ..__about__ import __version__
    from ..core import RagX

    rag = RagX(corpus, model=model, **kwargs)
    app = FastAPI(title="ragX", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **rag.stats()}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict) -> Any:  # sync: FastAPI runs it in a worker thread
        messages = body.get("messages", [])
        question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if not question:
            return JSONResponse({"error": "no user message"}, status_code=400)
        ans = rag.ask(question)
        content = ans.text
        if ans.citations:
            content += "\n\nSources:\n" + "\n".join(f"- {c}" for c in ans.citations)
        return {
            "id": f"ragx-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": rag.settings.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "ragx": ans.to_dict(),
        }

    return app


def serve(
    corpus: str | Path,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs: Any,
) -> None:
    import uvicorn

    uvicorn.run(create_app(corpus, model=model, **kwargs), host=host, port=port)


__all__ = ["create_app", "serve"]

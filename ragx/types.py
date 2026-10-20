"""Core data types for ragX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Citation:
    """An exact, verifiable reference into the corpus."""

    path: str
    line_start: int
    line_end: int
    quote: str = ""

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line_start}"
        if self.line_end != self.line_start:
            loc += f"-{self.line_end}"
        return loc

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "quote": self.quote,
            "ref": str(self),
        }


@dataclass
class Answer:
    """The result of ``RagX.ask``."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float | None = None
    trace: list[Event] = field(default_factory=list)
    grounded: bool = True
    usage: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    @property
    def sources(self) -> list[str]:
        return [str(c) for c in self.citations]

    def to_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        """A JSON-serializable view of the answer (for the CLI/server/logging)."""
        d: dict[str, Any] = {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "usage": self.usage,
        }
        if include_trace:
            d["trace"] = [e.to_dict() for e in self.trace]
        return d


EventKind = Literal[
    "plan",
    "search",
    "fuzzy",
    "rank",
    "list",
    "toc",
    "read",
    "peek",
    "semantic",
    "hop",
    "backfill",
    "observation",
    "thought",
    "verify",
    "answer",
    "error",
]


@dataclass
class Event:
    """A single step in the agent's foraging loop (streamed to the UI)."""

    kind: EventKind
    text: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    step: int = 0

    def __str__(self) -> str:
        return f"[{self.kind}] {self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "detail": self.detail, "step": self.step}

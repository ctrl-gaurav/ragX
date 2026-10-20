"""Passages: the units the ragX method ranks, reads and composes over.

Every ingested file is cut into short passages along line boundaries (about ``passage_words``
words each, never splitting a line, and starting a new passage at a Markdown heading). A
passage keeps its source path and line range, so ranking, reading and citation all point at an
exact ``path:start-end`` span. This mirrors the paper's setting, where the corpus is a set of
short titled passages.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .ingest import Corpus

_HEADING = re.compile(r"^#{1,6}\s+(.*)")


@dataclass(frozen=True)
class Passage:
    pid: int
    path: str  # source path, as shown in citations
    start: int  # first line (1-based, inclusive)
    end: int  # last line (inclusive)
    title: str  # file title, plus the nearest heading when there is one
    text: str

    @property
    def name(self) -> str:
        """The handle the agent reads by and the answer cites: ``path:start-end``."""
        return f"{self.path}:{self.start}-{self.end}"

    def numbered(self) -> str:
        return "\n".join(f"{self.start + i}: {ln}" for i, ln in enumerate(self.text.splitlines()))


@dataclass
class PassageStore:
    passages: list[Passage] = field(default_factory=list)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        self._by_name = {p.name: p for p in self.passages}
        self._by_path: dict[str, list[Passage]] = {}
        for p in self.passages:
            self._by_path.setdefault(p.path, []).append(p)

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, pid: int) -> Passage:
        return self.passages[pid]

    def resolve(self, ref: str) -> Passage | None:
        """Find the passage a READ argument names: an exact ``path:start-end``, a ``path:line``
        inside a passage, or a bare path (its first passage). Tolerates quotes and a trailing
        title after the name."""
        ref = ref.strip().strip("`'\"").split()[0] if ref.strip() else ""
        if not ref:
            return None
        if ref in self._by_name:
            return self._by_name[ref]
        m = re.match(r"^(.*?):(\d+)(?:-(\d+))?$", ref)
        path, line = (m.group(1), int(m.group(2))) if m else (ref, None)
        cands = self._by_path.get(path)
        if cands is None:  # tolerate a basename or a path suffix
            for p, ps in self._by_path.items():
                if p.endswith(path) or p.split("/")[-1] == path:
                    cands = ps
                    break
        if not cands:
            return None
        if line is None:
            return cands[0]
        for p in cands:
            if p.start <= line <= p.end:
                return p
        return min(cands, key=lambda p: abs(p.start - line))


def build_passages(corpus: Corpus, words: int = 120) -> PassageStore:
    """Cut every file of ``corpus`` into passages of about ``words`` words."""
    out: list[Passage] = []
    h = hashlib.sha1(f"w={words}".encode())
    for f in corpus.files:
        h.update(f"{f.source_path}\0{f.sha1}\0".encode())
        try:
            lines = (
                (corpus.mirror_dir / f.mirror_path)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            continue
        for start, end, heading in _segments(lines, words):
            body = "\n".join(lines[start - 1 : end])
            if not body.strip():
                continue
            title = f.title if not heading or heading == f.title else f"{f.title} — {heading}"
            out.append(Passage(len(out), f.source_path, start, end, title, body))
    return PassageStore(out, h.hexdigest()[:16])


def _segments(lines: list[str], words: int) -> list[tuple[int, int, str]]:
    """(start, end, heading) line spans of about ``words`` words; a heading opens a new span."""
    spans: list[tuple[int, int, str]] = []
    heading = ""
    start, n_words = 0, 0  # start == 0: no open span
    for i, ln in enumerate(lines, 1):
        m = _HEADING.match(ln)
        if m and start and n_words > 0:
            spans.append((start, i - 1, heading))
            start, n_words = 0, 0
        if m:
            heading = m.group(1).strip()
        if not start:
            start = i
        n_words += len(ln.split())
        if n_words >= words:
            spans.append((start, i, heading))
            start, n_words = 0, 0
    if start:
        spans.append((start, len(lines), heading))
    return spans


__all__ = ["Passage", "PassageStore", "build_passages"]

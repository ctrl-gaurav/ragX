"""Instant, incremental ingestion → a line-numbered text mirror + a tiny index.

No embeddings, no GPU. Adding/changing a file rebuilds only that file. The mirror is
what the tools grep over; the index powers ranking, file listing, and section nav.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from .loaders import is_supported, load_text

_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass
class FileEntry:
    mirror_path: str  # relative to mirror_dir, e.g. "report.md.txt"
    source_path: str  # original path shown in citations
    title: str
    n_lines: int
    sha1: str
    loader: str


@dataclass
class Corpus:
    root: Path
    mirror_dir: Path
    cache_dir: Path
    files: list[FileEntry] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0
    _by_mirror: dict[str, FileEntry] = field(default_factory=dict)

    def source_of(self, mirror_rel: str) -> str:
        e = self._by_mirror.get(mirror_rel)
        return e.source_path if e else mirror_rel

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "lines": sum(f.n_lines for f in self.files),
            "vocab": len(self.df),
        }


def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def _iter_files(root: Path, settings) -> list[Path]:
    if root.is_file():
        return [root] if is_supported(root) else []
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not is_supported(p):
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch(rel, pat) for pat in settings.exclude):
            continue
        if settings.include and not any(fnmatch(rel, pat) for pat in settings.include):
            continue
        out.append(p)
    return out


def _headings(text: str) -> list[tuple[int, int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def ingest(corpus_path: str | Path, settings) -> Corpus:
    root = Path(corpus_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"corpus path not found: {root}")
    cache_dir = settings.resolve_cache_dir(root)
    mirror_dir = cache_dir / "mirror"
    index_dir = cache_dir / "index"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    prev = _load_manifest(index_dir)
    base = root.parent if root.is_file() else root

    files: list[FileEntry] = []
    df: dict[str, int] = {}
    headings_out: list[dict] = []

    for src in _iter_files(root, settings):
        rel = src.relative_to(base).as_posix()
        mirror_rel = rel + ".txt"
        mfile = mirror_dir / mirror_rel
        raw = src.read_bytes()
        sha = _sha1(raw)
        old = prev.get(mirror_rel)
        if old and old.get("sha1") == sha and mfile.exists():
            text = mfile.read_text(encoding="utf-8", errors="replace")
            loader = old.get("loader", "text")
        else:
            text, loader = load_text(src)
            mfile.parent.mkdir(parents=True, exist_ok=True)
            mfile.write_text(text, encoding="utf-8")
        lines = text.splitlines()
        title = _title(src, lines)
        files.append(FileEntry(mirror_rel, rel, title, len(lines), sha, loader))
        # document frequencies (one count per file)
        for tok in {w.lower() for w in _WORD.findall(text)}:
            df[tok] = df.get(tok, 0) + 1
        for ln, lvl, htext in _headings(text):
            headings_out.append(
                {"path": rel, "mirror": mirror_rel, "line": ln, "level": lvl, "text": htext}
            )

    # prune mirror files for sources that disappeared
    keep = {f.mirror_path for f in files}
    for stale in mirror_dir.rglob("*.txt"):
        if stale.relative_to(mirror_dir).as_posix() not in keep:
            stale.unlink(missing_ok=True)

    corpus = Corpus(
        root=root,
        mirror_dir=mirror_dir,
        cache_dir=cache_dir,
        files=files,
        df=df,
        n_docs=len(files),
        _by_mirror={f.mirror_path: f for f in files},
    )
    _write_index(index_dir, files, df, headings_out, root)
    return corpus


def _title(src: Path, lines: list[str]) -> str:
    for line in lines[:5]:
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            return m.group(1).strip()
    return src.stem


def _load_manifest(index_dir: Path) -> dict[str, dict]:
    f = index_dir / "files.jsonl"
    out: dict[str, dict] = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["mirror_path"]] = d
    return out


def _write_index(index_dir: Path, files, df, headings, root) -> None:
    with (index_dir / "files.jsonl").open("w", encoding="utf-8") as f:
        for e in files:
            f.write(json.dumps(e.__dict__) + "\n")
    with (index_dir / "headings.jsonl").open("w", encoding="utf-8") as f:
        for h in headings:
            f.write(json.dumps(h) + "\n")
    (index_dir / "df.json").write_text(
        json.dumps({"n_docs": len(files), "df": df}), encoding="utf-8"
    )
    (index_dir / "manifest.json").write_text(
        json.dumps({"root": str(root), "n_files": len(files)}), encoding="utf-8"
    )


__all__ = ["ingest", "Corpus", "FileEntry"]

"""Document loaders: convert any supported file to plain, grep-able text.

Each loader returns ``(text, loader_name)``. Text keeps one logical unit per line so
that ``file:line`` citations are meaningful. No GPU, no heavy deps beyond pypdf/bs4.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".log", ".tex"}
CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rb",
    ".php",
    ".sh",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}
HTML_EXTS = {".html", ".htm", ".xhtml"}
SUPPORTED = TEXT_EXTS | CODE_EXTS | HTML_EXTS | {".pdf", ".docx", ".csv", ".tsv", ".json", ".jsonl"}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


def load_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXTS or ext in CODE_EXTS:
            return path.read_text(encoding="utf-8", errors="replace"), "text"
        if ext in HTML_EXTS:
            return _load_html(path), "html"
        if ext == ".pdf":
            return _load_pdf(path), "pdf"
        if ext == ".docx":
            return _load_docx(path), "docx"
        if ext in {".csv", ".tsv"}:
            return _load_csv(path, "\t" if ext == ".tsv" else ","), "csv"
        if ext in {".json", ".jsonl"}:
            return path.read_text(encoding="utf-8", errors="replace"), "json"
    except Exception as e:  # noqa: BLE001
        return f"[ragx: failed to load {path.name}: {e}]", "error"
    return path.read_text(encoding="utf-8", errors="replace"), "text"


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # one block element per line
    lines = [t.strip() for t in soup.get_text("\n").splitlines() if t.strip()]
    return "\n".join(lines)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[str] = []
    for i, page in enumerate(reader.pages, 1):
        out.append(f"[page {i}]")
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if line:
                out.append(line)
    return "\n".join(out)


def _load_docx(path: Path) -> str:
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(f"{ns}p"):
        text = "".join(t.text or "" for t in p.iter(f"{ns}t"))
        if text.strip():
            paras.append(text.strip())
    return "\n".join(paras)


def _load_csv(path: Path, delim: str) -> str:
    out = io.StringIO()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    for row in rows[1:]:
        pairs = [f"{h}={v}" for h, v in zip(header, row, strict=False) if v]
        out.write(" | ".join(pairs) + "\n")
    return out.getvalue()

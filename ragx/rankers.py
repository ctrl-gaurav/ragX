"""Rankers over passages: BM25, dense, reciprocal-rank fusion and a cross-encoder reranker.

Every ranker answers ``search(query, k) -> list[int]`` (passage ids, best first). BM25 needs
nothing beyond the standard library; the dense and rerank rankers need the ``dense`` extra
(``pip install "pyragx[dense]"``) and cache their embeddings next to the text mirror, so a corpus
is embedded once and only changed passages are embedded again.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from pathlib import Path
from typing import Protocol

from .passages import PassageStore

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Ranker(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[int]: ...


class MissingExtraError(ImportError):
    """A ranker needs an optional dependency that is not installed."""


def _need_dense():
    try:
        import numpy as np
        import sentence_transformers
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise MissingExtraError(
            "the dense, fused and rerank rankers need the 'dense' extra: "
            'pip install "pyragx[dense]"'
        ) from e
    return np, sentence_transformers


# --------------------------------------------------------------------------------------- BM25
class BM25Ranker:
    """Okapi BM25 over passages (title and text), from an in-memory inverted index."""

    name = "bm25"

    def __init__(self, store: PassageStore, k1: float = 0.9, b: float = 0.4):
        self.store, self.k1, self.b = store, k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        for p in store.passages:
            toks = _tokens(f"{p.title}\n{p.text}")
            self.lengths.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.postings.setdefault(t, []).append((p.pid, c))
        n = max(1, len(self.lengths))
        self.avgdl = max(1.0, sum(self.lengths) / n)
        self.idf = {
            t: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5))
            for t, ps in self.postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        acc: dict[int, float] = {}
        for t in set(_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for pid, tf in self.postings[t]:
                dl = self.lengths[pid]
                acc[pid] = acc.get(pid, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return acc

    def search(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        return [pid for pid, _ in heapq.nlargest(k, sc.items(), key=lambda x: (x[1], -x[0]))]


# -------------------------------------------------------------------------------------- dense
class DenseRanker:
    """Exact inner-product search over normalised passage embeddings.

    The default embedder is the paper's, ``Qwen/Qwen3-Embedding-0.6B``; any
    sentence-transformers model works. Queries use the model's ``query`` prompt when it has one.
    """

    name = "dense"

    def __init__(
        self,
        store: PassageStore,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_dir: Path | None = None,
        device: str | None = None,
        batch_size: int = 32,
        encoder=None,
    ):
        self.store, self.model_name = store, model
        self._np = None
        if encoder is None:
            np, st = _need_dense()
            self._np = np
            encoder = st.SentenceTransformer(model, device=device)
        else:
            import numpy as np

            self._np = np
        self.encoder = encoder
        self.batch_size = batch_size
        self.matrix = self._embed_all(cache_dir)

    def _key(self, i: int) -> str:
        p = self.store[i]
        return hashlib.sha1(f"{p.name}\0{p.title}\0{p.text}".encode()).hexdigest()[:20]

    def _encode(self, texts: list[str], query: bool = False):
        kw: dict = {"normalize_embeddings": True, "show_progress_bar": False}
        if query and "query" in (getattr(self.encoder, "prompts", None) or {}):
            kw["prompt_name"] = "query"
        if not query:
            kw["batch_size"] = self.batch_size
        return self._np.asarray(self.encoder.encode(texts, **kw), dtype="float32")

    def _embed_all(self, cache_dir: Path | None):
        np = self._np
        keys = [self._key(i) for i in range(len(self.store))]
        cached: dict[str, int] = {}
        old = None
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            kf, vf = cache_dir / f"{slug}.keys.json", cache_dir / f"{slug}.npy"
            if kf.exists() and vf.exists():
                try:
                    old_keys = json.loads(kf.read_text(encoding="utf-8"))
                    old = np.load(vf)
                    cached = {k: j for j, k in enumerate(old_keys) if j < len(old)}
                except (OSError, ValueError):
                    cached, old = {}, None
        todo = [i for i, k in enumerate(keys) if k not in cached]
        dim = None
        new = None
        if todo:
            texts = [f"{self.store[i].title}\n{self.store[i].text}" for i in todo]
            new = self._encode(texts)
            dim = new.shape[1]
        elif old is not None:
            dim = old.shape[1]
        if dim is None:  # empty corpus
            return np.zeros((0, 1), dtype="float32")
        mat = np.zeros((len(keys), dim), dtype="float32")
        pos = {i: j for j, i in enumerate(todo)}
        for i, k in enumerate(keys):
            mat[i] = new[pos[i]] if i in pos else old[cached[k]]
        if cache_dir is not None and todo:
            np.save(cache_dir / f"{slug}.npy", mat)
            (cache_dir / f"{slug}.keys.json").write_text(json.dumps(keys), encoding="utf-8")
        return mat

    def search(self, query: str, k: int) -> list[int]:
        np = self._np
        if not len(self.matrix):
            return []
        qv = self._encode([query], query=True)[0]
        sims = self.matrix @ qv
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-sims[top], kind="stable")]]


# ------------------------------------------------------------------------------------- fused
class FusedRanker:
    """Reciprocal-rank fusion of BM25 and dense rankings (Cormack et al., 2009)."""

    name = "fused"

    def __init__(self, bm25: BM25Ranker, dense: DenseRanker, depth: int = 100, c: int = 60):
        self.bm25, self.dense, self.depth, self.c = bm25, dense, depth, c

    def search(self, query: str, k: int) -> list[int]:
        score: dict[int, float] = {}
        for ranking in (self.bm25.search(query, self.depth), self.dense.search(query, self.depth)):
            for r, pid in enumerate(ranking):
                score[pid] = score.get(pid, 0.0) + 1.0 / (self.c + r + 1)
        return [pid for pid, _ in heapq.nlargest(k, score.items(), key=lambda x: (x[1], -x[0]))]


# ------------------------------------------------------------------------------------ rerank
def _ce_max_length(model_name: str, want: int = 1024) -> int:
    """The cross-encoder's input cap: ``want``, or fewer when the model has fewer positions."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:  # noqa: BLE001
        return want
    pos = getattr(cfg, "max_position_embeddings", None)
    if not isinstance(pos, int) or pos <= 0:
        return want
    if getattr(cfg, "model_type", "") in ("roberta", "xlm-roberta", "camembert"):
        pos -= 2
    return min(want, pos)


class RerankRanker:
    """The dense top ``depth`` reordered by a cross-encoder (default ``BAAI/bge-reranker-v2-m3``)."""

    name = "rerank"

    def __init__(
        self,
        dense: DenseRanker,
        model: str = "BAAI/bge-reranker-v2-m3",
        depth: int = 100,
        device: str | None = None,
        cross_encoder=None,
    ):
        self.dense, self.depth = dense, depth
        if cross_encoder is None:
            _, st = _need_dense()
            cross_encoder = st.CrossEncoder(model, max_length=_ce_max_length(model), device=device)
        self.ce = cross_encoder

    def search(self, query: str, k: int) -> list[int]:
        cands = self.dense.search(query, max(k, self.depth))
        if not cands:
            return []
        store = self.dense.store
        pairs = [(query, f"{store[i].title}\n{store[i].text}") for i in cands]
        scores = list(self.ce.predict(pairs, show_progress_bar=False))
        order = sorted(range(len(cands)), key=lambda j: (-float(scores[j]), j))
        return [cands[j] for j in order[:k]]


def make_ranker(kind: str, store: PassageStore, settings, cache_dir: Path | None = None) -> Ranker:
    """Build the ranker a variant names: ``bm25``, ``dense``, ``fused`` or ``rerank``."""
    if kind == "bm25":
        return BM25Ranker(store)
    dense = DenseRanker(
        store,
        model=settings.embed_model,
        cache_dir=cache_dir,
        device=settings.device,
    )
    if kind == "dense":
        return dense
    if kind == "fused":
        return FusedRanker(BM25Ranker(store), dense, depth=settings.rerank_depth)
    if kind == "rerank":
        return RerankRanker(
            dense, model=settings.rerank_model, depth=settings.rerank_depth, device=settings.device
        )
    raise ValueError(f"unknown retriever {kind!r}; use bm25, dense, fused or rerank")


__all__ = [
    "BM25Ranker",
    "DenseRanker",
    "FusedRanker",
    "MissingExtraError",
    "Ranker",
    "RerankRanker",
    "make_ranker",
]

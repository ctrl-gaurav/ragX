"""The named ragX variants, as defined in the paper.

- ``ragx``: dense ranker, forage loop and slot chain, backfill 48. The full method.
- ``ragx-core``: as ``ragx`` without the backfill.
- ``ragx-lexical`` (default): BM25 ranker, no backfill. Needs no embedding index or GPU.
- ``ragx-fused``: BM25 and dense rankings fused by reciprocal rank, no backfill.
- ``ragx-rerank``: the dense top 100 reordered by a cross-encoder, no backfill.
- ``ragx-fast``: dense ranker, forage loop only. The cheapest variant.
- ``ragx-think``: ``ragx`` with thinking on in every call.
- ``classic``: ragX's original file-level agent with the grep toolbelt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    retriever: str  # bm25 | dense | fused | rerank | none
    route: str  # deep | fast | classic
    backfill: int
    think: bool = False
    summary: str = ""


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant(
            "ragx",
            "dense",
            "deep",
            48,
            summary="The full method: forage loop, slot chain, backfill.",
        ),
        Variant(
            "ragx-core", "dense", "deep", 0, summary="Forage loop and slot chain, no backfill."
        ),
        Variant(
            "ragx-lexical", "bm25", "deep", 0, summary="BM25 only: no embedding index, no GPU."
        ),
        Variant("ragx-fused", "fused", "deep", 0, summary="BM25 and dense rankings fused by RRF."),
        Variant(
            "ragx-rerank",
            "rerank",
            "deep",
            0,
            summary="Dense top 100 reordered by a cross-encoder.",
        ),
        Variant("ragx-fast", "dense", "fast", 0, summary="Forage loop only: the cheapest variant."),
        Variant("ragx-think", "dense", "deep", 48, think=True, summary="ragx with thinking on."),
        Variant("classic", "none", "classic", 0, summary="The original grep-toolbelt agent."),
    )
}

DEFAULT_VARIANT = "ragx-lexical"
_ALIASES = {
    "ragx-full": "ragx",
    "full": "ragx",
    "ragx-lex": "ragx-lexical",
    "lexical": "ragx-lexical",
}


def get_variant(name: str) -> Variant:
    key = name.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose one of: {', '.join(VARIANTS)}")
    return VARIANTS[key]


__all__ = ["DEFAULT_VARIANT", "VARIANTS", "Variant", "get_variant"]

"""Iterative hop resolution — the ``route="deep"`` phase.

On deep multi-hop questions the later-hop documents are often *lexically unreachable* from the
question itself: "who owns the record label of the X performer" shares no vocabulary with the
"Warner Records" article, so no query rewriting can rank it. The bridge entity has to be
*resolved first*, then searched. This module decomposes the question into ordered sub-questions
with ``#N`` placeholders, then walks the chain: rank the current sub-question (with earlier
answers substituted in), answer it from distilled excerpts with one cheap LLM call, and feed the
answer forward. Every hop's ranked files join the surfaced set the synthesizer composes over.
Embedding-free throughout — the per-hop retrieval is the same df-BM25 ranker.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..llm import LLM
from ..tools import Toolbelt
from ..types import Event

_DECOMP_SYS = (
    "Break the multi-hop question into the minimal ordered list of 2-4 simple single-fact "
    "sub-questions. When a later sub-question depends on an earlier answer, write #1 (or #2, #3) "
    "where that answer goes. Output ONLY the sub-questions, one per line, no numbering.\n"
    "Example: 'Who owns the record label of the performer of Song X?' ->\n"
    "Who performs Song X?\nWhat record label does #1 record for?\nWho owns #2?"
)
_SUBQ_SYS = (
    "Answer the question with the SHORTEST exact entity, name, date, or number from the excerpts. "
    "Output ONLY the answer. If the excerpts lack it, output your best guess."
)


def decompose(question: str, llm: LLM, max_hops: int = 4) -> list[str]:
    """Ordered sub-questions with ``#N`` placeholders (empty list if decomposition fails)."""
    reply = llm.chat(
        [
            {"role": "system", "content": _DECOMP_SYS},
            {"role": "user", "content": f"Question: {question}\n\nSub-questions:"},
        ],
        max_tokens=160,
    )
    subqs = [
        re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln).strip() for ln in (reply or "").splitlines()
    ]
    return [s for s in subqs if s and "?" in s][:max_hops]


def resolve_chain(
    question: str,
    tools: Toolbelt,
    llm: LLM,
    *,
    k: int = 4,
    max_hops: int = 4,
) -> Iterator[Event]:
    """Walk the decomposed chain hop-by-hop; yields ``hop`` events.

    Side effects: each hop's top-ranked files are added to ``tools.seen_files`` (so the
    synthesizer composes over them), and the resolved ``question -> answer`` chain is stored on
    ``tools.chain_notes`` for the synthesis prompt.
    """
    subqs = decompose(question, llm, max_hops=max_hops)
    answers: list[str] = []
    notes: list[str] = []
    for sq in subqs:
        for j, a in enumerate(answers):
            sq = sq.replace(f"#{j + 1}", a)
        files = tools.rank_files(sq, top=k)
        parts = []
        for j, src in enumerate(files):
            tools.seen_files.add(src)
            body = tools.file_text(src)
            if body:
                excerpt = tools.distill(sq, body)
                parts.append(f"[{src}]\n{excerpt}")
                # top files per hop feed the composer directly (Toolbelt.hop_evidence): they are
                # lexically unreachable from the question, so question-ranked extras would drop them
                if j < 2 and src not in tools.hop_evidence:
                    tools.hop_evidence[src] = excerpt
        reply = llm.chat(
            [
                {"role": "system", "content": _SUBQ_SYS},
                {
                    "role": "user",
                    "content": f"Question: {sq}\n\nExcerpts:\n"
                    + "\n\n".join(parts)
                    + "\n\nAnswer:",
                },
            ],
            max_tokens=48,
        )
        ans = (reply or "").strip().splitlines()[0][:80] if reply else ""
        answers.append(ans)
        notes.append(f"- {sq} -> {ans}")
        yield Event("hop", f"{sq} → {ans}")
    tools.chain_notes = notes


__all__ = ["decompose", "resolve_chain"]

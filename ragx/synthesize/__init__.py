"""Synthesis + grounding gate.

Given the evidence the agent actually read, produce a final, grounded answer with exact
``file:line`` citations. Optionally re-synthesize with a stronger ``answer_model`` (the
scout/synthesizer split) and verify each citation's quote really occurs at that location.
"""

from __future__ import annotations

import re

from ..llm import LLM
from ..tools import Toolbelt
from ..types import Answer, Citation

_WORD = re.compile(r"[A-Za-z0-9]+")

_SYNTH_SYS = (
    "You answer a question using ONLY the provided excerpts from a document corpus.\n"
    "- Give the SHORTEST exact answer that fully answers the question — usually a name, entity, "
    "number, date, or yes/no — optionally followed by one short clarifying clause.\n"
    "- Copy names, numbers and dates verbatim from the excerpts; never paraphrase a fact or add "
    "information that is not in the excerpts.\n"
    "- Lead with the answer. Do not restate the question or explain your reasoning.\n"
    "- If the excerpts do not contain the answer, reply exactly: I don't know."
)

# compose_reasoning pass 1 (see Settings.compose_reasoning). Free-form analysis only — the
# strict answer shape is enforced by running the normal _SYNTH_SYS compose over this analysis
# afterwards. A single reason-then-answer call is NOT robust: over 20+ excerpts the model
# rambles past any token budget and the answer line never arrives; two calls make the answer
# structural instead of hoping the prompt is obeyed.
_REASON_SYS = (
    "You analyze a question against provided excerpts from a document corpus. The question may "
    "combine several facts (dates, numbers, names) and may require arithmetic (differences of "
    "years, counts) or tracking exactly which entity/attribute is asked for. Reason briefly "
    "(at most ~120 words): pull out each needed fact with its source, do any arithmetic digit "
    "by digit, and re-read the question to confirm WHICH thing it asks for. End by stating the "
    "most likely answer."
)

# Deterministic answer-type detection. Failure analysis on multi-hop QA showed the dominant
# residual errors are wrong-TYPE answers (a person when a place is asked; an entity when a date
# is asked) and two-option comparison questions answered off-menu. One hint line fixes many of
# those (+1.7 to +1.8 answer-F1) without the overthinking that heavier chain-of-thought
# synthesis prompts induce.
_QT_YESNO = re.compile(
    r"^(are|is|was|were|did|do|does|has|have|had|can|could|will|would)\b", re.IGNORECASE
)
_QT_OPTS = re.compile(r"\b(?:which|who|what)\b[^,?]*,\s*(.+?)\s+or\s+(.+?)\s*\?", re.IGNORECASE)
_OPT_HINT = "The answer is ONE of the options named in the question — copy that option exactly."


def answer_type_hint(question: str) -> str | None:
    """A one-line expected-answer-type hint for the synthesis prompt (``None`` if undetected)."""
    q = question.strip()
    ql = q.lower()
    m = _QT_OPTS.search(q)
    if m and len(m.group(1)) > 2:
        return _OPT_HINT
    if _QT_YESNO.match(ql):
        # "Is A or B the ..." is an option question, not yes/no
        return _OPT_HINT if " or " in ql else "Answer with exactly 'yes' or 'no'."
    if re.search(r"\bhow (many|much|long|old|tall|far)\b", ql):
        return "The answer is a number (or a short quantity phrase)."
    if re.search(r"\b(what year|which year|when|what date|date of (birth|death))\b", ql):
        return "The answer is a date or year."
    if re.search(
        r"^where\b|\bwhere (is|was|are|were|did|does)\b|place of (birth|death|burial)"
        r"|\bwhat (city|town|country|state|place|county)\b",
        ql,
    ):
        return "The answer is a PLACE name (city, region, country, or specific site)."
    if re.search(r"^(who|whose|whom)\b|\bwho (is|was|are|were)\b", ql):
        return "The answer is a person's name (or organization)."
    return None


def hint_kind(question: str) -> str:
    """Coarse category of the expected answer (``number`` · ``date`` · ``comparison`` · ``yesno``
    · ``place`` · ``person`` · ``other``); drives ``compose_reasoning="auto"``."""
    h = answer_type_hint(question) or ""
    if h == _OPT_HINT:
        return "comparison"
    if "yes' or 'no" in h:
        return "yesno"
    if "number" in h:
        return "number"
    if "date" in h:
        return "date"
    if "PLACE" in h:
        return "place"
    if "person" in h:
        return "person"
    return "other"


_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "did",
        "does",
        "do",
        "who",
        "what",
        "which",
        "where",
        "when",
        "whose",
        "whom",
        "how",
        "why",
        "that",
        "this",
        "it",
        "its",
        "as",
        "from",
        "into",
        "over",
        "under",
        "than",
        "then",
    }
)


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s)}


def synthesize(
    question: str,
    draft: str,
    tools: Toolbelt,
    *,
    answer_llm: LLM | None = None,
    verify: bool = True,
    abstain: bool = False,
    compose_surfaced: bool = True,
    surf_cap: int = 16,
    typed_hints: bool = True,
    distill_extras: bool = False,
    chain_notes: bool = False,
    hop_priority: bool = True,
    compose_reasoning: bool = False,
) -> Answer:
    reads = tools.reads
    # Compose over the reads PLUS a capped set of surfaced-but-unread docs (see
    # Toolbelt.surfaced_evidence). Augmenting recovers answer-F1 where the agent under-reads (+16 F1
    # HotpotQA, +7.5 MuSiQue); the cap stops extra docs from diluting cleanly-read comparison
    # questions (2WikiMultiHop). cap=8 is parity-or-better on answer-F1 on both in our sweep, and
    # the extras are picked by df-BM25 relevance to the question (not first-surfaced order).
    # route="deep" surfaces many more files (per-hop ranking), so its extras are sentence-distilled
    # (distill_extras) to fit the budget instead of hard-capped.
    evidence_src = (
        tools.surfaced_evidence(
            extra_cap=surf_cap,
            rank_query=question,
            distill_query=question if distill_extras else None,
            priority=(tools.hop_evidence or None) if hop_priority else None,
        )
        if compose_surfaced
        else reads
    )
    # 1. optional re-synthesis from gathered evidence (improves small-model answers)
    text = draft
    if answer_llm is not None and evidence_src:
        max_docs = 28 if distill_extras else 12
        evidence = "\n\n".join(
            f"[{src}:{a}-{b}]\n{body}" for src, a, b, body in evidence_src[:max_docs]
        )
        hint = answer_type_hint(question) if typed_hints else None
        hint_line = f"\n({hint})" if hint else ""
        # Chain NOTES are prompt-only and off by default (Settings.deep_chain_notes): a wrong
        # resolved answer misleads the composer more than the hop's document (kept in evidence
        # via hop_evidence) helps. Measured: MuSiQue-4B 24.5 -> 28.9 F1 from dropping the notes.
        chain = (
            "\n\nResolved sub-questions:\n" + "\n".join(tools.chain_notes)
            if (chain_notes and tools.chain_notes)
            else ""
        )
        user_prompt = f"Question: {question}\n\nExcerpts:\n{evidence}{chain}{hint_line}\n\nAnswer:"
        # compose_reasoning may be True / False / "auto": in auto mode the reasoning pass runs
        # only for question types where it was measured to pay (numbers, dates, comparisons);
        # entity questions keep the single strict compose.
        if compose_reasoning == "auto":
            compose_reasoning = hint_kind(question) in ("number", "date", "comparison")
        if compose_reasoning:
            # Two-step compose (Settings.compose_reasoning): free-form analysis over the
            # evidence, then the strict compose over the question + that analysis. Fixes both
            # dominant multi-constraint failure modes — arithmetic that the answer-only prompt
            # has no room for, and answering an intermediate entity instead of the asked slot
            # (FRAMES: 35.3 vs 22.9 answer-F1, identical retrieval; +1 call).
            reasoning = answer_llm.chat(
                [
                    {"role": "system", "content": _REASON_SYS},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nExcerpts:\n{evidence}"
                            f"{chain}{hint_line}\n\nReasoning:"
                        ),
                    },
                ],
                max_tokens=400,
            ).strip()
            if reasoning:
                user_prompt = (
                    f"Question: {question}\n\nAnalysis of the source documents:\n{reasoning}"
                    f"{hint_line}\n(Even if the analysis is inconclusive, commit to the single "
                    "most likely specific answer.)\n\nAnswer:"
                )
        text = (
            answer_llm.chat(
                [
                    {"role": "system", "content": _SYNTH_SYS},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
            ).strip()
            or draft
        )

    # 2. citations come from the evidence the composer ACTUALLY saw (reads AND the surfaced
    #    extras), never from reads alone: an answer composed from an unread surfaced document
    #    must cite that document, not an unrelated read.
    pool = evidence_src if (answer_llm is not None and evidence_src) else reads
    ans_tok = _tokens(text) - _STOP
    ans_norm = " ".join(_WORD.findall(text.lower()))
    cites: list[Citation] = []
    grounded_any = False
    for src, a, b, body in pool:
        body_norm = " ".join(_WORD.findall(body.lower()))
        span_hit = bool(ans_norm) and ans_norm in body_norm  # the answer occurs verbatim here
        overlap = ans_tok & _tokens(body)
        # covered: (almost) every content token of the answer appears in THIS excerpt —
        # "Dana Whitfield founded Acme in 2014" is grounded by a line that has all those tokens
        covered = bool(ans_tok) and len(overlap) >= 0.8 * len(ans_tok)
        if not verify or span_hit or covered:
            quote = body.strip().splitlines()[0][:160] if body.strip() else ""
            cites.append(Citation(path=src, line_start=a, line_end=b, quote=quote))
            grounded_any = grounded_any or span_hit or covered
    # de-dup, keep order
    seen: set[tuple] = set()
    uniq: list[Citation] = []
    for c in cites:
        key = (c.path, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq and pool:  # nothing overlaps: cite what was read, but the answer is NOT grounded
        uniq = [
            Citation(p, a, b, body.splitlines()[0][:160] if body else "") for p, a, b, body in reads
        ]

    # `grounded` means: the answer (or ≥80% of its content tokens) is present in ONE cited
    # excerpt the composer actually saw. Scattered token overlap is not grounding.
    grounded = grounded_any if verify else bool(uniq)
    if not pool and not tools.reads:
        grounded = False
    if abstain and not grounded:
        text = "I couldn't find enough evidence in the corpus to answer that confidently."
        grounded = False
    return Answer(text=text, citations=uniq[:8], grounded=grounded)


def answers_agree(a: str, b: str, threshold: float = 0.8) -> bool:
    """Token-F1 agreement between two candidate answers. Used by route="both" as a
    perturbation-consistency signal: when the notes-off composition and the (sometimes-wrong)
    notes-injected composition agree, the answer is right far above base rate (answer-F1
    75.5/83.2 vs a 61/63 base on HotpotQA/2WikiMultiHop at 60-64% coverage) — a deterministic
    confidence signal that needs no extra verification machinery. The signal comes from the
    DIVERSITY of the two prompts; a same-policy re-compose agrees too often to be informative."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ta == tb
    overlap = len(ta & tb)
    return (2 * overlap / (len(ta) + len(tb))) >= threshold


__all__ = ["answer_type_hint", "answers_agree", "hint_kind", "synthesize"]

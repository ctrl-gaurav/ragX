"""Configuration for ragX. Loads ``.env`` and exposes typed settings.

Model strings use the form ``provider:model``, e.g.::

    openai:gpt-4o-mini
    anthropic:claude-3-5-sonnet-latest
    google:gemini-1.5-flash
    groq:llama-3.3-70b-versatile
    together:Qwen/Qwen2.5-7B-Instruct-Turbo
    cerebras:gpt-oss-120b
    vllm:Qwen/Qwen3.5-9B                 (an OpenAI-compatible server, e.g. vLLM)
    hf:Qwen/Qwen3.5-4B                   (in-process transformers; needs pyragx[local])

Append ``@low`` / ``@medium`` / ``@high`` to set the reasoning effort of hosted reasoning
models (``openai:o4-mini@high``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .variants import DEFAULT_VARIANT

# Load .env from cwd and parents (never logs values).
load_dotenv()


def _default_model() -> str:
    """Pick a sensible default based on which API keys are present."""
    for env, spec in [
        ("OPENAI_API_KEY", "openai:gpt-4o-mini"),
        ("ANTHROPIC_API_KEY", "anthropic:claude-3-5-sonnet-latest"),
        ("GOOGLE_API_KEY", "google:gemini-1.5-flash"),
        ("GROQ_API_KEY", "groq:llama-3.3-70b-versatile"),
        ("TOGETHER_API_KEY", "together:Qwen/Qwen2.5-7B-Instruct-Turbo"),
        ("CEREBRAS_API_KEY", "cerebras:gpt-oss-120b"),
    ]:
        if os.environ.get(env):
            return spec
    # No API key? Fall back to a small local model (requires pyragx[local]).
    return os.environ.get("RAGX_MODEL", "hf:Qwen/Qwen3.5-4B")


class Settings(BaseModel):
    """Runtime settings for a :class:`~ragx.core.RagX` instance."""

    model: str = Field(default_factory=_default_model)
    scout_model: str | None = None  # model that drives the agent (defaults to model)
    answer_model: str | None = None  # model for the final compose call (defaults to model)

    # ---- the method (see ragx.variants) ---------------------------------------------------
    variant: str = DEFAULT_VARIANT  # ragx | ragx-core | ragx-lexical | ragx-fused | ragx-rerank |
    #                                 ragx-fast | ragx-think | classic
    retriever: str | None = None  # override the variant's ranker: bm25 | dense | fused | rerank
    backfill: int | None = None  # override the variant's backfill depth
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"  # dense ranker (needs pyragx[dense])
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # cross-encoder for ragx-rerank
    rerank_depth: int = 100  # dense candidates the reranker (and fusion) consider
    device: str | None = None  # "cpu", "cuda", ... for the embedder and reranker
    passage_words: int = 120  # passage size when files are cut for ranking

    steps: int = 10  # forager turns
    rank_k: int = 8  # passages listed per RANK
    hop_k: int = 4  # passages retrieved per hop of the slot chain
    max_hops: int = 4  # sub-questions at most
    read_cap: int = 8  # read passages composed
    extra_cap: int = 12  # extra passages composed (hop passages first)
    tools: list[str] = Field(default_factory=lambda: ["RANK", "READ"])  # add "SEARCH" for grep
    fence: bool = True  # wrap tool results and evidence as untrusted data
    compose_reasoning: bool | str = False  # reasoning pass before the compose (True / "auto")

    # ---- models ---------------------------------------------------------------------------
    think: bool | None = None  # thinking mode on every call (None = the variant decides)
    think_budget: int = 8192  # extra tokens a thinking call may spend
    max_context_tokens: int | None = None  # reader context window; local models default 32768
    temperature: float = 0.0
    cache: bool = False  # memoize LLM responses and tool results

    # ---- the classic agent (variant="classic") --------------------------------------------
    max_steps: int = 5
    auto_budget: bool = True
    max_result_lines: int = 40
    max_read_chars: int = 4000
    prerank: bool = True
    route: str = "agentic"  # agentic | adaptive | deep | both
    single_shot_reads: int = 4
    deep_hops: int = 4
    deep_surf_cap: int = 12
    deep_chain_notes: bool = False
    semantic_fallback: bool = False
    fuzzy: bool = True
    context_lines: int = 1
    compose_surfaced: bool = True
    surf_cap: int = 16
    sandbox_observations: bool = False

    # ---- ingestion ------------------------------------------------------------------------
    cache_dir: str | None = None  # where the text mirror, index and embeddings live
    include: list[str] = Field(default_factory=list)  # glob allowlist (empty = all supported)
    exclude: list[str] = Field(default_factory=lambda: [".git/**", "node_modules/**", ".ragx/**"])
    follow_symlinks: bool = False

    # ---- answers --------------------------------------------------------------------------
    verify_citations: bool = True  # an answer is grounded only if a cited passage contains it
    abstain_on_weak_evidence: bool = False
    typed_hints: bool = True  # answer-type hint in the compose prompt

    def budget_for(self, model: str | None = None) -> tuple[int, int]:
        """(max_steps, surf_cap) for the classic agent: small scouts (<= 2.5B) forage less."""
        if not self.auto_budget:
            return self.max_steps, self.surf_cap
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model or self.scout())
        if not m:
            return self.max_steps, self.surf_cap
        if float(m.group(1)) <= 2.5:
            return min(self.max_steps, 2), min(self.surf_cap, 8)
        return self.max_steps, self.surf_cap

    def scout(self) -> str:
        return self.scout_model or self.model

    def answerer(self) -> str:
        return self.answer_model or self.model

    def resolve_cache_dir(self, corpus: Path) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        if corpus.is_file():
            return corpus.parent / ".ragx"
        return corpus / ".ragx"

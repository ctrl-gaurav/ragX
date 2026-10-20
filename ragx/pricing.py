"""Rough per-query cost estimation.

Maps a ``provider:model`` spec to public list prices (USD per million tokens) so callers
can see an approximate ``$`` cost next to latency and token counts. Local backends
(``hf:`` / ``vllm:``) are billed at $0 — the whole point of running them yourself.

Prices drift; treat these as configurable estimates, not quotes. Override or extend at
runtime via :func:`set_price`. Unknown models return ``None`` (cost simply not shown).
"""

from __future__ import annotations

# (input_usd_per_mtok, output_usd_per_mtok), matched by case-insensitive substring of the
# full ``provider:model`` spec. Most specific (longest) match wins.
_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4.1-mini": (0.40, 1.60),
    "openai:gpt-4.1": (2.00, 8.00),
    # Google
    "google:gemini-1.5-flash": (0.075, 0.30),
    "google:gemini-1.5-pro": (1.25, 5.00),
    "google:gemini-2.0-flash": (0.10, 0.40),
    # Anthropic
    "anthropic:claude-3-5-haiku": (0.80, 4.00),
    "anthropic:claude-3-5-sonnet": (3.00, 15.00),
    # Groq (OSS, very cheap)
    "groq:llama-3.3-70b": (0.59, 0.79),
    "groq:llama-3.1-8b": (0.05, 0.08),
    # Together / Fireworks (representative OSS hosting)
    "together:qwen": (0.20, 0.20),
    "together:meta-llama": (0.20, 0.20),
    "fireworks:": (0.20, 0.20),
    "cerebras:": (0.10, 0.10),
}

_LOCAL_PREFIXES = ("hf:", "vllm:")


def set_price(spec_substring: str, input_per_mtok: float, output_per_mtok: float) -> None:
    """Register/override a price (USD per million tokens) for specs containing the substring."""
    _PRICES[spec_substring.lower()] = (input_per_mtok, output_per_mtok)


def price_for(spec: str) -> tuple[float, float] | None:
    """Return (input, output) USD/Mtok for a spec, or ``None`` if unknown. Local = (0, 0)."""
    s = spec.lower()
    if s.startswith(_LOCAL_PREFIXES):
        return (0.0, 0.0)
    best: tuple[float, float] | None = None
    best_len = -1
    for key, price in _PRICES.items():
        if key in s and len(key) > best_len:
            best, best_len = price, len(key)
    return best


def estimate_cost(spec: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate the USD cost of a call given token counts. ``None`` if the model is unpriced."""
    price = price_for(spec)
    if price is None:
        return None
    pin, pout = price
    return round((prompt_tokens * pin + completion_tokens * pout) / 1_000_000, 6)

"""
Model pricing — turns token counts into dollars.

Rates are Anthropic first-party API list prices in USD per million tokens.
Cache reads bill at ~0.1x the input rate and cache writes at ~1.25x, which is
why the boardroom's frozen analyst personas are worth caching: eight identical
system prefixes per cycle otherwise pay full input price every time.

Update ``RATES`` when Anthropic publishes new prices.  An unknown model falls
back to ``_FALLBACK`` (Sonnet-tier) so an unrecognised ID over-estimates rather
than silently billing a user zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rate:
    """USD per million tokens."""
    input_per_mtok: float
    output_per_mtok: float

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * 1.25

    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * 0.10


# Anthropic first-party list prices, USD / 1M tokens.
RATES: dict[str, Rate] = {
    "claude-fable-5":    Rate(10.00, 50.00),
    "claude-opus-5":     Rate(5.00, 25.00),
    "claude-opus-4-8":   Rate(5.00, 25.00),
    "claude-opus-4-7":   Rate(5.00, 25.00),
    "claude-opus-4-6":   Rate(5.00, 25.00),
    "claude-sonnet-5":   Rate(2.00, 10.00),
    "claude-sonnet-4-6": Rate(3.00, 15.00),
    "claude-haiku-4-5":  Rate(1.00, 5.00),
}

_FALLBACK = Rate(3.00, 15.00)


def rate_for(model: str) -> Rate:
    """Resolve a rate, tolerating dated snapshot IDs and version suffixes.

    ``claude-haiku-4-5-20251001`` resolves to the ``claude-haiku-4-5`` rate;
    an ID matching nothing falls back to Sonnet-tier pricing.
    """
    if not model:
        return _FALLBACK
    key = model.strip().lower()
    if key in RATES:
        return RATES[key]
    # Longest known prefix wins, so "claude-opus-4-8-2026..." beats "claude-opus".
    best: Optional[str] = None
    for known in RATES:
        if key.startswith(known) and (best is None or len(known) > len(best)):
            best = known
    return RATES[best] if best else _FALLBACK


def cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Dollar cost of one API call.

    ``input_tokens`` should be the *uncached* input count — Anthropic reports
    cache reads and writes separately in ``usage``, and double-counting them
    here would roughly double every cached call's apparent cost.
    """
    r = rate_for(model)
    return (
        input_tokens       / 1e6 * r.input_per_mtok
        + output_tokens      / 1e6 * r.output_per_mtok
        + cache_read_tokens  / 1e6 * r.cache_read_per_mtok
        + cache_write_tokens / 1e6 * r.cache_write_per_mtok
    )


def format_usd(amount: float) -> str:
    """Human-readable dollars that stay readable at sub-cent amounts."""
    if amount >= 1:
        return f"${amount:,.2f}"
    if amount >= 0.01:
        return f"${amount:.3f}"
    return f"${amount:.5f}"


def format_usd_md(amount: float) -> str:
    """Same, escaped for Markdown.

    Streamlit renders Markdown in captions, progress labels and ``st.markdown``,
    where a pair of unescaped ``$`` is parsed as LaTeX — "``$1``/``$5`` per
    million" silently renders as "5 per million". Use this anywhere the string
    lands in Markdown; ``format_usd`` is fine inside ``st.metric``.
    """
    return format_usd(amount).replace("$", r"\$")

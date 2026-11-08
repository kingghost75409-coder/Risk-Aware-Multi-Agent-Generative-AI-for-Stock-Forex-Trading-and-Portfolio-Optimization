"""Labeled sample financial headlines, for demo/testing only.

Clearly synthetic/hand-authored — not scraped from a real news feed. A
production deployment would replace `get_sample_headlines` with a real
news API client (NewsAPI, Benzinga, Alpha Vantage News, etc.) returning
the same {"text", "published_at"} shape.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_TEMPLATES_BY_TONE = {
    "positive": [
        "{sym} beats estimates on strong quarterly revenue growth",
        "{sym} raised guidance for next fiscal year amid robust demand",
        "Analysts upgrade {sym} citing record revenue and margin expansion",
        "{sym} announces share buyback program worth billions",
        "{sym} stock hits all-time high after positive earnings beat",
    ],
    "neutral": [
        "{sym} to report quarterly earnings next week",
        "{sym} holds annual shareholder meeting, no major announcements",
        "Market awaits {sym} guidance update amid mixed sector trends",
        "{sym} announces new product lineup for upcoming quarter",
    ],
    "negative": [
        "{sym} misses estimates as guidance cut weighs on outlook",
        "Analysts downgrade {sym} citing margin pressure and slowing demand",
        "{sym} faces regulatory scrutiny amid antitrust concerns",
        "{sym} announces layoffs as part of cost-cutting plan",
        "{sym} stock slides on profit warning ahead of earnings",
    ],
}

# deterministic per-symbol tone mix so results are reproducible across runs
_SYMBOL_TONE_BIAS = {
    "AAPL": ["positive", "neutral", "positive", "neutral"],
    "MSFT": ["positive", "positive", "neutral", "positive"],
    "NVDA": ["positive", "positive", "positive", "neutral"],
    "AMZN": ["neutral", "positive", "neutral", "positive"],
    "GOOGL": ["neutral", "negative", "neutral", "positive"],
    "TSLA": ["negative", "positive", "neutral", "negative"],
}


def get_sample_headlines(symbol: str, n: int = 8) -> list[dict]:
    symbol = symbol.upper()
    tones = _SYMBOL_TONE_BIAS.get(symbol, ["neutral", "positive", "neutral", "negative"])
    now = datetime.now(timezone.utc)
    headlines: list[dict] = []
    for i in range(n):
        tone = tones[i % len(tones)]
        templates = _TEMPLATES_BY_TONE[tone]
        text = templates[i % len(templates)].format(sym=symbol)
        published = now - timedelta(hours=(n - i) * 6)
        headlines.append({"text": text, "published_at": published.isoformat(), "tone_label": tone})
    return headlines

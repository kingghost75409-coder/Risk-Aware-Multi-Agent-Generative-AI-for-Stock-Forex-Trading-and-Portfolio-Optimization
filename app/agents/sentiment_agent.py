"""Financial Sentiment Agent.

Uses VADER (a lexicon/rule-based sentiment model, not an LLM) so sentiment
scores are deterministic and reproducible. A small finance-domain lexicon
boost is layered on top of the general VADER lexicon to better capture
financial terms VADER wasn't built for (e.g. "beat estimates", "guidance cut").

In production this module would pull real headlines from a news API; here
it accepts a list of headline strings (see app/data/news_sample.py for a
labeled sample set used for demo/testing) so the scoring logic itself is
fully real and testable without a live news feed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_FINANCE_LEXICON_BOOST = {
    "beat estimates": 2.5, "beats estimates": 2.5, "raised guidance": 2.8,
    "guidance cut": -2.8, "missed estimates": -2.5, "misses estimates": -2.5,
    "downgrade": -2.0, "upgrade": 2.0, "record revenue": 2.5,
    "record profit": 2.5, "layoffs": -2.2, "bankruptcy": -3.5,
    "sec investigation": -3.0, "buyback": 1.5, "share buyback": 1.5,
    "earnings beat": 2.5, "earnings miss": -2.5, "profit warning": -2.8,
    "all-time high": 2.0, "recall": -1.8, "lawsuit": -1.5,
    "antitrust": -1.5, "acquisition": 1.2, "merger": 1.0,
}


class SentimentResult(BaseModel):
    symbol: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    label: str  # POSITIVE / NEUTRAL / NEGATIVE
    confidence: float = Field(ge=0.0, le=1.0)
    trend: str  # IMPROVING / STABLE / DETERIORATING
    n_headlines: int
    headline_scores: list[dict]
    timestamp: str


class SentimentAgent:
    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        for phrase, boost in _FINANCE_LEXICON_BOOST.items():
            self._vader.lexicon[phrase] = boost

    def _score_headline(self, text: str) -> float:
        return self._vader.polarity_scores(text)["compound"]

    def analyze(self, symbol: str, headlines: list[dict]) -> SentimentResult:
        """headlines: list of {"text": str, "published_at": iso-str}, most-recent last."""
        if not headlines:
            return SentimentResult(
                symbol=symbol, sentiment_score=0.0, label="NEUTRAL", confidence=0.0,
                trend="STABLE", n_headlines=0, headline_scores=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        scored = []
        for h in headlines:
            s = self._score_headline(h["text"])
            scored.append({"text": h["text"], "published_at": h.get("published_at", ""), "score": round(s, 3)})

        avg = sum(s["score"] for s in scored) / len(scored)
        label = "POSITIVE" if avg > 0.15 else ("NEGATIVE" if avg < -0.15 else "NEUTRAL")
        confidence = min(0.95, 0.4 + abs(avg) * 0.6 + min(len(scored), 10) * 0.02)

        if len(scored) >= 4:
            half = len(scored) // 2
            first_half_avg = sum(s["score"] for s in scored[:half]) / half
            second_half_avg = sum(s["score"] for s in scored[half:]) / (len(scored) - half)
            diff = second_half_avg - first_half_avg
            trend = "IMPROVING" if diff > 0.1 else ("DETERIORATING" if diff < -0.1 else "STABLE")
        else:
            trend = "STABLE"

        return SentimentResult(
            symbol=symbol,
            sentiment_score=round(avg, 3),
            label=label,
            confidence=round(confidence, 3),
            trend=trend,
            n_headlines=len(scored),
            headline_scores=scored,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

"""Live news provider — fetches real market news from free sources.

Sources (in priority order):
1. Finnhub News API (FINNHUB_API_KEY in .env — free tier: 60 calls/min)
2. Google News RSS (no key needed, good fallback)
3. Synthetic sample headlines (final fallback)

Each adapter returns a list of dicts with the shape:
    {"headline": str, "source": str, "url": str, "published_at": str, "related": str}
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


class FinnhubNewsProvider:
    """Fetches market news from Finnhub's free API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def get_news(self, symbol: str, limit: int = 15) -> list[dict[str, Any]]:
        import urllib.request
        import json

        url = f"{self.base_url}/company-news?symbol={symbol}&from=" \
              f"{self._days_ago(7)}&to={self._today()}&token={self.api_key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StockAITrader/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return [
                {
                    "headline": item.get("headline", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published_at": datetime.fromtimestamp(
                        item.get("datetime", 0), tz=timezone.utc
                    ).isoformat(),
                    "related": item.get("related", symbol),
                    "summary": item.get("summary", ""),
                    "image": item.get("image", ""),
                }
                for item in data[:limit]
            ]
        except Exception as exc:
            logger.warning("Finnhub news failed for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _days_ago(n: int) -> str:
        from datetime import timedelta
        return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")

    def get_market_news(self, limit: int = 20) -> list[dict[str, Any]]:
        """General market news (not symbol-specific)."""
        import urllib.request
        import json

        url = f"{self.base_url}/news?category=general&token={self.api_key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StockAITrader/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return [
                {
                    "headline": item.get("headline", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published_at": datetime.fromtimestamp(
                        item.get("datetime", 0), tz=timezone.utc
                    ).isoformat(),
                    "related": item.get("related", "MARKET"),
                    "summary": item.get("summary", ""),
                    "image": item.get("image", ""),
                }
                for item in data[:limit]
            ]
        except Exception as exc:
            logger.warning("Finnhub market news failed: %s", exc)
            return []


class GoogleNewsRSSProvider:
    """Fetches news from Google News RSS — no API key needed."""

    def get_news(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        import urllib.request
        import urllib.parse

        query = urllib.parse.quote(f"{symbol} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StockAITrader/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read().decode()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")[:limit]
            results = []
            for item in items:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                source_name = item.findtext("source", "Google News")
                results.append({
                    "headline": title,
                    "source": source_name,
                    "url": link,
                    "published_at": pub_date,
                    "related": symbol,
                    "summary": "",
                    "image": "",
                })
            return results
        except Exception as exc:
            logger.warning("Google News RSS failed for %s: %s", symbol, exc)
            return []


def get_news_for_symbol(symbol: str, limit: int = 15) -> list[dict[str, Any]]:
    """Get news for a symbol, trying providers in order."""
    import os
    symbol = symbol.upper()

    # Try Finnhub first
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    if finnhub_key:
        news = FinnhubNewsProvider(finnhub_key).get_news(symbol, limit)
        if news:
            return news

    # Try Google News RSS
    news = GoogleNewsRSSProvider().get_news(symbol, limit)
    if news:
        return news

    # Fallback to synthetic sample
    from app.data.news_sample import get_sample_headlines
    sample = get_sample_headlines(symbol)
    return [
        {
            "headline": h["text"],
            "source": "Synthetic Sample",
            "url": "",
            "published_at": h.get("published_at", datetime.now(timezone.utc).isoformat()),
            "related": symbol,
            "summary": "",
            "image": "",
        }
        for h in sample[:limit]
    ]

"""Market data provider layer.

`MarketDataProvider` is an abstract adapter so a real-time WebSocket feed
can be swapped in later without touching any AI agent code. `YFinanceProvider`
is the default (free, delayed/EOD data — NOT a tick-level execution feed).
`SyntheticProvider` generates a labeled, reproducible random-walk dataset so
the system remains testable when Yahoo Finance is unreachable (e.g. sandboxed
or offline environments) or rate-limited.
"""
from __future__ import annotations

import abc
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class MarketDataProvider(abc.ABC):
    """Abstract interface every market data source must implement."""

    name: str = "abstract"

    @abc.abstractmethod
    def get_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Return a DataFrame indexed by tz-aware UTC timestamp with OHLCV_COLUMNS."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_latest(self, symbol: str) -> dict:
        """Return the most recent available bar as a flat dict."""
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    """Wraps yfinance. EOD/delayed data only — do not treat as a live execution feed."""

    name = "yfinance"

    def get_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        is_intraday = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h")
        fetch_period = "5d" if (is_intraday and period.lower() in ("1d", "intraday")) else period
        df = ticker.history(period=fetch_period, interval=interval, auto_adjust=False, prepost=is_intraday)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {symbol}")
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume",
        })
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df[OHLCV_COLUMNS]

    def get_latest(self, symbol: str) -> dict:
        df = self.get_historical(symbol, period="5d", interval="1d")
        row = df.iloc[-1]
        return {"symbol": symbol, "timestamp": df.index[-1].isoformat(), **row.to_dict()}


class SyntheticProvider(MarketDataProvider):
    """Deterministic geometric-brownian-motion generator.

    Clearly labeled synthetic data — used only as a fallback for offline
    testing/demo, per project rule: no unlabeled fake market data.
    """

    name = "synthetic"

    _SEEDS = {  # stable per-symbol seed so repeated calls are reproducible
        "AAPL": 1, "MSFT": 2, "NVDA": 3, "AMZN": 4, "GOOGL": 5, "TSLA": 6,
    }

    def _n_bars(self, period: str) -> int:
        unit = period[-1]
        n = int(period[:-1]) if period[:-1].isdigit() else 2
        return {"d": n, "mo": n * 21, "y": n * 252}.get(unit, 504)

    def get_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        if period.lower() in ("1d", "intraday") or interval in ("1m", "2m", "5m", "15m"):
            n = 78
            idx = pd.date_range(end=end, periods=n, freq="5min", tz="UTC")
            mu, sigma = 0.00005, 0.003  # intraday 5m vol
        else:
            n = max(self._n_bars(period), 60)
            idx = pd.bdate_range(end=end, periods=n, tz="UTC")
            mu, sigma = 0.0004, 0.018  # daily drift / vol, roughly equity-like

        seed = self._SEEDS.get(symbol.upper(), abs(hash(symbol)) % 1000)
        rng = np.random.default_rng(seed)

        rets = rng.normal(mu, sigma, n)
        price = 100 * np.exp(np.cumsum(rets))

        close = price
        open_ = close * (1 + rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        volume = rng.integers(2_000_000, 15_000_000, n)

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "adj_close": close, "volume": volume,
        }, index=idx)
        df.index.name = "timestamp"
        return df

    def get_latest(self, symbol: str) -> dict:
        df = self.get_historical(symbol, period="5d", interval="1d")
        row = df.iloc[-1]
        return {"symbol": symbol, "timestamp": df.index[-1].isoformat(), **row.to_dict(), "synthetic": True}


def get_provider(prefer: str = "yfinance") -> MarketDataProvider:
    """Factory. Returns yfinance provider; caller should fall back to
    SyntheticProvider on failure via `get_historical_with_fallback` below.
    """
    if prefer == "synthetic":
        return SyntheticProvider()
    return YFinanceProvider()


def get_historical_with_fallback(symbol: str, period: str = "2y", interval: str = "1d") -> tuple[pd.DataFrame, str]:
    """Try the live provider; fall back to synthetic on any failure.

    Returns (dataframe, source_name) so downstream consumers/UI can display
    clearly whether the data is real or synthetic.
    """
    from app.config import settings
    try:
        df = YFinanceProvider().get_historical(symbol, period, interval)
        return df, "yfinance"
    except Exception as exc:  # noqa: BLE001 — deliberately broad: network/API can fail many ways
        logger.warning("yfinance failed for %s (%s); falling back to synthetic=%s",
                        symbol, exc, settings.use_synthetic_on_failure)
        if not settings.use_synthetic_on_failure:
            raise
        return SyntheticProvider().get_historical(symbol, period, interval), "synthetic"

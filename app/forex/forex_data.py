"""Forex Market Data & Analytics Provider.

Provides:
- Supported Forex pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD
- Pair normalization, base/quote extraction, and yfinance ticker mapping (e.g. EURUSD=X)
- Pip calculations (0.01 for JPY pairs, 0.0001 for standard G10 pairs)
- Market sessions (Sydney, Tokyo, London, New York)
- ATR, realized volatility, spread modeling, and trend metrics
- Deterministic synthetic generator for offline/sandboxed environments
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FOREX_PAIRS: list[str] = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "USD/CHF",
    "NZD/USD",
]

_YF_TICKER_MAP: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
}

_TYPICAL_SPREAD_PIPS: dict[str, float] = {
    "EUR/USD": 1.2,
    "GBP/USD": 1.6,
    "USD/JPY": 1.4,
    "AUD/USD": 1.8,
    "USD/CAD": 2.0,
    "USD/CHF": 1.9,
    "NZD/USD": 2.2,
}

_SYNTHETIC_BASE_RATES: dict[str, float] = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2720,
    "USD/JPY": 154.60,
    "AUD/USD": 0.6540,
    "USD/CAD": 1.3820,
    "USD/CHF": 0.8850,
    "NZD/USD": 0.5980,
}


def normalize_forex_pair(pair: str) -> str:
    """Normalize pair representation (e.g., 'eurusd', 'EURUSD', 'EUR-USD', 'EUR/USD')."""
    clean = pair.upper().strip().replace("-", "/").replace("_", "")
    if "/" not in clean and len(clean) == 6:
        clean = f"{clean[:3]}/{clean[3:]}"
    return clean


def is_supported_pair(pair: str) -> bool:
    norm = normalize_forex_pair(pair)
    return norm in SUPPORTED_FOREX_PAIRS


def get_pip_size(pair: str) -> float:
    """Return pip unit size: 0.01 for JPY pairs, 0.0001 for standard currency pairs."""
    norm = normalize_forex_pair(pair)
    return 0.01 if "JPY" in norm else 0.0001


def calculate_pips(entry_price: float, exit_price: float, pair: str) -> float:
    """Calculate pip difference between entry and exit prices."""
    pip_size = get_pip_size(pair)
    return round((exit_price - entry_price) / pip_size, 2)


def get_active_sessions(utc_time: datetime | None = None) -> list[str]:
    """Identify which global Forex trading sessions are active at the given UTC time."""
    now = utc_time or datetime.now(timezone.utc)
    hour = now.hour

    sessions = []
    # Sydney: 22:00 - 07:00 UTC
    if hour >= 22 or hour < 7:
        sessions.append("Sydney")
    # Tokyo: 00:00 - 09:00 UTC
    if 0 <= hour < 9:
        sessions.append("Tokyo")
    # London: 08:00 - 17:00 UTC
    if 8 <= hour < 17:
        sessions.append("London")
    # New York: 13:00 - 22:00 UTC
    if 13 <= hour < 22:
        sessions.append("New York")

    return sessions or ["Asian Bridge"]


class ForexQuote(BaseModel):
    pair: str
    bid: float
    ask: float
    mid: float
    spread_pips: float
    change_24h_pct: float
    change_pips: float
    high_24h: float
    low_24h: float
    atr_14: float
    annualized_volatility: float
    active_sessions: list[str]
    is_synthetic: bool = False
    timestamp: str
    data_quality: str = "REALTIME_DELAYED"


class ForexDataProvider:
    """Retrieves Forex historical and real-time quotes with offline synthetic fallback."""

    def __init__(self, use_synthetic: bool = False):
        self.use_synthetic = use_synthetic

    def get_quote(self, pair: str) -> ForexQuote:
        norm = normalize_forex_pair(pair)
        if not is_supported_pair(norm):
            raise ValueError(f"Pair '{pair}' is not supported. Supported pairs: {', '.join(SUPPORTED_FOREX_PAIRS)}")

        pip_size = get_pip_size(norm)
        spread_pips = _TYPICAL_SPREAD_PIPS.get(norm, 1.5)

        if not self.use_synthetic:
            try:
                import yfinance as yf
                ticker_sym = _YF_TICKER_MAP.get(norm, f"{norm.replace('/', '')}=X")
                ticker = yf.Ticker(ticker_sym)
                df = ticker.history(period="5d", interval="1d", auto_adjust=False)
                if not df.empty and len(df) >= 2:
                    last_close = float(df["Close"].iloc[-1])
                    prev_close = float(df["Close"].iloc[-2])
                    high_24h = float(df["High"].iloc[-1])
                    low_24h = float(df["Low"].iloc[-1])

                    # Half-spread
                    half_spread = (spread_pips * pip_size) / 2.0
                    bid = round(last_close - half_spread, 5 if "JPY" not in norm else 3)
                    ask = round(last_close + half_spread, 5 if "JPY" not in norm else 3)
                    change_pct = round((last_close - prev_close) / max(1e-6, prev_close), 4)
                    change_pips = calculate_pips(prev_close, last_close, norm)

                    # Compute ATR
                    df_atr = self.get_historical(norm, period="1mo", interval="1d")
                    atr_val = self._compute_atr(df_atr)
                    vol_ann = self._compute_volatility(df_atr)

                    return ForexQuote(
                        pair=norm,
                        bid=bid,
                        ask=ask,
                        mid=round(last_close, 5 if "JPY" not in norm else 3),
                        spread_pips=spread_pips,
                        change_24h_pct=change_pct,
                        change_pips=change_pips,
                        high_24h=round(high_24h, 5 if "JPY" not in norm else 3),
                        low_24h=round(low_24h, 5 if "JPY" not in norm else 3),
                        atr_14=round(atr_val, 5 if "JPY" not in norm else 3),
                        annualized_volatility=round(vol_ann, 4),
                        active_sessions=get_active_sessions(),
                        is_synthetic=False,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data_quality="YFINANCE_EOD_DELAYED",
                    )
            except Exception as exc:
                logger.debug("yfinance quote lookup failed for %s, falling back to synthetic: %s", norm, exc)

        # Fallback to deterministic synthetic quote
        return self._get_synthetic_quote(norm)

    def get_historical(self, pair: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        norm = normalize_forex_pair(pair)
        if not is_supported_pair(norm):
            raise ValueError(f"Pair '{pair}' is not supported. Supported pairs: {', '.join(SUPPORTED_FOREX_PAIRS)}")

        if not self.use_synthetic:
            try:
                import yfinance as yf
                ticker_sym = _YF_TICKER_MAP.get(norm, f"{norm.replace('/', '')}=X")
                ticker = yf.Ticker(ticker_sym)
                df = ticker.history(period=period, interval=interval, auto_adjust=False)
                if not df.empty and len(df) >= 5:
                    df = df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low", "Close": "close",
                        "Adj Close": "adj_close", "Volume": "volume",
                    })
                    if "adj_close" not in df.columns:
                        df["adj_close"] = df["close"]
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.index.name = "timestamp"
                    return df[["open", "high", "low", "close", "adj_close", "volume"]]
            except Exception as exc:
                logger.debug("yfinance historical fetch failed for %s, using synthetic: %s", norm, exc)

        return self._generate_synthetic_historical(norm, period=period, interval=interval)

    def _get_synthetic_quote(self, pair: str) -> ForexQuote:
        base_rate = _SYNTHETIC_BASE_RATES.get(pair, 1.0850)
        pip_size = get_pip_size(pair)
        spread_pips = _TYPICAL_SPREAD_PIPS.get(pair, 1.5)
        half_spread = (spread_pips * pip_size) / 2.0

        bid = round(base_rate - half_spread, 5 if "JPY" not in pair else 3)
        ask = round(base_rate + half_spread, 5 if "JPY" not in pair else 3)
        prev_close = base_rate * 0.9985
        change_pct = round((base_rate - prev_close) / prev_close, 4)
        change_pips = calculate_pips(prev_close, base_rate, pair)

        atr_val = 0.0065 if "JPY" not in pair else 0.85
        vol_ann = 0.075

        return ForexQuote(
            pair=pair,
            bid=bid,
            ask=ask,
            mid=base_rate,
            spread_pips=spread_pips,
            change_24h_pct=change_pct,
            change_pips=change_pips,
            high_24h=round(base_rate * 1.004, 5 if "JPY" not in pair else 3),
            low_24h=round(base_rate * 0.996, 5 if "JPY" not in pair else 3),
            atr_14=atr_val,
            annualized_volatility=vol_ann,
            active_sessions=get_active_sessions(),
            is_synthetic=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_quality="DETERMINISTIC_SYNTHETIC",
        )

    def _generate_synthetic_historical(self, pair: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        n_bars = 90
        if period == "1mo":
            n_bars = 30
        elif period == "1y":
            n_bars = 252

        end = datetime.now(timezone.utc)
        idx = pd.bdate_range(end=end, periods=n_bars, tz="UTC")
        base_price = _SYNTHETIC_BASE_RATES.get(pair, 1.0850)

        seed = abs(hash(pair)) % 10000
        rng = np.random.default_rng(seed)

        # Forex returns: lower vol than stocks (e.g. 7-10% ann vol -> ~0.5% daily)
        daily_vol = 0.0055
        returns = rng.normal(0.0001, daily_vol, size=n_bars)
        price_series = base_price * np.exp(np.cumsum(returns) - np.sum(returns))

        highs = price_series * (1 + rng.uniform(0.001, 0.004, size=n_bars))
        lows = price_series * (1 - rng.uniform(0.001, 0.004, size=n_bars))
        opens = price_series * (1 + rng.normal(0, 0.0015, size=n_bars))
        volumes = rng.integers(10000, 500000, size=n_bars)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": price_series,
                "adj_close": price_series,
                "volume": volumes,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        return df

    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        if len(df) < 2:
            return 0.0060
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(window=min(window, len(tr)), min_periods=1).mean()
        return float(atr_series.iloc[-1])

    def _compute_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.075
        rets = df["close"].pct_change().dropna()
        if rets.empty:
            return 0.075
        return float(rets.std() * math.sqrt(252))


_provider_instance: ForexDataProvider | None = None


def get_forex_provider() -> ForexDataProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = ForexDataProvider()
    return _provider_instance

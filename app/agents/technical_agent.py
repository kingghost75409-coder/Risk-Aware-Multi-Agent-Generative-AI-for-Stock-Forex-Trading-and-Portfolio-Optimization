"""Technical Analysis Agent.

All indicators are computed deterministically in pandas/numpy. No LLM is
ever used to calculate a numeric indicator.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TechnicalSignal(BaseModel):
    symbol: str
    signal: str  # BUY / SELL / HOLD
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: dict
    reasoning: str
    timestamp: str


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


class TechnicalAgent:
    """Computes indicators and derives a deterministic, rule-based signal."""

    def analyze(self, df: pd.DataFrame, symbol: str) -> TechnicalSignal:
        close = df["adj_close"]
        volume = df["volume"]

        rsi = _rsi(close)
        macd_line, macd_signal, macd_hist = _macd(close)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        bb_mid, bb_upper, bb_lower, pct_b = _bollinger(close)
        momentum_10 = close.pct_change(10)
        vol_sma = volume.rolling(20).mean()
        volume_ratio = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)
        realized_vol = close.pct_change().rolling(20).std() * np.sqrt(252)

        last = -1
        indicators = {
            "rsi_14": round(float(rsi.iloc[last]), 2),
            "macd_line": round(float(macd_line.iloc[last]), 4),
            "macd_signal": round(float(macd_signal.iloc[last]), 4),
            "macd_histogram": round(float(macd_hist.iloc[last]), 4),
            "sma_20": round(float(sma20.iloc[last]), 2) if not np.isnan(sma20.iloc[last]) else None,
            "sma_50": round(float(sma50.iloc[last]), 2) if not np.isnan(sma50.iloc[last]) else None,
            "ema_12": round(float(ema12.iloc[last]), 2),
            "ema_26": round(float(ema26.iloc[last]), 2),
            "bollinger_pct_b": round(float(pct_b.iloc[last]), 3) if not np.isnan(pct_b.iloc[last]) else None,
            "momentum_10d_pct": round(float(momentum_10.iloc[last]) * 100, 2) if not np.isnan(momentum_10.iloc[last]) else None,
            "volume_ratio_vs_20d_avg": round(float(volume_ratio.iloc[last]), 2),
            "realized_volatility_annualized": round(float(realized_vol.iloc[last]), 3) if not np.isnan(realized_vol.iloc[last]) else None,
            "trend_strength": "UP" if (not np.isnan(sma20.iloc[last]) and not np.isnan(sma50.iloc[last])
                                        and sma20.iloc[last] > sma50.iloc[last]) else "DOWN",
        }

        signal, confidence, reasoning = self._vote(indicators, close, sma20, sma50)

        return TechnicalSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _vote(ind: dict, close: pd.Series, sma20: pd.Series, sma50: pd.Series) -> tuple[str, float, str]:
        """Simple, transparent, rule-based scoring across independent signals.

        Each sub-rule casts a vote in [-1, +1]; the average maps to
        confidence, and the sign (with a neutral band) maps to BUY/SELL/HOLD.
        This keeps the technical signal fully deterministic and auditable.
        """
        votes = []
        notes = []

        rsi = ind["rsi_14"]
        if rsi < 30:
            votes.append(1.0); notes.append(f"RSI {rsi} oversold (<30)")
        elif rsi > 70:
            votes.append(-1.0); notes.append(f"RSI {rsi} overbought (>70)")
        else:
            votes.append(0.0); notes.append(f"RSI {rsi} neutral")

        if ind["macd_histogram"] > 0:
            votes.append(0.7); notes.append("MACD histogram positive (bullish momentum)")
        else:
            votes.append(-0.7); notes.append("MACD histogram negative (bearish momentum)")

        if ind["sma_20"] and ind["sma_50"]:
            if ind["sma_20"] > ind["sma_50"]:
                votes.append(0.6); notes.append("SMA20 above SMA50 (uptrend)")
            else:
                votes.append(-0.6); notes.append("SMA20 below SMA50 (downtrend)")

        pct_b = ind["bollinger_pct_b"]
        if pct_b is not None:
            if pct_b < 0.05:
                votes.append(0.5); notes.append("price near lower Bollinger band")
            elif pct_b > 0.95:
                votes.append(-0.5); notes.append("price near upper Bollinger band")
            else:
                votes.append(0.0)

        mom = ind["momentum_10d_pct"]
        if mom is not None:
            votes.append(np.clip(mom / 10, -0.5, 0.5))
            notes.append(f"10d momentum {mom}%")

        score = float(np.mean(votes))
        confidence = float(min(0.95, 0.5 + abs(score) * 0.5))

        if score > 0.15:
            signal = "BUY"
        elif score < -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"

        reasoning = f"Composite technical score={score:.2f}. " + "; ".join(notes)
        return signal, round(confidence, 3), reasoning

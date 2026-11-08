"""Market Regime Detection Agent.

Implementation choice: rule-based classification over rolling trend +
realized volatility, rather than a Hidden Markov Model or unsupervised
clustering.

Reason (documented per project requirement): with ~1-2 years of daily data
per symbol, an HMM or KMeans regime model has too few effective volatility
"episodes" to fit a stable, non-overfit model, and its regime labels would
be unstable across reruns — poor for reproducibility. A trend + volatility
quantile rule is fully deterministic, easy to audit, and is a standard,
defensible baseline regime classifier in the trading-systems literature.
The module is written so an HMM (via hmmlearn/statsmodels) or KMeans
implementation can be swapped in behind the same `analyze()` interface.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class RegimeResult(BaseModel):
    symbol: str
    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    trend_component: str
    volatility_component: str
    details: dict
    timestamp: str


class MarketRegimeAgent:
    def analyze(self, df: pd.DataFrame, symbol: str) -> RegimeResult:
        close = df["adj_close"]
        returns = close.pct_change().dropna()

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        trend_slope = (sma20.iloc[-1] / sma20.iloc[-10] - 1) if len(sma20.dropna()) > 10 else 0.0

        realized_vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_history = returns.rolling(20).std().dropna() * np.sqrt(252)
        vol_percentile = float((vol_history < realized_vol_20).mean()) if len(vol_history) else 0.5

        is_uptrend = bool(sma20.iloc[-1] > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else trend_slope > 0
        trend_component = "BULL" if is_uptrend else "BEAR"
        if abs(trend_slope) < 0.005:
            trend_component = "SIDEWAYS"

        volatility_component = "HIGH_VOLATILITY" if vol_percentile > 0.66 else (
            "LOW_VOLATILITY" if vol_percentile < 0.33 else "NORMAL_VOLATILITY")

        if trend_component == "SIDEWAYS":
            regime = f"SIDEWAYS_{volatility_component}"
        else:
            regime = f"{volatility_component}_{trend_component}"

        # confidence: how decisively trend & vol sit away from their "ambiguous" midpoints
        trend_conf = min(1.0, abs(trend_slope) * 40)
        vol_conf = abs(vol_percentile - 0.5) * 2
        confidence = float(np.clip(0.4 + 0.3 * trend_conf + 0.3 * vol_conf, 0.4, 0.97))

        return RegimeResult(
            symbol=symbol,
            regime=regime,
            confidence=round(confidence, 3),
            trend_component=trend_component,
            volatility_component=volatility_component,
            details={
                "sma20_vs_sma50_slope_10d": round(float(trend_slope), 4),
                "realized_vol_annualized": round(float(realized_vol_20), 3) if not np.isnan(realized_vol_20) else None,
                "vol_percentile_vs_own_history": round(vol_percentile, 3),
                "method": "rule_based_trend_volatility (see module docstring for rationale)",
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

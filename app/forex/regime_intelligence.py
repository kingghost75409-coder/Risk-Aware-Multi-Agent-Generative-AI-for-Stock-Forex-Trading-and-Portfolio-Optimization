"""Forex Market Regime Intelligence AI.

Identifies the current market condition for each supported Forex pair:
- Trending (Bullish or Bearish)
- Ranging
- High-Volatility
- Low-Volatility
- Crisis / Extreme-Market Condition

Calculates supporting indicators: ADX, Bollinger Band width, ATR-to-price ratio,
20/50 SMA slope, and realized volatility.

Provides:
- Regime classification and confidence scoring
- Transition & change detection with DB persistence
- Strategy signal confidence calibration (deterministic only, no LLM inflation)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.db.models import get_session, ForexRegimeLogRecord
from app.forex.forex_data import ForexDataProvider, get_forex_provider, normalize_forex_pair
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ForexRegimeType(str, Enum):
    TRENDING_BULLISH = "Trending Bullish"
    TRENDING_BEARISH = "Trending Bearish"
    RANGING = "Ranging"
    HIGH_VOLATILITY = "High-Volatility"
    LOW_VOLATILITY = "Low-Volatility"
    CRISIS_EXTREME = "Crisis / Extreme-Market Condition"


class VolatilityLevel(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ForexRegimeAssessment(BaseModel):
    pair: str
    regime: ForexRegimeType
    regime_confidence: float = Field(..., ge=0.0, le=1.0)
    trend_direction: TrendDirection
    volatility_level: VolatilityLevel
    supporting_indicators: dict[str, float]
    data_quality: str
    timestamp: str
    explanation: str
    strategy_suitability: dict[str, str]


class RegimeChangeAlert(BaseModel):
    pair: str
    previous_regime: str
    new_regime: str
    timestamp: str
    reason: str
    is_significant: bool = True


class ForexMarketRegimeIntelligence:
    """Evaluates quantitative indicators across currency bars to infer the active Forex market regime."""

    def __init__(self, data_provider: ForexDataProvider | None = None):
        self.provider = data_provider or get_forex_provider()
        self._last_regimes: dict[str, str] = {}

    def analyze_regime(self, pair: str, df: pd.DataFrame | None = None) -> ForexRegimeAssessment:
        norm = normalize_forex_pair(pair)
        if df is None or len(df) < 15:
            df = self.provider.get_historical(norm, period="3mo", interval="1d")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 1. Moving Averages
        sma_20 = float(close.rolling(20, min_periods=5).mean().iloc[-1])
        sma_50 = float(close.rolling(50, min_periods=10).mean().iloc[-1])
        last_price = float(close.iloc[-1])

        # 2. Realized Volatility (20-day annualized)
        rets = close.pct_change().dropna()
        vol_20d = float(rets.tail(20).std() * math.sqrt(252)) if len(rets) >= 5 else 0.08
        if math.isnan(vol_20d) or vol_20d <= 0.0:
            vol_20d = 0.08

        # 3. Bollinger Band Width
        bb_std = float(close.rolling(20, min_periods=5).std().iloc[-1])
        bb_upper = sma_20 + (2.0 * bb_std)
        bb_lower = sma_20 - (2.0 * bb_std)
        bb_width = float((bb_upper - bb_lower) / max(1e-5, sma_20))

        # 4. ADX estimation (Average Directional Index)
        adx, p_di, m_di = self._compute_adx(high, low, close, window=14)

        # 5. ATR to Price Ratio
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14, min_periods=2).mean().iloc[-1])
        atr_ratio = atr_14 / max(1e-5, last_price)

        # Indicators dictionary
        indicators = {
            "last_price": round(last_price, 5 if "JPY" not in norm else 3),
            "sma_20": round(sma_20, 5 if "JPY" not in norm else 3),
            "sma_50": round(sma_50, 5 if "JPY" not in norm else 3),
            "adx_14": round(adx, 2),
            "plus_di": round(p_di, 2),
            "minus_di": round(m_di, 2),
            "bollinger_band_width": round(bb_width, 4),
            "annualized_volatility": round(vol_20d, 4),
            "atr_14": round(atr_14, 5 if "JPY" not in norm else 3),
            "atr_price_ratio": round(atr_ratio, 4),
        }

        # Determine Volatility Level
        if vol_20d >= 0.22 or atr_ratio >= 0.020:
            vol_level = VolatilityLevel.EXTREME
        elif vol_20d >= 0.13 or atr_ratio >= 0.012:
            vol_level = VolatilityLevel.HIGH
        elif vol_20d <= 0.055 and bb_width <= 0.012:
            vol_level = VolatilityLevel.LOW
        else:
            vol_level = VolatilityLevel.NORMAL

        # Determine Trend Direction
        if last_price > sma_20 > sma_50 and p_di > m_di:
            trend_dir = TrendDirection.BULLISH
        elif last_price < sma_20 < sma_50 and m_di > p_di:
            trend_dir = TrendDirection.BEARISH
        else:
            trend_dir = TrendDirection.NEUTRAL

        # Determine Market Regime
        if vol_level == VolatilityLevel.EXTREME:
            regime = ForexRegimeType.CRISIS_EXTREME
            confidence = min(0.95, 0.65 + (vol_20d * 1.2))
            explanation = (
                f"Severe volatility spike detected (annualized vol: {vol_20d:.1%}, ATR ratio: {atr_ratio:.2%}). "
                f"Market is exhibiting crisis or extreme liquidity dislocation characteristics."
            )
        elif vol_level == VolatilityLevel.HIGH and adx < 22:
            regime = ForexRegimeType.HIGH_VOLATILITY
            confidence = min(0.90, 0.60 + (vol_20d * 1.5))
            explanation = (
                f"High volatility ({vol_20d:.1%} ann.) without directional persistence (ADX: {adx:.1f}). "
                f"Expect erratic intraday swings and widening bid/ask spreads."
            )
        elif adx >= 25 and trend_dir == TrendDirection.BULLISH:
            regime = ForexRegimeType.TRENDING_BULLISH
            confidence = min(0.92, 0.50 + (adx / 60.0))
            explanation = (
                f"Strong directional upward momentum confirmed by ADX ({adx:.1f}) and Bullish moving average alignment "
                f"(Price > SMA20 > SMA50)."
            )
        elif adx >= 25 and trend_dir == TrendDirection.BEARISH:
            regime = ForexRegimeType.TRENDING_BEARISH
            confidence = min(0.92, 0.50 + (adx / 60.0))
            explanation = (
                f"Strong directional downward momentum confirmed by ADX ({adx:.1f}) and Bearish moving average alignment "
                f"(Price < SMA20 < SMA50)."
            )
        elif vol_level == VolatilityLevel.LOW:
            regime = ForexRegimeType.LOW_VOLATILITY
            confidence = min(0.88, 0.65 + (0.015 - bb_width) * 10)
            explanation = (
                f"Volatility compression in progress (BB Width: {bb_width:.4f}, Ann. Vol: {vol_20d:.1%}). "
                f"Price is consolidating inside a tight band; breakout potential is accumulating."
            )
        else:
            regime = ForexRegimeType.RANGING
            confidence = min(0.85, 0.55 + max(0.0, (25 - adx) / 50.0))
            explanation = (
                f"Range-bound consolidation with weak directional conviction (ADX: {adx:.1f}). "
                f"Mean-reverting oscillators and support/resistance boundaries dominate price action."
            )

        strategy_suitability = {
            "trend_following": "FAVORABLE" if "Trending" in regime.value else "UNFAVORABLE",
            "mean_reversion": "FAVORABLE" if regime in (ForexRegimeType.RANGING, ForexRegimeType.LOW_VOLATILITY) else "RISKY",
            "breakout": "FAVORABLE" if regime == ForexRegimeType.LOW_VOLATILITY else "NEUTRAL",
            "event_arbitrage": "UNFAVORABLE" if regime == ForexRegimeType.CRISIS_EXTREME else "NEUTRAL",
        }

        assessment = ForexRegimeAssessment(
            pair=norm,
            regime=regime,
            regime_confidence=round(confidence, 2),
            trend_direction=trend_dir,
            volatility_level=vol_level,
            supporting_indicators=indicators,
            data_quality="CALCULATED_OHLCV",
            timestamp=datetime.now(timezone.utc).isoformat(),
            explanation=explanation,
            strategy_suitability=strategy_suitability,
        )

        # Check for regime change
        prev = self._last_regimes.get(norm)
        if prev and prev != regime.value:
            self._log_regime_change(norm, prev, regime.value, explanation)
        self._last_regimes[norm] = regime.value

        return assessment

    def calibrate_signal_confidence(
        self,
        base_signal: str,
        base_confidence: float,
        strategy_type: str,
        regime_assessment: ForexRegimeAssessment,
    ) -> tuple[float, str, bool]:
        """Calibrate signal confidence based on detected regime.
        
        Returns: (calibrated_confidence, rationale, is_validated)
        Rule: NEVER allows LLM or heuristic to arbitrarily inflate confidence.
        Can only penalize or confirm bounded confidence.
        """
        regime = regime_assessment.regime
        calibrated = base_confidence
        rationale = f"Strategy '{strategy_type}' evaluated under regime '{regime.value}'."
        is_validated = True

        if regime == ForexRegimeType.CRISIS_EXTREME:
            calibrated = min(base_confidence * 0.40, 0.35)
            rationale = "CRISIS regime enforces severe 60% confidence attenuation to protect capital."
        elif strategy_type in ("trend_following", "momentum"):
            if "Trending" in regime.value:
                calibrated = min(base_confidence * 1.0, 0.88)
                rationale = f"Signal is well-aligned with active {regime.value} regime."
            elif regime == ForexRegimeType.RANGING:
                calibrated = base_confidence * 0.70
                rationale = "Momentum signal penalized due to choppy Ranging market conditions (whipsaw risk)."
            elif regime == ForexRegimeType.HIGH_VOLATILITY:
                calibrated = base_confidence * 0.60
                rationale = "High volatility introduces adverse slippage and erratic breakout failures."
        elif strategy_type in ("mean_reversion", "range_bound"):
            if regime == ForexRegimeType.RANGING:
                calibrated = min(base_confidence * 1.0, 0.85)
                rationale = "Mean-reversion strategy aligns with active Ranging regime."
            elif "Trending" in regime.value:
                calibrated = base_confidence * 0.65
                rationale = "Mean-reversion penalized against persistent directional trend."
        else:
            # Unvalidated strategy
            is_validated = False
            calibrated = base_confidence * 0.80
            rationale = f"Strategy '{strategy_type}' has no verified empirical calibration matrix for {regime.value}; flagged as unvalidated."

        return round(calibrated, 2), rationale, is_validated

    def _compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> tuple[float, float, float]:
        if len(close) < window + 2:
            return 22.0, 20.0, 18.0

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=window, min_periods=window // 2).mean()
        plus_di = 100.0 * (pd.Series(plus_dm, index=close.index).rolling(window, min_periods=window // 2).mean() / atr)
        minus_di = 100.0 * (pd.Series(minus_dm, index=close.index).rolling(window, min_periods=window // 2).mean() / atr)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_series = dx.rolling(window=window, min_periods=window // 2).mean()

        adx_val = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 22.0
        p_val = float(plus_di.dropna().iloc[-1]) if not plus_di.dropna().empty else 20.0
        m_val = float(minus_di.dropna().iloc[-1]) if not minus_di.dropna().empty else 18.0

        return adx_val, p_val, m_val

    def _log_regime_change(self, pair: str, prev_regime: str, new_regime: str, reason: str) -> None:
        try:
            session = get_session()
            log_entry = ForexRegimeLogRecord(
                pair=pair,
                previous_regime=prev_regime,
                new_regime=new_regime,
                reason=reason,
                details_json="{}",
                timestamp=datetime.now(timezone.utc),
            )
            session.add(log_entry)
            session.commit()
            session.close()
            logger.info("Forex Regime Change detected for %s: %s -> %s", pair, prev_regime, new_regime)
        except Exception as exc:
            logger.debug("Failed to persist ForexRegimeLogRecord: %s", exc)


_regime_instance: ForexMarketRegimeIntelligence | None = None


def get_regime_intelligence() -> ForexMarketRegimeIntelligence:
    global _regime_instance
    if _regime_instance is None:
        _regime_instance = ForexMarketRegimeIntelligence()
    return _regime_instance

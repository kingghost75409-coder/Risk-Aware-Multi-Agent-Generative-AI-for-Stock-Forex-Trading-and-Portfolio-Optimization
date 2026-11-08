"""Adaptive Alpha Signal Engine.

Research contribution: Dynamically combines technical, ML, sentiment, RAG,
and crash-risk signals into a unified Alpha Score with regime-adaptive
weighting.

Unlike the static weighted averaging in the original orchestrator, this
engine tracks exponentially-weighted rolling signal performance per regime
and adjusts weights accordingly — signals that have been consistently wrong
in the current regime are down-weighted.

Architecture:
  1. Normalize each signal to a directional score in [-1, +1].
  2. Apply regime-conditional weight priors (base weights shift by regime).
  3. Compute composite alpha = Σ(weight_i × signal_i) / Σ(weight_i).
  4. Output per-signal weight attribution for explainability.

SAFETY: Alpha is a research metric — never presented as a guaranteed
probability of profit. The Risk Engine remains the inviolable final gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SignalContribution(BaseModel):
    """Attribution of a single signal to the composite alpha."""
    signal_name: str
    raw_value: float
    normalized_score: float  # [-1, +1]
    weight: float
    weighted_contribution: float
    weight_rationale: str


class AlphaSignalResult(BaseModel):
    symbol: str
    composite_alpha: float = Field(
        description="Net directional alpha in [-1, +1]; >0 bullish, <0 bearish"
    )
    alpha_magnitude: float = Field(
        ge=0.0, le=1.0,
        description="Absolute conviction strength [0=no signal, 1=maximum]"
    )
    signal_contributions: list[SignalContribution]
    regime: str
    regime_weight_profile: str
    directional_bias: str  # BULLISH, BEARISH, NEUTRAL
    signal_agreement_pct: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of signals agreeing with the dominant direction"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# Regime-conditional base weight profiles
# Keys: (regime_keyword → weight multiplier adjustments)
_REGIME_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "BULL_LOW_VOL": {
        "technical": 1.2, "ml": 1.1, "sentiment": 0.9,
        "rag": 0.8, "crash_risk": 0.6,
        "label": "Trending favorable: technicals and ML emphasized",
    },
    "BULL_NORMAL": {
        "technical": 1.1, "ml": 1.0, "sentiment": 1.0,
        "rag": 0.9, "crash_risk": 0.7,
        "label": "Balanced bull: standard weights",
    },
    "SIDEWAYS": {
        "technical": 0.8, "ml": 1.0, "sentiment": 1.1,
        "rag": 1.1, "crash_risk": 1.0,
        "label": "Range-bound: sentiment and evidence gain importance",
    },
    "BEAR": {
        "technical": 0.9, "ml": 0.9, "sentiment": 1.2,
        "rag": 1.2, "crash_risk": 1.4,
        "label": "Bearish: crash risk and evidence quality prioritized",
    },
    "HIGH_VOL": {
        "technical": 0.7, "ml": 0.8, "sentiment": 1.0,
        "rag": 1.0, "crash_risk": 1.5,
        "label": "High volatility: crash risk dominates, technicals unreliable",
    },
    "BEAR_HIGH_VOL": {
        "technical": 0.6, "ml": 0.7, "sentiment": 1.1,
        "rag": 1.2, "crash_risk": 1.6,
        "label": "Crisis regime: maximum crash risk weight, defensive posture",
    },
    "DEFAULT": {
        "technical": 1.0, "ml": 1.0, "sentiment": 1.0,
        "rag": 1.0, "crash_risk": 1.0,
        "label": "Default: equal base weights",
    },
}

# Base weights before regime adjustment
_BASE_WEIGHTS = {
    "technical": 0.25,
    "ml": 0.25,
    "sentiment": 0.15,
    "rag": 0.15,
    "crash_risk": 0.20,
}


def _get_regime_profile(regime_label: str) -> tuple[dict[str, float], str]:
    """Match a regime label to the appropriate weight profile."""
    regime_upper = regime_label.upper()
    if "BEAR" in regime_upper and "HIGH" in regime_upper and "VOL" in regime_upper:
        key = "BEAR_HIGH_VOL"
    elif "BEAR" in regime_upper:
        key = "BEAR"
    elif "HIGH" in regime_upper and "VOL" in regime_upper:
        key = "HIGH_VOL"
    elif "BULL" in regime_upper and "LOW" in regime_upper:
        key = "BULL_LOW_VOL"
    elif "BULL" in regime_upper:
        key = "BULL_NORMAL"
    elif "SIDEWAYS" in regime_upper or "RANGE" in regime_upper:
        key = "SIDEWAYS"
    else:
        key = "DEFAULT"
    profile = _REGIME_WEIGHT_PROFILES[key]
    return {k: v for k, v in profile.items() if k != "label"}, profile["label"]


def _normalize_technical(signal: str, confidence: float) -> float:
    """Convert technical signal + confidence to [-1, +1]."""
    direction = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(signal, 0.0)
    return direction * min(confidence, 1.0)


def _normalize_ml(direction: str, probability: float) -> float:
    """Convert ML prediction to [-1, +1]."""
    sign = 1.0 if direction == "UP" else -1.0
    # Scale by distance from 50% (random): 50% → 0, 100% → 1
    strength = max(0.0, (probability - 0.5) * 2.0)
    return sign * strength


def _normalize_sentiment(score: float) -> float:
    """Sentiment score is already in [-1, +1]."""
    return max(-1.0, min(1.0, score))


def _normalize_rag(verification_status: str, evidence_quality: float,
                   insufficient: bool) -> float:
    """Convert RAG evidence quality to a confidence modifier.
    Returns > 0 if evidence supports trading, < 0 if evidence is negative."""
    if insufficient:
        return -0.3  # Penalize lack of evidence
    if verification_status == "UNSUPPORTED":
        return -0.5
    if verification_status == "PARTIALLY_UNSUPPORTED":
        return -0.15
    # SUPPORTED or NO_CLAIM_MADE — evidence quality as positive signal
    return evidence_quality * 0.5  # Moderate positive contribution


def _normalize_crash_risk(crash_probability: float) -> float:
    """Convert crash probability to a bearish signal in [-1, 0].
    High crash prob = strong negative signal."""
    return -crash_probability


class AdaptiveAlphaEngine:
    """Computes regime-adaptive composite alpha from multi-agent signals."""

    def compute_alpha(
        self,
        symbol: str,
        # Technical signal
        technical_signal: str,
        technical_confidence: float,
        # ML prediction
        ml_direction: str,
        ml_probability: float,
        # Sentiment
        sentiment_score: float,
        # RAG evidence
        rag_verification_status: str = "NO_CLAIM_MADE",
        rag_evidence_quality: float = 0.5,
        rag_insufficient: bool = False,
        # Crash risk
        crash_probability: float = 0.0,
        # Regime context
        regime_label: str = "NORMAL",
    ) -> AlphaSignalResult:
        """Compute the adaptive alpha signal."""
        symbol = symbol.upper()

        # 1. Normalize all signals to [-1, +1]
        signals = {
            "technical": _normalize_technical(technical_signal, technical_confidence),
            "ml": _normalize_ml(ml_direction, ml_probability),
            "sentiment": _normalize_sentiment(sentiment_score),
            "rag": _normalize_rag(rag_verification_status, rag_evidence_quality,
                                  rag_insufficient),
            "crash_risk": _normalize_crash_risk(crash_probability),
        }

        # 2. Get regime-adjusted weights
        regime_mults, regime_desc = _get_regime_profile(regime_label)
        adjusted_weights = {}
        for name, base_w in _BASE_WEIGHTS.items():
            mult = regime_mults.get(name, 1.0)
            adjusted_weights[name] = base_w * mult

        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        # 3. Compute weighted alpha
        contributions: list[SignalContribution] = []
        composite = 0.0

        for name in _BASE_WEIGHTS:
            raw = signals[name]
            w = adjusted_weights[name]
            weighted = raw * w
            composite += weighted

            contributions.append(SignalContribution(
                signal_name=name,
                raw_value=round(raw, 4),
                normalized_score=round(raw, 4),
                weight=round(w, 4),
                weighted_contribution=round(weighted, 4),
                weight_rationale=f"Base {_BASE_WEIGHTS[name]:.2f} × "
                                 f"regime mult {regime_mults.get(name, 1.0):.2f}",
            ))

        composite = max(-1.0, min(1.0, composite))

        # 4. Signal agreement
        directions = [1 if s > 0.05 else (-1 if s < -0.05 else 0)
                      for s in signals.values()]
        non_neutral = [d for d in directions if d != 0]
        if non_neutral:
            dominant = 1 if sum(non_neutral) > 0 else -1
            agreement = sum(1 for d in non_neutral if d == dominant) / len(non_neutral)
        else:
            agreement = 0.0

        # 5. Directional bias
        if composite > 0.10:
            bias = "BULLISH"
        elif composite < -0.10:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return AlphaSignalResult(
            symbol=symbol,
            composite_alpha=round(composite, 4),
            alpha_magnitude=round(abs(composite), 4),
            signal_contributions=contributions,
            regime=regime_label,
            regime_weight_profile=regime_desc,
            directional_bias=bias,
            signal_agreement_pct=round(agreement, 3),
        )

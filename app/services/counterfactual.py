"""Counterfactual Trading Explanation Service.

Research requirement: Explainable AI in quantitative trading must not only explain
"why this trade was chosen", but what counterfactual changes in input parameters
would have flipped the decision (contrastive / counterfactual explanations).

Sections:
  1. CURRENT DECISION (BUY, SELL, HOLD, ABSTAIN)
  2. WHY (primary empirical drivers and weights)
  3. WHAT WOULD CHANGE THE DECISION (counterfactual perturbations required to alter decision)

Counterfactual Scenarios Evaluated:
  - Volatility increase by +25% or +50%
  - Sentiment dropping to strongly negative (< -0.40)
  - CVaR 95% breaching the risk ceiling
  - ML Prediction confidence eroding below 0.52
  - Regime transition from Bull/Sideways to Bear_High_Volatility
"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.agents.confidence_engine import ConfidenceDimensions


class CounterfactualCondition(BaseModel):
    parameter: str
    baseline_value: str
    counterfactual_shift: str
    consequence_decision: str
    explanation: str
    # Numerical sensitivity fields (Upgrade 6)
    flip_threshold: float | None = None
    current_headroom_pct: float | None = None
    sensitivity_magnitude: str = ""  # LOW, MODERATE, HIGH, CRITICAL

    @property
    def distance_to_flip(self) -> float | None:
        return self.current_headroom_pct


class CounterfactualExplanation(BaseModel):
    symbol: str
    current_decision: str
    primary_reasons_why: list[str]
    what_would_change_decision: list[CounterfactualCondition]
    summary_text: str
    closest_flip_parameter: str | None = None
    closest_flip_headroom_pct: float | None = None
    overall_decision_stability: str = ""  # FRAGILE, MODERATE, ROBUST
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def has_sensitivity_metrics(self) -> bool:
        return any(c.current_headroom_pct is not None for c in self.what_would_change_decision)

    @property
    def flip_conditions(self) -> list[CounterfactualCondition]:
        return self.what_would_change_decision


class CounterfactualService:
    """Computes contrastive sensitivity analysis for trading decisions."""

    def generate_explanation(
        self,
        symbol: str,
        current_decision: str,
        conviction_score: float,
        technical_signal: str,
        technical_confidence: float,
        ml_direction: str,
        ml_probability: float,
        sentiment_score: float,
        regime_label: str,
        cvar_95: float,
        vol_percentile: float,
        confidence_dims: ConfidenceDimensions,
    ) -> CounterfactualExplanation:
        symbol = symbol.upper()
        why: list[str] = []
        conditions: list[CounterfactualCondition] = []

        # 1. WHY section
        if current_decision == "BUY":
            why.append(f"Combined directional conviction ({conviction_score:+.2f}) exceeded BUY threshold (+0.15).")
            if technical_signal == "BUY":
                why.append(f"Technical momentum and moving averages aligned bullishly (confidence={technical_confidence:.2f}).")
            if ml_direction == "UP":
                why.append(f"Machine learning model projected an upward trajectory with {ml_probability:.1%} probability.")
            if sentiment_score > 0:
                why.append(f"News sentiment was supportive ({sentiment_score:+.2f}).")
            why.append(f"Deterministic risk engine validated position size within VaR/CVaR limits.")
        elif current_decision == "SELL":
            why.append(f"Combined conviction ({conviction_score:+.2f}) dropped below SELL threshold (-0.15).")
            if technical_signal == "SELL":
                why.append("Bearish technical breakdown detected across moving averages.")
            if ml_direction == "DOWN":
                why.append(f"ML model projected downside with {1 - ml_probability:.1%} probability.")
        elif current_decision == "ABSTAIN":
            why.append("Multi-source uncertainty or evidence deficiency triggered protective abstention.")
            if confidence_dims.prediction_confidence < 0.20:
                why.append("ML model probability was ambiguous (near 50% coin-flip).")
            if confidence_dims.evidence_quality < 0.30:
                why.append("Insufficient verified financial filing citations available.")
            if confidence_dims.market_stability < 0.30:
                why.append("Market stability depleted due to extreme volatility.")
        else:  # HOLD
            why.append(f"Conviction ({conviction_score:+.2f}) remained within the neutral band [-0.15, +0.15].")
            why.append("Signals were balanced or lacked strong directional momentum.")

        # 2. WHAT WOULD CHANGE THE DECISION (Counterfactual conditions with numerical sensitivity)
        if current_decision == "BUY":
            # Condition 1: Volatility increase
            th_v, hr_v, mag_v = self._compute_sensitivity(vol_percentile, 0.65, "Asset Volatility")
            conditions.append(CounterfactualCondition(
                parameter="Asset Volatility",
                baseline_value=f"Percentile {vol_percentile:.1%}",
                counterfactual_shift="Volatility increases by +25% (or breaches 65th percentile)",
                consequence_decision="HOLD / ABSTAIN",
                explanation="Elevated volatility compresses the adaptive risk budget and widens tail VaR, triggering protective abstention.",
                flip_threshold=th_v,
                current_headroom_pct=hr_v,
                sensitivity_magnitude=mag_v,
            ))
            # Condition 2: Sentiment shock
            th_s, hr_s, mag_s = self._compute_sensitivity(sentiment_score, -0.30, "News Sentiment")
            conditions.append(CounterfactualCondition(
                parameter="News Sentiment",
                baseline_value=f"{sentiment_score:+.2f}",
                counterfactual_shift="Sentiment score drops below -0.30 (material negative news)",
                consequence_decision="HOLD",
                explanation="Negative sentiment would subtract 0.15+ from combined conviction, pulling it back into the neutral zone.",
                flip_threshold=th_s,
                current_headroom_pct=hr_s,
                sensitivity_magnitude=mag_s,
            ))
            # Condition 3: CVaR breach
            th_c, hr_c, mag_c = self._compute_sensitivity(cvar_95, 0.06, "CVaR 95%")
            conditions.append(CounterfactualCondition(
                parameter="CVaR 95%",
                baseline_value=f"{cvar_95:.2%}",
                counterfactual_shift="CVaR 95% exceeds 6.0%",
                consequence_decision="FORCED HOLD",
                explanation="Deterministic Risk Engine would veto the trade unconditionally regardless of AI agent enthusiasm.",
                flip_threshold=th_c,
                current_headroom_pct=hr_c,
                sensitivity_magnitude=mag_c,
            ))
            # Condition 4: ML confidence decrease
            th_m, hr_m, mag_m = self._compute_sensitivity(ml_probability, 0.52, "ML Win Probability")
            conditions.append(CounterfactualCondition(
                parameter="ML Win Probability",
                baseline_value=f"{ml_probability:.1%}",
                counterfactual_shift="Probability falls below 52.0%",
                consequence_decision="ABSTAIN",
                explanation="High Shannon entropy would trigger the Uncertainty-Aware Abstention layer.",
                flip_threshold=th_m,
                current_headroom_pct=hr_m,
                sensitivity_magnitude=mag_m,
            ))
            # Condition 5: Regime change
            conditions.append(CounterfactualCondition(
                parameter="Market Regime",
                baseline_value=regime_label,
                counterfactual_shift="Regime transitions from Bullish/Neutral to BEAR_HIGH_VOLATILITY",
                consequence_decision="SELL / ABSTAIN",
                explanation="Macro trend headwind switches regime component penalty from positive to -0.50.",
                sensitivity_magnitude="HIGH",
            ))

        elif current_decision in {"HOLD", "ABSTAIN"}:
            # What would flip it to BUY?
            th_m, hr_m, mag_m = self._compute_sensitivity(ml_probability, 0.65, "ML Prediction Sharpness")
            conditions.append(CounterfactualCondition(
                parameter="ML Prediction Sharpness",
                baseline_value=f"{ml_probability:.1%}",
                counterfactual_shift="ML probability increases above 65.0% with test accuracy > 55%",
                consequence_decision="BUY",
                explanation="Strong ML probability certainty would elevate conviction above the +0.15 threshold.",
                flip_threshold=th_m,
                current_headroom_pct=hr_m,
                sensitivity_magnitude=mag_m,
            ))
            th_e, hr_e, mag_e = self._compute_sensitivity(confidence_dims.evidence_quality, 0.70, "RAG Evidence Verification")
            conditions.append(CounterfactualCondition(
                parameter="RAG Evidence Verification",
                baseline_value=f"Quality {confidence_dims.evidence_quality:.2f}",
                counterfactual_shift="New 10-Q/10-K filing confirms >10% revenue acceleration with >2 citations",
                consequence_decision="BUY",
                explanation="High evidence quality lifts the uncertainty abstention lock.",
                flip_threshold=th_e,
                current_headroom_pct=hr_e,
                sensitivity_magnitude=mag_e,
            ))
            th_v, hr_v, mag_v = self._compute_sensitivity(vol_percentile, 0.30, "Market Volatility & Regime")
            conditions.append(CounterfactualCondition(
                parameter="Market Volatility & Regime",
                baseline_value=regime_label,
                counterfactual_shift="Vol percentile drops below 30th percentile and SMA20 > SMA50 establishes BULL regime",
                consequence_decision="BUY",
                explanation="Calm trending regime expands adaptive risk budget and boosts regime score.",
                flip_threshold=th_v,
                current_headroom_pct=hr_v,
                sensitivity_magnitude=mag_v,
            ))

        elif current_decision == "SELL":
            conditions.append(CounterfactualCondition(
                parameter="Technical Support",
                baseline_value=f"RSI {technical_confidence:.2f}",
                counterfactual_shift="RSI crosses above 50 and 20-day SMA slopes positive",
                consequence_decision="HOLD",
                explanation="Trend stabilization neutralizes the negative conviction component.",
            ))
            conditions.append(CounterfactualCondition(
                parameter="Earnings Surprise / Positive News",
                baseline_value=f"{sentiment_score:+.2f}",
                counterfactual_shift="Sentiment jumps to strongly positive (> +0.50)",
                consequence_decision="ABSTAIN / HOLD",
                explanation="Divergence between bearish technicals and bullish news creates conflict, forcing abstention.",
            ))

        # Format concise human-readable summary
        summary = (
            f"CURRENT DECISION: {current_decision}\n\n"
            f"WHY:\n• " + "\n• ".join(why) + "\n\n"
            f"WHAT WOULD CHANGE THE DECISION:\n"
            + "\n".join([f"• {c.parameter}: If {c.counterfactual_shift.lower()} -> flips to {c.consequence_decision}."
                         + (f" [Headroom: {c.current_headroom_pct:.1f}% — {c.sensitivity_magnitude}]" if c.current_headroom_pct is not None else "")
                        for c in conditions[:3]])
        )

        # Compute overall decision stability
        headrooms = [c.current_headroom_pct for c in conditions if c.current_headroom_pct is not None]
        closest_flip = None
        closest_headroom = None
        if headrooms:
            min_headroom = min(headrooms)
            closest_idx = next(i for i, c in enumerate(conditions)
                               if c.current_headroom_pct == min_headroom)
            closest_flip = conditions[closest_idx].parameter
            closest_headroom = min_headroom

        if closest_headroom is not None:
            if closest_headroom < 15.0:
                stability = "FRAGILE"
            elif closest_headroom < 40.0:
                stability = "MODERATE"
            else:
                stability = "ROBUST"
        else:
            stability = "UNKNOWN"

        return CounterfactualExplanation(
            symbol=symbol,
            current_decision=current_decision,
            primary_reasons_why=why,
            what_would_change_decision=conditions,
            summary_text=summary,
            closest_flip_parameter=closest_flip,
            closest_flip_headroom_pct=closest_headroom,
            overall_decision_stability=stability,
        )

    @staticmethod
    def _compute_sensitivity(
        current_value: float,
        flip_threshold: float,
        parameter_name: str,
    ) -> tuple[float, float, str]:
        """Compute numerical distance-to-flip and classify sensitivity.

        Returns (flip_threshold, headroom_pct, magnitude_label).
        """
        if flip_threshold == 0:
            return flip_threshold, 100.0, "LOW"

        distance = abs(current_value - flip_threshold)
        headroom = (distance / max(abs(flip_threshold), 1e-6)) * 100.0

        if headroom < 10.0:
            magnitude = "CRITICAL"
        elif headroom < 25.0:
            magnitude = "HIGH"
        elif headroom < 50.0:
            magnitude = "MODERATE"
        else:
            magnitude = "LOW"

        return flip_threshold, round(headroom, 2), magnitude

"""Uncertainty-Aware Abstention Layer for Trading Decisions.

Research Goal: In real financial markets and safety-critical AI trading, forced
binary/ternary actions (BUY/SELL/HOLD) under high uncertainty lead to poor risk-adjusted
outcomes and excessive drawdown.

This module introduces ABSTAIN as a first-class decision action alongside BUY, SELL, HOLD.
The system abstains if:
  1. Prediction Uncertainty is high (Shannon entropy close to maximum or probability close to 0.50).
  2. Financial Evidence Quality is insufficient or contradictory.
  3. Market Stability is critically low (extreme volatility spike / ambiguous regime).
  4. Risk conditions are hazardous (VaR/CVaR near limit, portfolio drawdown elevated).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.agents.confidence_engine import ConfidenceDimensions


class AbstentionEvaluation(BaseModel):
    should_abstain: bool
    decision: str  # BUY / SELL / HOLD / ABSTAIN
    abstention_reason: str | None = None
    prediction_uncertainty: float = Field(ge=0.0, le=1.0, description="Normalized Shannon entropy of prediction probabilities")
    evidence_deficiency: float = Field(ge=0.0, le=1.0, description="1.0 - evidence_quality")
    market_instability: float = Field(ge=0.0, le=1.0, description="1.0 - market_stability")
    risk_hazard: float = Field(ge=0.0, le=1.0, description="1.0 - risk_confidence")
    triggers: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UncertaintyEvaluator:
    """Evaluates multi-source uncertainty and guards trading decisions via ABSTAIN."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.70,
        min_evidence_quality: float = 0.25,
        min_market_stability: float = 0.20,
        min_risk_confidence: float = 0.25,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.min_evidence_quality = min_evidence_quality
        self.min_market_stability = min_market_stability
        self.min_risk_confidence = min_risk_confidence

    def calculate_entropy(self, p: float) -> float:
        """Normalized Shannon entropy for binary prediction (UP vs DOWN).

        Entropy is 1.0 when p=0.50 (maximum uncertainty) and 0.0 when p=0 or 1.
        """
        p = max(1e-6, min(1.0 - 1e-6, float(p)))
        q = 1.0 - p
        entropy = -(p * math.log2(p) + q * math.log2(q))
        return float(entropy)  # already normalized between 0.0 and 1.0 for binary

    def evaluate(
        self,
        raw_decision: str,
        ml_probability: float,
        confidence_dims: ConfidenceDimensions,
        insufficient_rag_evidence: bool = False,
        rag_conflict_detected: bool = False,
    ) -> AbstentionEvaluation:
        # 1. Prediction uncertainty
        uncertainty = self.calculate_entropy(ml_probability)

        evidence_deficiency = max(0.0, min(1.0, 1.0 - confidence_dims.evidence_quality))
        market_instability = max(0.0, min(1.0, 1.0 - confidence_dims.market_stability))
        risk_hazard = max(0.0, min(1.0, 1.0 - confidence_dims.risk_confidence))

        triggers: list[str] = []

        # Check triggers for active directional intent (BUY or SELL)
        if raw_decision in {"BUY", "SELL"}:
            if uncertainty > self.uncertainty_threshold:
                triggers.append(
                    f"High ML prediction uncertainty ({uncertainty:.2f} > threshold {self.uncertainty_threshold:.2f}, "
                    f"prob={ml_probability:.2f})"
                )

            if insufficient_rag_evidence or confidence_dims.evidence_quality < self.min_evidence_quality:
                triggers.append(
                    f"Insufficient financial evidence quality ({confidence_dims.evidence_quality:.2f} < "
                    f"threshold {self.min_evidence_quality:.2f})"
                )

            if rag_conflict_detected:
                triggers.append("Contradictory or conflicting financial news/filing evidence detected")

            if confidence_dims.market_stability < self.min_market_stability:
                triggers.append(
                    f"Market instability too severe ({confidence_dims.market_stability:.2f} < "
                    f"threshold {self.min_market_stability:.2f})"
                )

            if confidence_dims.risk_confidence < self.min_risk_confidence:
                triggers.append(
                    f"Risk confidence depleted ({confidence_dims.risk_confidence:.2f} < "
                    f"threshold {self.min_risk_confidence:.2f})"
                )

        should_abstain = len(triggers) > 0

        if should_abstain:
            final_action = "ABSTAIN"
            reason = "ABSTAIN: " + "; ".join(triggers)
        else:
            final_action = raw_decision
            reason = None

        return AbstentionEvaluation(
            should_abstain=should_abstain,
            decision=final_action,
            abstention_reason=reason,
            prediction_uncertainty=round(uncertainty, 4),
            evidence_deficiency=round(evidence_deficiency, 4),
            market_instability=round(market_instability, 4),
            risk_hazard=round(risk_hazard, 4),
            triggers=triggers,
        )

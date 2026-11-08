"""Multi-Dimensional Confidence Engine for Trading Decisions.

Research requirement: replaces single-number naive confidence with 5 independent,
transparent dimensions:
  1. Signal Confidence: Technical indicator alignment and momentum strength.
  2. Prediction Confidence: ML classifier probability certainty & entropy margin.
  3. Evidence Quality: Financial RAG citation coverage, recency, and verification.
  4. Market Stability: Inverse volatility percentile and regime clarity.
  5. Risk Confidence: Distance between portfolio risk metrics and regulatory limits.

CRITICAL RESEARCH RULE: The composite score is a transparent weighted decision
conviction index, NOT a guaranteed probability of profit or expected return.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ConfidenceDimensions(BaseModel):
    signal_confidence: float = Field(ge=0.0, le=1.0, description="Agreement among technical signals and momentum")
    prediction_confidence: float = Field(ge=0.0, le=1.0, description="ML prediction probability sharpness")
    evidence_quality: float = Field(ge=0.0, le=1.0, description="Grounding, citation count, and verification status of RAG evidence")
    market_stability: float = Field(ge=0.0, le=1.0, description="Regime stability and inverse volatility percentile")
    risk_confidence: float = Field(ge=0.0, le=1.0, description="Safety margin relative to VaR, CVaR, and concentration limits")
    composite_confidence: float = Field(ge=0.0, le=1.0, description="Weighted transparent conviction score [0.0, 1.0]")
    dimension_notes: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = "Research simulation metric. Does not represent guaranteed probability of profit or risk-free return."
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MultiDimensionalConfidenceEngine:
    """Computes five independent confidence dimensions and transparent synthesis."""

    # Explicit weights for composite conviction (sum = 1.0)
    WEIGHT_SIGNAL = 0.25
    WEIGHT_PREDICTION = 0.25
    WEIGHT_EVIDENCE = 0.20
    WEIGHT_STABILITY = 0.15
    WEIGHT_RISK = 0.15

    def calculate(
        self,
        technical_confidence: float,
        ml_probability: float,
        rag_verification_status: str,
        rag_citations_count: int,
        regime_confidence: float,
        vol_percentile: float,
        risk_status: str,
        risk_violations: list[str],
        var_95: float,
        max_portfolio_vol: float = 0.35,
        realized_vol: float = 0.20,
    ) -> ConfidenceDimensions:
        # 1. Signal Confidence: directly grounded in technical indicator agreement
        sig_conf = max(0.0, min(1.0, float(technical_confidence)))

        # 2. Prediction Confidence: distance from uniform uncertainty (0.50)
        # Prob = 0.50 -> Pred Conf = 0.0; Prob = 1.0 or 0.0 -> Pred Conf = 1.0
        prob_dist = abs(float(ml_probability) - 0.50) * 2.0
        pred_conf = max(0.0, min(1.0, prob_dist))

        # 3. Evidence Quality: derived from citation grounding & verification status
        if rag_verification_status == "SUPPORTED":
            base_ev = 0.85
        elif rag_verification_status == "PARTIALLY_UNSUPPORTED":
            base_ev = 0.45
        elif rag_verification_status == "NO_CLAIM_MADE":
            base_ev = 0.20
        else:  # UNSUPPORTED
            base_ev = 0.10

        # Bonus for depth of citations (up to +0.15)
        citation_bonus = min(0.15, rag_citations_count * 0.05)
        ev_quality = max(0.0, min(1.0, base_ev + citation_bonus))

        # 4. Market Stability: calmer market (lower vol percentile) & higher regime confidence
        # High vol percentile (e.g. 0.90) => lower stability (1.0 - 0.90 = 0.10)
        inv_vol = max(0.0, min(1.0, 1.0 - float(vol_percentile)))
        mkt_stability = max(0.0, min(1.0, 0.5 * inv_vol + 0.5 * float(regime_confidence)))

        # 5. Risk Confidence: checks violations and headroom
        if risk_status != "PASS" or len(risk_violations) > 0:
            risk_conf = 0.10
        else:
            vol_headroom = max(0.0, (max_portfolio_vol - realized_vol) / max_portfolio_vol)
            var_penalty = min(0.4, var_95 * 5.0)  # higher VaR reduces risk confidence
            risk_conf = max(0.20, min(1.0, 0.70 + 0.30 * vol_headroom - var_penalty))

        # Composite Conviction
        composite = (
            self.WEIGHT_SIGNAL * sig_conf
            + self.WEIGHT_PREDICTION * pred_conf
            + self.WEIGHT_EVIDENCE * ev_quality
            + self.WEIGHT_STABILITY * mkt_stability
            + self.WEIGHT_RISK * risk_conf
        )
        composite = max(0.0, min(1.0, round(composite, 4)))

        notes = {
            "signal": f"Technical indicator alignment confidence: {sig_conf:.2f}",
            "prediction": f"ML prediction sharpness: {pred_conf:.2f} (prob={ml_probability:.2f})",
            "evidence": f"RAG evidence quality: {ev_quality:.2f} ({rag_verification_status}, {rag_citations_count} citations)",
            "stability": f"Market stability: {mkt_stability:.2f} (vol percentile={vol_percentile:.2f})",
            "risk": f"Risk engine confidence: {risk_conf:.2f} (status={risk_status}, {len(risk_violations)} violations)",
        }

        return ConfidenceDimensions(
            signal_confidence=round(sig_conf, 4),
            prediction_confidence=round(pred_conf, 4),
            evidence_quality=round(ev_quality, 4),
            market_stability=round(mkt_stability, 4),
            risk_confidence=round(risk_conf, 4),
            composite_confidence=composite,
            dimension_notes=notes,
        )

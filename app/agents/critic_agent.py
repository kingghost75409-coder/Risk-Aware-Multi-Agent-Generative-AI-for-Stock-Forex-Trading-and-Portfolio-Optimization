"""Self-Critiquing Trading Agent.

Research concept: Single-pass LLM or rule-based decisions suffer from confirmation bias
and unexamined contradictions between modules (e.g., technical momentum screams BUY while
ML predicts a drop, or sentiment is exuberant while 10-K filings disclose looming supply shocks).

The Critic Agent acts as an adversarial Devil's Advocate that:
  1. Inspects the Trader's proposed decision against each agent's individual evidence.
  2. Pinpoints logical weaknesses, empirical conflicts, and blind spots.
  3. Formulates a severity score in [0.0, 1.0].
  4. If contradictions are severe, recommends a down-regulated or corrected decision
     (e.g., BUY -> HOLD or ABSTAIN).
  5. Passes to the deterministic Risk Engine for inviolable mathematical verification.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.agents.ml_agent import MLPrediction
from app.agents.regime_agent import RegimeResult
from app.agents.sentiment_agent import SentimentResult
from app.agents.technical_agent import TechnicalSignal
from app.rag.rag_agent import RAGResult
from app.risk.risk_engine import RiskCheckResult


class CritiqueReport(BaseModel):
    symbol: str
    proposed_decision: str
    critic_verdict: str  # ENDORSED / CHALLENGED / REVISE_TO_HOLD / REVISE_TO_ABSTAIN
    critic_severity: float = Field(ge=0.0, le=1.0, description="Severity of detected contradictions [0=None, 1=Severe]")
    identified_weaknesses: list[str]
    suggested_decision: str
    critic_reasoning: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CriticAgent:
    """Interrogates candidate trading decisions across cross-agent signals."""

    def critique(
        self,
        symbol: str,
        proposed_decision: str,
        conviction_score: float,
        technical: TechnicalSignal,
        prediction: MLPrediction,
        sentiment: SentimentResult,
        regime: RegimeResult,
        rag_evidence: RAGResult,
        risk: RiskCheckResult,
    ) -> CritiqueReport:
        weaknesses: list[str] = []
        severity = 0.0

        if proposed_decision == "BUY":
            # 1. Check Technical contradictions
            rsi = technical.indicators.get("rsi_14")
            if rsi is not None and rsi > 70:
                weaknesses.append(f"Overbought condition: RSI-14 is {rsi:.1f} (>70), high risk of mean reversion pullback.")
                severity += 0.25

            # 2. Check ML Prediction contradiction
            if prediction.predicted_direction == "DOWN":
                weaknesses.append(f"ML Directional Conflict: ML model predicts DOWN (prob={1 - prediction.probability:.2f}).")
                severity += 0.35
            elif prediction.probability < 0.53:
                weaknesses.append(f"Weak ML Edge: ML win probability {prediction.probability:.2f} is barely above coin-flip (0.50).")
                severity += 0.15

            # 3. Check Sentiment divergence
            if sentiment.label == "NEGATIVE" or sentiment.sentiment_score < -0.15:
                weaknesses.append(f"Sentiment Divergence: Market news sentiment is negative ({sentiment.sentiment_score:.2f}).")
                severity += 0.20

            # 4. Check Regime mismatch
            if "BEAR" in regime.regime:
                weaknesses.append(f"Macro Trend Headwind: Asset is in a {regime.regime} regime; buying into broad down-trend.")
                severity += 0.30
            elif "HIGH_VOLATILITY" in regime.regime:
                weaknesses.append("Elevated Volatility Regime: Wide bid-ask spreads and whipsaws increase stop-out risk.")
                severity += 0.15

            # 5. Check RAG Risk warnings
            if rag_evidence.verification.get("status") == "NO_CLAIM_MADE" or rag_evidence.insufficient_evidence:
                weaknesses.append("Unverified Fundamentals: No timely filing evidence in knowledge base supporting growth thesis.")
                severity += 0.15
            elif "risk" in rag_evidence.answer.lower() and "adverse" in rag_evidence.answer.lower():
                weaknesses.append("Filing Risk Disclosure: RAG retrieved significant material risk disclosures.")
                severity += 0.10

            # 6. Check Risk Engine headroom
            if risk.var_95 > 0.05:
                weaknesses.append(f"High Tail Risk: Daily VaR 95% is elevated at {risk.var_95:.2%}.")
                severity += 0.20

        elif proposed_decision == "SELL":
            rsi = technical.indicators.get("rsi_14")
            if rsi is not None and rsi < 30:
                weaknesses.append(f"Oversold condition: RSI-14 is {rsi:.1f} (<30), shorting into potential exhaustion bounce.")
                severity += 0.25

            if prediction.predicted_direction == "UP" and prediction.probability > 0.60:
                weaknesses.append(f"ML Directional Conflict: ML model projects UP probability {prediction.probability:.2f}.")
                severity += 0.35

            if "BULL" in regime.regime:
                weaknesses.append("Trend Headwind: Shorting into a secular Bull trend.")
                severity += 0.30

        elif proposed_decision in {"HOLD", "ABSTAIN"}:
            # For Hold/Abstain, verify if there's an obvious missed opportunity or if hold is justified
            if abs(conviction_score) > 0.40 and len(risk.violations) == 0:
                weaknesses.append("Caution: Strong directional conviction was suppressed to non-trade action.")
                severity += 0.10

        severity = min(1.0, severity)

        # Determine verdict and suggested decision
        if severity >= 0.60:
            critic_verdict = "REVISE_TO_ABSTAIN" if rag_evidence.insufficient_evidence else "REVISE_TO_HOLD"
            suggested_decision = "ABSTAIN" if rag_evidence.insufficient_evidence else "HOLD"
            reasoning = (
                f"Severe cross-agent contradiction detected (severity={severity:.2f}): "
                f"The proposed {proposed_decision} is undermined by {len(weaknesses)} major objection(s). "
                f"Recommended de-escalation to {suggested_decision}."
            )
        elif severity >= 0.35:
            critic_verdict = "CHALLENGED"
            suggested_decision = proposed_decision
            reasoning = (
                f"Moderate concerns noted (severity={severity:.2f}): "
                f"{len(weaknesses)} point(s) of friction. Proceed with tight risk boundaries."
            )
        else:
            critic_verdict = "ENDORSED"
            suggested_decision = proposed_decision
            reasoning = (
                f"Cross-agent consensus is coherent (severity={severity:.2f}). "
                "No critical contradictions identified."
            )

        return CritiqueReport(
            symbol=symbol,
            proposed_decision=proposed_decision,
            critic_verdict=critic_verdict,
            critic_severity=round(severity, 3),
            identified_weaknesses=weaknesses,
            suggested_decision=suggested_decision,
            critic_reasoning=reasoning,
        )

"""Horizon Strategy Router.

Determines whether the investment strategy request is for LONG_TERM or SHORT_TERM
(or COMBINED portfolio view), routes to the prioritized agents without duplicating
them, calculates separate Long-Term and Short-Term confidence dimensions, and applies
horizon-specific abstention conditions while strictly respecting the deterministic
Risk Engine and existing decision gate.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


class HorizonType(str, Enum):
    LONG_TERM = "LONG_TERM"
    SHORT_TERM = "SHORT_TERM"
    COMBINED = "COMBINED"


# Standardized horizon aliases
_HORIZON_MAP = {
    # Long-term horizons
    "6m": HorizonType.LONG_TERM,
    "6months": HorizonType.LONG_TERM,
    "6_months": HorizonType.LONG_TERM,
    "1y": HorizonType.LONG_TERM,
    "1year": HorizonType.LONG_TERM,
    "1_year": HorizonType.LONG_TERM,
    "3y": HorizonType.LONG_TERM,
    "3years": HorizonType.LONG_TERM,
    "3_years": HorizonType.LONG_TERM,
    "5y": HorizonType.LONG_TERM,
    "5years": HorizonType.LONG_TERM,
    "5_years": HorizonType.LONG_TERM,
    "long_term": HorizonType.LONG_TERM,
    "long-term": HorizonType.LONG_TERM,
    "longterm": HorizonType.LONG_TERM,

    # Short-term horizons
    "intraday": HorizonType.SHORT_TERM,
    "1d": HorizonType.SHORT_TERM,
    "1day": HorizonType.SHORT_TERM,
    "1_day": HorizonType.SHORT_TERM,
    "1w": HorizonType.SHORT_TERM,
    "1week": HorizonType.SHORT_TERM,
    "1_week": HorizonType.SHORT_TERM,
    "1-4w": HorizonType.SHORT_TERM,
    "1_4_weeks": HorizonType.SHORT_TERM,
    "1-4_weeks": HorizonType.SHORT_TERM,
    "1_to_4_weeks": HorizonType.SHORT_TERM,
    "1-4weeks": HorizonType.SHORT_TERM,
    "short_term": HorizonType.SHORT_TERM,
    "short-term": HorizonType.SHORT_TERM,
    "shortterm": HorizonType.SHORT_TERM,

    # Combined
    "combined": HorizonType.COMBINED,
    "all": HorizonType.COMBINED,
}


def parse_horizon(horizon_str: str | HorizonType | None) -> HorizonType:
    if isinstance(horizon_str, HorizonType):
        return horizon_str
    if not horizon_str:
        return HorizonType.LONG_TERM
    key = str(horizon_str).strip().lower()
    return _HORIZON_MAP.get(key, HorizonType.LONG_TERM)


class LongTermConfidence(BaseModel):
    fundamental_confidence: float = Field(ge=0.0, le=1.0)
    growth_confidence: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    long_term_risk_confidence: float = Field(ge=0.0, le=1.0)
    composite_long_term_confidence: float = Field(ge=0.0, le=1.0)
    composite_confidence: float = 0.0  # Alias for integration


class ShortTermConfidence(BaseModel):
    signal_confidence: float = Field(ge=0.0, le=1.0)
    prediction_confidence: float = Field(ge=0.0, le=1.0)
    market_stability: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    risk_confidence: float = Field(ge=0.0, le=1.0)
    composite_short_term_confidence: float = Field(ge=0.0, le=1.0)


class HorizonRoutingResult(BaseModel):
    horizon_type: HorizonType
    requested_horizon: str
    prioritized_agents: list[str]
    supporting_agents: list[str]
    routing_rationale: str
    long_term_confidence: LongTermConfidence | None = None
    short_term_confidence: ShortTermConfidence | None = None
    should_abstain: bool = False
    abstention_reason: str | None = None
    decision: str = "HOLD"


class HorizonStrategyRouter:
    """Routes requests based on the selected horizon. Reuses existing agents,
    evaluates separate long-term vs short-term confidence, and enforces
    horizon-aware abstention."""

    @staticmethod
    def parse_horizon(horizon_str: str | HorizonType | None) -> HorizonType:
        return parse_horizon(horizon_str)

    def route_horizon(
        self,
        requested_horizon: str,
        decision: Any,
        fundamental_evidence_count: int = 1,
    ) -> HorizonRoutingResult:
        htype = parse_horizon(requested_horizon)
        symbol = getattr(decision, "symbol", "AAPL")
        crash_p = decision.crash_risk.crash_probability if getattr(decision, "crash_risk", None) else 0.1
        rag_qual = getattr(decision.rag_evidence, "evidence_quality", 0.7) if getattr(decision, "rag_evidence", None) else 0.5
        rag_insufficient = (fundamental_evidence_count == 0) or getattr(decision.rag_evidence, "insufficient_evidence", False) if getattr(decision, "rag_evidence", None) else False
        tech_conf = decision.technical.confidence if getattr(decision, "technical", None) else 0.6
        ml_prob = decision.prediction.probability if getattr(decision, "prediction", None) else 0.55
        risk_violations = len(decision.risk.violations) if getattr(decision, "risk", None) else 0
        risk_status = decision.risk.status if getattr(decision, "risk", None) else "PASS"

        res = self.route(
            horizon=requested_horizon,
            symbol=symbol,
            crash_probability=crash_p,
            rag_evidence_quality=rag_qual,
            rag_insufficient=rag_insufficient,
            technical_confidence=tech_conf,
            ml_probability=ml_prob,
            risk_violations_count=risk_violations,
            risk_status=risk_status,
        )
        if res.should_abstain:
            res.decision = "ABSTAIN"
        else:
            res.decision = getattr(decision, "final_decision", "HOLD")
        return res

    def route(
        self,
        horizon: str | HorizonType,
        symbol: str,
        vol_percentile: float = 0.5,
        crash_probability: float = 0.0,
        rag_evidence_quality: float = 0.5,
        rag_insufficient: bool = False,
        fundamental_quality_score: float = 70.0,
        valuation_score: float = 65.0,
        technical_confidence: float = 0.6,
        ml_probability: float = 0.55,
        risk_violations_count: int = 0,
        risk_status: str = "PASS",
    ) -> HorizonRoutingResult:
        htype = parse_horizon(horizon)
        symbol = symbol.upper()

        if htype == HorizonType.LONG_TERM:
            prioritized = [
                "FinancialRAGAgent",
                "FundamentalQualityScorer",
                "GrowthEngine",
                "ValuationModel",
                "MarketRegimeAgent",
                "LongTermRiskEngine",
                "PortfolioOptimizer",
            ]
            supporting = ["MLPredictionAgent", "CrashRiskAgent", "TechnicalAgent"]
            rationale = (
                f"Long-term horizon ({horizon}) routes primary weight to Financial RAG, "
                f"Fundamental Quality, Balance Sheet Stability, and Long-Term Risk. "
                f"Technical indicators serve purely as secondary timing reference."
            )

            # Long-term confidence formulation
            fund_conf = min(1.0, max(0.1, fundamental_quality_score / 100.0))
            growth_conf = min(1.0, max(0.1, (fundamental_quality_score * 0.5 + valuation_score * 0.5) / 100.0))
            ev_quality = min(1.0, max(0.1, rag_evidence_quality))
            risk_conf = 1.0 - (0.4 * min(1.0, crash_probability) + 0.3 * (1.0 if risk_violations_count > 0 else 0.0))
            risk_conf = min(1.0, max(0.1, risk_conf))

            composite = (
                fund_conf * 0.35 +
                growth_conf * 0.25 +
                ev_quality * 0.20 +
                risk_conf * 0.20
            )
            lt_conf = LongTermConfidence(
                fundamental_confidence=round(fund_conf, 3),
                growth_confidence=round(growth_conf, 3),
                evidence_quality=round(ev_quality, 3),
                long_term_risk_confidence=round(risk_conf, 3),
                composite_long_term_confidence=round(composite, 3),
                composite_confidence=round(composite, 3),
            )

            # Long-term abstention conditions
            should_abstain = False
            abstention_reason = None

            if rag_insufficient or ev_quality < 0.25:
                should_abstain = True
                abstention_reason = "LONG_TERM_ABSTAIN: Insufficient verifiable fundamental RAG evidence or filing excerpts."
            elif fundamental_quality_score < 35.0:
                should_abstain = True
                abstention_reason = f"LONG_TERM_ABSTAIN: Fundamental business quality score ({fundamental_quality_score:.1f}/100) falls below investment threshold."
            elif crash_probability >= 0.60:
                should_abstain = True
                abstention_reason = f"LONG_TERM_ABSTAIN: Systemic crash risk probability ({crash_probability:.1%}) exceeds safe accumulation ceiling."
            elif risk_status == "REJECT" or risk_violations_count >= 2:
                should_abstain = True
                abstention_reason = "LONG_TERM_ABSTAIN: Fundamental portfolio or asset-level risk limits breached."

            return HorizonRoutingResult(
                horizon_type=htype,
                requested_horizon=str(horizon),
                prioritized_agents=prioritized,
                supporting_agents=supporting,
                routing_rationale=rationale,
                long_term_confidence=lt_conf,
                short_term_confidence=None,
                should_abstain=should_abstain,
                abstention_reason=abstention_reason,
            )

        elif htype == HorizonType.SHORT_TERM:
            prioritized = [
                "TechnicalAgent",
                "MLPredictionAgent",
                "SentimentAgent",
                "MarketRegimeAgent",
                "CrashRiskAgent",
                "PositionSizer",
                "DeterministicRiskEngine",
            ]
            supporting = ["FinancialRAGAgent", "CriticAgent"]
            rationale = (
                f"Short-term tactical horizon ({horizon}) routes primary weight to Technical Momentum, "
                f"Short-term ML Direction, Volatility Regime, Event Sentiment, and Hard Stop-Loss Risk Sizing."
            )

            # Short-term confidence formulation
            sig_conf = min(1.0, max(0.1, technical_confidence))
            pred_conf = min(1.0, max(0.1, ml_probability))
            mkt_stability = min(1.0, max(0.1, 1.0 - vol_percentile))
            ev_quality = min(1.0, max(0.1, rag_evidence_quality))
            risk_conf = 1.0 if risk_status == "PASS" and risk_violations_count == 0 else 0.35

            composite = (
                sig_conf * 0.30 +
                pred_conf * 0.25 +
                mkt_stability * 0.15 +
                ev_quality * 0.15 +
                risk_conf * 0.15
            )
            st_conf = ShortTermConfidence(
                signal_confidence=round(sig_conf, 3),
                prediction_confidence=round(pred_conf, 3),
                market_stability=round(mkt_stability, 3),
                evidence_quality=round(ev_quality, 3),
                risk_confidence=round(risk_conf, 3),
                composite_short_term_confidence=round(composite, 3),
            )

            # Short-term abstention conditions
            should_abstain = False
            abstention_reason = None

            if sig_conf < 0.35 and abs(ml_probability - 0.5) < 0.04:
                should_abstain = True
                abstention_reason = "SHORT_TERM_ABSTAIN: Weak directional momentum and ML prediction neutrality."
            elif vol_percentile >= 0.90:
                should_abstain = True
                abstention_reason = f"SHORT_TERM_ABSTAIN: Extreme short-term volatility percentile ({vol_percentile:.1%}) exceeds tactical execution limits."
            elif crash_probability >= 0.55:
                should_abstain = True
                abstention_reason = f"SHORT_TERM_ABSTAIN: Elevated tail-risk/crash probability ({crash_probability:.1%}) triggers tactical risk gate."
            elif risk_status == "REJECT":
                should_abstain = True
                abstention_reason = "SHORT_TERM_ABSTAIN: Deterministic risk engine rejected short-term trade parameters."
            elif composite < 0.40:
                should_abstain = True
                abstention_reason = f"SHORT_TERM_ABSTAIN: Multi-dimensional short-term confidence ({composite:.1%}) below execution threshold."

            return HorizonRoutingResult(
                horizon_type=htype,
                requested_horizon=str(horizon),
                prioritized_agents=prioritized,
                supporting_agents=supporting,
                routing_rationale=rationale,
                long_term_confidence=None,
                short_term_confidence=st_conf,
                should_abstain=should_abstain,
                abstention_reason=abstention_reason,
            )

        else:  # COMBINED
            prioritized = [
                "PortfolioOptimizer",
                "DeterministicRiskEngine",
                "CombinedPortfolioManager",
                "MarketRegimeAgent",
                "FinancialRAGAgent",
                "CrashRiskAgent",
            ]
            supporting = ["TechnicalAgent", "MLPredictionAgent", "SentimentAgent"]
            rationale = "Combined Portfolio routes across both horizons to compute total portfolio risk, allocation drift, and cross-asset correlation."
            return HorizonRoutingResult(
                horizon_type=htype,
                requested_horizon="combined",
                prioritized_agents=prioritized,
                supporting_agents=supporting,
                routing_rationale=rationale,
                long_term_confidence=None,
                short_term_confidence=None,
                should_abstain=False,
                abstention_reason=None,
            )

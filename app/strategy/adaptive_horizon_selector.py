"""Adaptive Investment Horizon Selector.

Assesses whether an asset or opportunity is appropriate for:
- LONG_TERM_CANDIDATE
- SHORT_TERM_CANDIDATE
- BOTH_CANDIDATE
- NEITHER
- INSUFFICIENT_DATA

Reuses existing fundamental analysis, Financial RAG, technical indicators,
ML directional predictions, crash-risk analysis, and confidence measures
without duplicating underlying agents or bypassing the deterministic risk engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from app.orchestrator import TradingOrchestrator
from app.rag.rag_agent import FinancialRAGAgent
from app.strategy.long_term_planner import LongTermPlanner
from app.strategy.short_term_planner import ShortTermPlanner
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HorizonClassification(str, Enum):
    LONG_TERM_CANDIDATE = "LONG_TERM_CANDIDATE"
    SHORT_TERM_CANDIDATE = "SHORT_TERM_CANDIDATE"
    BOTH_CANDIDATE = "BOTH_CANDIDATE"
    NEITHER = "NEITHER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class HorizonAssessmentResult(BaseModel):
    symbol: str
    recommended_horizon: HorizonClassification
    long_term_score: float = Field(ge=0.0, le=100.0)
    short_term_score: float = Field(ge=0.0, le=100.0)
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    main_risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_components: dict[str, float]
    reversal_conditions: list[str]
    alternative_horizon_rationale: str
    user_override: dict[str, Any] | None = None
    timestamp: str


class AdaptiveInvestmentHorizonSelector:
    """Evaluates multi-horizon characteristics from verified agent pipelines."""

    def __init__(
        self,
        orchestrator: TradingOrchestrator | None = None,
        rag_agent: FinancialRAGAgent | None = None,
        long_term_planner: LongTermPlanner | None = None,
        short_term_planner: ShortTermPlanner | None = None,
    ):
        self.orchestrator = orchestrator or TradingOrchestrator()
        self.rag_agent = rag_agent or FinancialRAGAgent()
        self.long_term_planner = long_term_planner or LongTermPlanner(rag_agent=self.rag_agent)
        self.short_term_planner = short_term_planner or ShortTermPlanner(orchestrator=self.orchestrator)

    def assess(
        self,
        symbol: str,
        user_override: str | None = None,
        override_reason: str | None = None,
    ) -> HorizonAssessmentResult:
        sym = symbol.upper().strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Fetch Long-Term Fundamental Scores
        try:
            lt_scores = self.long_term_planner.evaluate_symbol(sym, horizon="3y")
            lt_composite = lt_scores.composite_long_term_score
            lt_quality = lt_scores.fundamental_quality
            lt_stability = lt_scores.financial_stability
            lt_risk = lt_scores.long_term_risk
            lt_citations = [c.get("document", "") for c in lt_scores.rag_citations if isinstance(c, dict)]
        except Exception as exc:
            logger.warning("Adaptive Horizon: Long-term evaluation failed for %s: %s", sym, exc)
            override_record = None
            if user_override:
                override_record = {
                    "original_recommended": HorizonClassification.INSUFFICIENT_DATA.value,
                    "user_selected": user_override.upper(),
                    "reason": override_reason or "Discretionary trader override",
                    "risk_engine_enforced": True,
                    "timestamp": now_iso,
                }
            return HorizonAssessmentResult(
                symbol=sym,
                recommended_horizon=HorizonClassification.INSUFFICIENT_DATA,
                long_term_score=0.0,
                short_term_score=0.0,
                supporting_evidence=[],
                conflicting_evidence=[f"Failed to retrieve fundamental data: {exc}"],
                main_risks=["Data unavailability prevents fiduciary assessment."],
                confidence=0.0,
                confidence_components={},
                reversal_conditions=["Ingest validated financial filings."],
                alternative_horizon_rationale="Insufficient reliable data.",
                user_override=override_record,
                timestamp=now_iso,
            )

        # 2. Fetch Short-Term Tactical Analysis
        try:
            st_opp = self.short_term_planner.evaluate_tactical_opportunity(sym, horizon="1_week")
            st_conf = st_opp.confidence
            st_signal = st_opp.signal
            ml_prob = st_conf.prediction_confidence if st_conf else 0.5
            sig_conf = st_conf.signal_confidence if st_conf else 0.5
            stab_conf = st_conf.market_stability if st_conf else 0.5
            risk_conf = st_conf.risk_confidence if st_conf else 0.5
            crash_p = 1.0 - risk_conf

            raw_st_score = (
                (1.0 if st_signal == "BUY" else 0.5 if st_signal == "HOLD" else 0.15) * 40.0
                + ml_prob * 30.0
                + sig_conf * 15.0
                + stab_conf * 15.0
            )
            st_score = min(100.0, max(0.0, raw_st_score))
        except Exception as exc:
            logger.warning("Adaptive Horizon: Short-term evaluation failed for %s: %s", sym, exc)
            st_score = 45.0
            st_signal = "HOLD"
            crash_p = 0.20
            st_conf = None

        # 3. Determine Candidate Suitability
        is_lt_suitable = (lt_composite >= 60.0) and (lt_risk != "EXTREME") and (lt_stability >= 50.0)
        is_st_suitable = (st_score >= 55.0) and (st_signal == "BUY") and (crash_p < 0.35)

        if is_lt_suitable and is_st_suitable:
            rec = HorizonClassification.BOTH_CANDIDATE
        elif is_lt_suitable:
            rec = HorizonClassification.LONG_TERM_CANDIDATE
        elif is_st_suitable:
            rec = HorizonClassification.SHORT_TERM_CANDIDATE
        else:
            rec = HorizonClassification.NEITHER

        # 4. Construct Supporting Evidence, Conflicting Signals, and Risks
        supporting: list[str] = []
        conflicting: list[str] = []
        risks: list[str] = []

        if is_lt_suitable:
            supporting.append(
                f"Robust fundamental quality ({lt_quality:.1f}/100) and balance sheet stability ({lt_stability:.1f}/100)."
            )
            if lt_citations:
                supporting.append(f"Supported by verified financial disclosures: {lt_citations[0]}")
        else:
            conflicting.append(
                f"Long-term fundamentals sub-optimal (composite {lt_composite:.1f}/100, risk level: {lt_risk})."
            )

        if is_st_suitable:
            supporting.append(
                f"Constructive short-term momentum and tactical signal ({st_signal}, score {st_score:.1f}/100)."
            )
        else:
            conflicting.append(
                f"Short-term technical/ML posture not compelling (signal: {st_signal}, score: {st_score:.1f}/100)."
            )

        if crash_p > 0.25:
            risks.append(f"Elevated tail crash probability: {crash_p * 100:.1f}%.")
        if lt_risk in {"HIGH", "EXTREME"}:
            risks.append(f"Elevated structural long-term risk: {lt_risk}.")
        if not risks:
            risks.append("Normal market volatility and macroeconomic beta exposure.")

        # 5. Composite Confidence
        conf_components = {
            "fundamental_quality": round(lt_quality / 100.0, 3),
            "financial_stability": round(lt_stability / 100.0, 3),
            "tactical_confidence": round((st_conf.composite_short_term_confidence if st_conf else 0.5), 3),
            "risk_confidence": round(1.0 - crash_p, 3),
        }
        overall_conf = round(
            conf_components["fundamental_quality"] * 0.35
            + conf_components["financial_stability"] * 0.25
            + conf_components["tactical_confidence"] * 0.20
            + conf_components["risk_confidence"] * 0.20,
            3,
        )

        # 6. Reversal Conditions & Alternative Horizon Rationale
        reversals = [
            "Earnings report with >10% EPS revision would alter long-term valuation score.",
            "Break below 50-day moving average or RSI < 40 would invalidate short-term tactical setup.",
            "Downside volatility spike > 40% would force immediate risk-engine abstention.",
        ]

        if rec == HorizonClassification.LONG_TERM_CANDIDATE:
            alt_rationale = (
                f"{sym} exhibits high balance-sheet quality and earnings durability, but short-term "
                f"momentum is currently unconfirmed or overextended (score {st_score:.1f}/100). "
                f"Accumulation on value is favored over short-term swing entry."
            )
        elif rec == HorizonClassification.SHORT_TERM_CANDIDATE:
            alt_rationale = (
                f"{sym} exhibits strong tactical momentum and positive ML directional skew, but "
                f"long-term valuation or financial stability scores ({lt_composite:.1f}/100) are "
                f"insufficient for multi-year buy-and-hold."
            )
        elif rec == HorizonClassification.BOTH_CANDIDATE:
            alt_rationale = (
                f"{sym} concurrently satisfies multi-year fundamental quality hurdles and short-term "
                f"tactical entry conditions. Suitable for both core holding and tactical scaling."
            )
        else:
            alt_rationale = (
                f"{sym} does not meet minimum quality thresholds for multi-year holding ({lt_composite:.1f}/100) "
                f"nor does it exhibit positive tactical momentum ({st_score:.1f}/100)."
            )

        # 7. User Override Tracking
        override_record = None
        if user_override:
            override_record = {
                "original_recommended": rec.value,
                "user_selected": user_override.upper(),
                "reason": override_reason or "Discretionary trader override",
                "risk_engine_enforced": True,
                "timestamp": now_iso,
            }
            logger.info("Horizon override recorded for %s: %s -> %s", sym, rec.value, user_override)

        return HorizonAssessmentResult(
            symbol=sym,
            recommended_horizon=rec,
            long_term_score=round(lt_composite, 1),
            short_term_score=round(st_score, 1),
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            main_risks=risks,
            confidence=overall_conf,
            confidence_components=conf_components,
            reversal_conditions=reversals,
            alternative_horizon_rationale=alt_rationale,
            user_override=override_record,
            timestamp=now_iso,
        )

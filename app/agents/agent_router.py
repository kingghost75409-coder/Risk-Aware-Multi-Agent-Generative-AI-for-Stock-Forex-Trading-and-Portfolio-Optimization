"""Dynamic Agent Router for Multi-Agent Trading Orchestration.

Research concept: Fixed pipelines execute every agent regardless of market context,
wasting compute and generating noisy, conflicting signals when assumptions are violated.

The Dynamic Agent Router inspects:
  - Market Regime (Trend & Volatility classification)
  - Realized Volatility Percentile
  - News Velocity & Volume
  - Data Validation Status
  - Market Events & Earnings Proximity

It outputs:
  - An active agent execution plan (nodes in the LangGraph-style state graph)
  - Explicit routing justification for full auditability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)

AgentName = Literal[
    "technical_agent",
    "ml_prediction_agent",
    "crash_risk_agent",
    "sentiment_agent",
    "regime_agent",
    "temporal_rag_agent",
    "knowledge_graph_agent",
    "critic_agent",
    "risk_engine",
    "stress_testing_agent",
    "adaptive_risk_budget",
    "counterfactual_service",
]


class RoutingDecision(BaseModel):
    symbol: str
    selected_agents: list[str]
    deactivated_agents: list[str]
    routing_reason: str
    market_context: dict
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DynamicAgentRouter:
    """Selects and schedules specialized agents dynamically based on market state."""

    ALL_AVAILABLE_AGENTS = [
        "technical_agent",
        "ml_prediction_agent",
        "crash_risk_agent",
        "sentiment_agent",
        "regime_agent",
        "temporal_rag_agent",
        "knowledge_graph_agent",
        "critic_agent",
        "risk_engine",
        "stress_testing_agent",
        "adaptive_risk_budget",
        "counterfactual_service",
    ]

    def route(
        self,
        symbol: str,
        vol_percentile: float,
        trend_slope: float,
        n_news_items: int,
        is_data_clean: bool = True,
        force_full_pipeline: bool = False,
    ) -> RoutingDecision:
        """Determines active agent nodes for LangGraph state machine execution."""
        selected: list[str] = []
        reasons: list[str] = []

        if not is_data_clean:
            # Degraded data: route only to minimal risk check & abstention
            selected = ["risk_engine"]
            return RoutingDecision(
                symbol=symbol,
                selected_agents=selected,
                deactivated_agents=[a for a in self.ALL_AVAILABLE_AGENTS if a not in selected],
                routing_reason="Degraded data validation: minimal risk gate activated to force safe hold/abstain.",
                market_context={"clean_data": False},
            )

        if force_full_pipeline:
            return RoutingDecision(
                symbol=symbol,
                selected_agents=list(self.ALL_AVAILABLE_AGENTS),
                deactivated_agents=[],
                routing_reason="Full pipeline requested explicitly for comprehensive audit/research.",
                market_context={"forced": True},
            )

        # Baseline: Core quantitative agents always participate
        selected.extend([
            "technical_agent",
            "regime_agent",
            "risk_engine",
            "adaptive_risk_budget",
        ])
        reasons.append("Activated core quantitative anchors (Technical, Regime, Adaptive Risk Budget, Risk Engine)")

        # Tail-Risk & Crash Protection: Activate CrashRiskAgent on elevated vol or negative trend
        if vol_percentile >= 0.60 or trend_slope < 0.0:
            selected.append("crash_risk_agent")
            reasons.append(
                f"Tail-risk threshold triggered (vol percentile {vol_percentile:.2f} >= 0.60 or "
                f"trend slope {trend_slope:+.3f} < 0): activated CrashRiskAgent for tail-risk modeling."
            )

        # High Volatility Regime: Volatility > 65th percentile
        is_high_vol = vol_percentile >= 0.65
        if is_high_vol:
            selected.extend([
                "ml_prediction_agent",
                "sentiment_agent",
                "stress_testing_agent",
                "critic_agent",
                "counterfactual_service",
            ])
            reasons.append(
                f"High volatility detected (vol percentile {vol_percentile:.2f} >= 0.65): "
                "activated Stress Testing, ML Prediction, Sentiment, Critic, and Counterfactual agents."
            )
        else:
            # Moderate / Normal Volatility
            selected.extend([
                "ml_prediction_agent",
                "critic_agent",
                "counterfactual_service",
            ])
            reasons.append(
                f"Normal/low volatility (vol percentile {vol_percentile:.2f} < 0.65): "
                "activated standard ML Prediction, Critic, and Counterfactual agents."
            )

        # News & Qualitative Activity
        if n_news_items > 0:
            selected.extend(["sentiment_agent", "temporal_rag_agent", "knowledge_graph_agent"])
            reasons.append(
                f"News & filing activity present ({n_news_items} items): "
                "activated Temporal RAG, Financial Knowledge Graph, and Sentiment Agent."
            )
        else:
            # Still run Knowledge Graph & Temporal RAG if not already included for baseline context
            if "temporal_rag_agent" not in selected:
                selected.append("temporal_rag_agent")
            if "knowledge_graph_agent" not in selected:
                selected.append("knowledge_graph_agent")

        # Deduplicate while preserving sequence
        seen = set()
        deduped_selected = []
        for a in selected:
            if a not in seen and a in self.ALL_AVAILABLE_AGENTS:
                seen.add(a)
                deduped_selected.append(a)

        deactivated = [a for a in self.ALL_AVAILABLE_AGENTS if a not in seen]

        return RoutingDecision(
            symbol=symbol,
            selected_agents=deduped_selected,
            deactivated_agents=deactivated,
            routing_reason=" | ".join(reasons),
            market_context={
                "vol_percentile": round(vol_percentile, 3),
                "trend_slope": round(trend_slope, 4),
                "n_news_items": n_news_items,
                "is_high_vol": is_high_vol,
            },
        )

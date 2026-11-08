"""Multi-Agent Decision Layer / Trading Decision Engine.

Flow conforming to advanced research specifications:
  Market Data -> Data Validation -> Feature Engineering -> Dynamic Agent Router ->
  Selected Specialized Agents -> Technical Analysis -> ML Prediction -> Sentiment Analysis ->
  Market Regime Detection -> Temporal Financial RAG / Graph RAG -> Multi-Agent Decision ->
  Self-Critique -> Multi-Dimensional Confidence -> Uncertainty Evaluation (ABSTAIN) ->
  Adaptive Risk Budget -> Risk Engine (Deterministic Gate) -> Position Sizing ->
  Counterfactual Explanation -> Audit Trail.

SAFETY RULE: The deterministic Risk Engine is independent from LLM decisions.
If the Risk Engine fails or uncertainty is high, the final decision becomes HOLD or ABSTAIN.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field

from app.agents.agent_router import DynamicAgentRouter, RoutingDecision
from app.agents.alpha_engine import AdaptiveAlphaEngine, AlphaSignalResult
from app.agents.confidence_engine import MultiDimensionalConfidenceEngine, ConfidenceDimensions
from app.agents.crash_risk_agent import CrashRiskAgent, CrashRiskResult
from app.agents.critic_agent import CriticAgent, CritiqueReport
from app.agents.ml_agent import MLPredictionAgent, MLPrediction
from app.agents.regime_agent import MarketRegimeAgent, RegimeResult
from app.agents.sentiment_agent import SentimentAgent, SentimentResult
from app.agents.stress_testing_agent import MarketStressTestingAgent, StressTestSuiteReport
from app.agents.technical_agent import TechnicalAgent, TechnicalSignal
from app.agents.uncertainty import UncertaintyEvaluator, AbstentionEvaluation
from app.data.news_sample import get_sample_headlines
from app.data.pipeline import load_and_prepare
from app.data.temporal_firewall import TemporalFirewall
from app.portfolio.optimizer import PositionSizer, PositionSize
from app.rag.knowledge_graph import FinancialKnowledgeGraph, SubgraphContext, get_financial_knowledge_graph
from app.rag.rag_agent import FinancialRAGAgent, RAGResult
from app.risk.adaptive_risk_budget import AdaptiveRiskBudget, AdaptiveRiskBudgetResult
from app.risk.risk_engine import RiskEngine, RiskCheckResult
from app.services.counterfactual import CounterfactualService, CounterfactualExplanation
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TradingDecision(BaseModel):
    symbol: str
    final_decision: str  # BUY / SELL / HOLD / ABSTAIN
    confidence: float
    position_size_pct: float
    data_source: str
    technical: TechnicalSignal
    prediction: MLPrediction
    sentiment: SentimentResult
    regime: RegimeResult
    rag_evidence: RAGResult
    risk: RiskCheckResult
    position_sizing: PositionSize
    reasoning_audit_trail: list[str]
    timestamp: str

    # Advanced Research Upgrades
    abstention_reason: str | None = None
    selected_agents: list[str] = Field(default_factory=list)
    routing_reason: str = ""
    confidence_dimensions: ConfidenceDimensions | None = None
    subgraph_context: SubgraphContext | None = None
    critic_report: CritiqueReport | None = None
    adaptive_risk_budget: AdaptiveRiskBudgetResult | None = None
    counterfactual_explanation: CounterfactualExplanation | None = None
    crash_risk: CrashRiskResult | None = None
    alpha_signal: AlphaSignalResult | None = None
    temporal_audit: str | None = None


class TradingOrchestrator:
    """Owns instances of each agent; runs the dynamically routed, uncertainty-aware,
    temporally grounded multi-agent pipeline and returns a fully audited TradingDecision."""

    def __init__(self) -> None:
        self.technical_agent = TechnicalAgent()
        self.ml_agent = MLPredictionAgent()
        self.sentiment_agent = SentimentAgent()
        self.regime_agent = MarketRegimeAgent()
        self.rag_agent = FinancialRAGAgent()
        self.risk_engine = RiskEngine()
        self.position_sizer = PositionSizer()

        # Advanced research modules
        self.agent_router = DynamicAgentRouter()
        self.confidence_engine = MultiDimensionalConfidenceEngine()
        self.uncertainty_evaluator = UncertaintyEvaluator()
        self.knowledge_graph = get_financial_knowledge_graph()
        self.critic_agent = CriticAgent()
        self.adaptive_risk_budget = AdaptiveRiskBudget()
        self.counterfactual_service = CounterfactualService()
        self.stress_testing_agent = MarketStressTestingAgent()
        self.crash_risk_agent = CrashRiskAgent()
        self.alpha_engine = AdaptiveAlphaEngine()

    def analyze_symbol(
        self,
        symbol: str,
        current_portfolio_exposure_pct: float = 0.0,
        current_daily_pnl_pct: float = 0.0,
        as_of_date: str | None = None,
        force_full_pipeline: bool = False,
    ) -> TradingDecision:
        symbol = symbol.upper()
        audit: list[str] = []

        # 1. Market Data Loading & Validation
        df, validation_report, source = load_and_prepare(symbol)
        audit.append(
            f"DATA: fetched {validation_report.rows_out} clean bars for {symbol} (source={source}); "
            f"{len(validation_report.warnings)} validation warning(s)."
        )

        if not validation_report.is_usable:
            return self._hold_on_insufficient_data(symbol, source, audit)

        # 1b. Temporal Financial Evidence Firewall Enforcement (Upgrade 1)
        firewall = TemporalFirewall(as_of_date=as_of_date)
        df = firewall.enforce_ohlcv(df, symbol)
        audit.append(f"TEMPORAL FIREWALL: {firewall.get_audit_summary()}")

        # Preliminary volatility & regime proxies for dynamic routing
        close = df["adj_close"]
        returns = close.pct_change().dropna()
        realized_vol_20 = float(returns.rolling(20).std().iloc[-1] * (252 ** 0.5)) if len(returns) >= 20 else 0.20
        vol_history = returns.rolling(20).std().dropna() * (252 ** 0.5)
        vol_percentile = float((vol_history < realized_vol_20).mean()) if len(vol_history) else 0.50
        sma20 = close.rolling(20).mean()
        trend_slope = float((sma20.iloc[-1] / sma20.iloc[-10] - 1)) if len(sma20.dropna()) > 10 else 0.0

        headlines = get_sample_headlines(symbol)

        # 2. Dynamic Agent Router
        routing = self.agent_router.route(
            symbol=symbol,
            vol_percentile=vol_percentile,
            trend_slope=trend_slope,
            n_news_items=len(headlines),
            is_data_clean=validation_report.is_usable,
            force_full_pipeline=force_full_pipeline,
        )
        audit.append(f"DYNAMIC ROUTER: selected {len(routing.selected_agents)} agents -> {', '.join(routing.selected_agents)}")
        audit.append(f"DYNAMIC ROUTER REASON: {routing.routing_reason}")

        # 3. Specialized Agents Execution
        technical = self.technical_agent.analyze(df, symbol)
        audit.append(f"TECHNICAL AGENT: signal={technical.signal} confidence={technical.confidence} — {technical.reasoning}")

        prediction = self.ml_agent.analyze(df, symbol)
        audit.append(
            f"PREDICTION AGENT: direction={prediction.predicted_direction} "
            f"probability={prediction.probability} model={prediction.model} "
            f"test_accuracy={prediction.validation_metrics.get('test_accuracy')}"
        )

        # Tail-Risk & Crash Prediction (Upgrade 2)
        crash_risk = self.crash_risk_agent.analyze(df, symbol)
        audit.append(
            f"CRASH RISK AGENT: crash_prob={crash_risk.crash_probability:.1%} "
            f"expected_downside={crash_risk.expected_downside_pct:.1%} "
            f"risk_tier={crash_risk.risk_tier} tail_score={crash_risk.tail_risk_score:.2f} "
            f"top_drivers={', '.join(crash_risk.top_risk_drivers[:3]) if crash_risk.top_risk_drivers else 'n/a'}"
        )

        sentiment = self.sentiment_agent.analyze(symbol, headlines)
        audit.append(
            f"SENTIMENT AGENT: label={sentiment.label} score={sentiment.sentiment_score} "
            f"trend={sentiment.trend} (n={sentiment.n_headlines} headlines)"
        )

        regime = self.regime_agent.analyze(df, symbol)
        audit.append(f"REGIME AGENT: regime={regime.regime} confidence={regime.confidence}")

        # 4. Financial Knowledge Graph & Hybrid Temporal RAG
        subgraph = self.knowledge_graph.get_subgraph(symbol)
        audit.append(f"KNOWLEDGE GRAPH: {subgraph.graph_summary}")

        rag_evidence = self.rag_agent.get_evidence_summary(symbol, as_of_date=as_of_date)
        audit.append(
            f"TEMPORAL RAG: status={rag_evidence.verification['status']} "
            f"(as_of_date={as_of_date or 'latest'}) citations={len(rag_evidence.citations)} — "
            f"\"{rag_evidence.answer[:120]}...\""
            if len(rag_evidence.answer) > 120
            else f"TEMPORAL RAG: status={rag_evidence.verification['status']} — {rag_evidence.answer}"
        )

        # 5. Adaptive Alpha Signal Engine (Upgrade 3)
        alpha_result = self.alpha_engine.compute_alpha(
            symbol=symbol,
            technical_signal=technical.signal,
            technical_confidence=technical.confidence,
            ml_direction=prediction.predicted_direction,
            ml_probability=prediction.probability,
            sentiment_score=sentiment.sentiment_score,
            rag_verification_status=rag_evidence.verification.get("status", "NO_CLAIM_MADE"),
            rag_evidence_quality=getattr(rag_evidence, "evidence_quality", 0.5),
            rag_insufficient=rag_evidence.insufficient_evidence,
            crash_probability=crash_risk.crash_probability if crash_risk else 0.0,
            regime_label=regime.regime,
        )
        conviction = alpha_result.composite_alpha
        raw_signal = {"BULLISH": "BUY", "BEARISH": "SELL", "NEUTRAL": "HOLD"}.get(
            alpha_result.directional_bias, "HOLD"
        )
        audit.append(
            f"ADAPTIVE ALPHA ENGINE: composite={alpha_result.composite_alpha:+.3f} "
            f"bias={alpha_result.directional_bias} agreement={alpha_result.signal_agreement_pct:.1%} "
            f"profile='{alpha_result.regime_weight_profile}' -> proposed={raw_signal}"
        )

        # 6. Preliminary Risk Assessment for Critic Context
        prelim_risk = self.risk_engine.evaluate(
            symbol=symbol, df=df, proposed_signal=raw_signal,
            proposed_position_pct=0.05,
            current_portfolio_exposure_pct=current_portfolio_exposure_pct,
            current_daily_pnl_pct=current_daily_pnl_pct,
        )

        # 7. Self-Critiquing Trading Agent (Devil's Advocate Stage)
        critic = self.critic_agent.critique(
            symbol=symbol,
            proposed_decision=raw_signal,
            conviction_score=conviction,
            technical=technical,
            prediction=prediction,
            sentiment=sentiment,
            regime=regime,
            rag_evidence=rag_evidence,
            risk=prelim_risk,
        )
        audit.append(
            f"CRITIC AGENT: verdict={critic.critic_verdict} (severity={critic.critic_severity:.2f}) "
            f"— {len(critic.identified_weaknesses)} weakness(es) identified: {'; '.join(critic.identified_weaknesses) if critic.identified_weaknesses else 'none'}"
        )

        intermediate_decision = critic.suggested_decision if critic.critic_verdict in {"REVISE_TO_HOLD", "REVISE_TO_ABSTAIN"} else raw_signal

        # Tail-Risk Circuit Breaker: If crash probability >= 65%, suppress BUY to HOLD
        if crash_risk and crash_risk.crash_probability >= 0.65 and intermediate_decision == "BUY":
            intermediate_decision = "HOLD"
            audit.append(
                f"TAIL-RISK CIRCUIT BREAKER: crash probability {crash_risk.crash_probability:.1%} "
                f">= 65% critical threshold -> forced BUY to HOLD."
            )

        # 8. Multi-Dimensional Confidence Engine (5 dimensions)
        confidence_dims = self.confidence_engine.calculate(
            technical_confidence=technical.confidence,
            ml_probability=prediction.probability,
            rag_verification_status=rag_evidence.verification.get("status", "UNSUPPORTED"),
            rag_citations_count=len(rag_evidence.citations),
            regime_confidence=regime.confidence,
            vol_percentile=vol_percentile,
            risk_status=prelim_risk.status,
            risk_violations=prelim_risk.violations,
            var_95=prelim_risk.var_95,
            realized_vol=realized_vol_20,
        )
        audit.append(
            f"CONFIDENCE DIMENSIONS: composite={confidence_dims.composite_confidence:.3f} "
            f"[Signal={confidence_dims.signal_confidence:.2f}, Prediction={confidence_dims.prediction_confidence:.2f}, "
            f"Evidence={confidence_dims.evidence_quality:.2f}, Stability={confidence_dims.market_stability:.2f}, "
            f"Risk={confidence_dims.risk_confidence:.2f}]"
        )

        # 9. Uncertainty-Aware Abstention Layer
        abstention = self.uncertainty_evaluator.evaluate(
            raw_decision=intermediate_decision,
            ml_probability=prediction.probability,
            confidence_dims=confidence_dims,
            insufficient_rag_evidence=rag_evidence.insufficient_evidence,
        )
        post_uncertainty_decision = abstention.decision
        if abstention.should_abstain:
            audit.append(f"UNCERTAINTY ABSTENTION: {abstention.abstention_reason}")

        # 10. Adaptive Risk Budget Module
        risk_budget = self.adaptive_risk_budget.calculate_budget(
            symbol=symbol,
            regime_label=regime.regime,
            vol_percentile=vol_percentile,
            current_daily_loss_pct=current_daily_pnl_pct,
        )
        audit.append(
            f"ADAPTIVE RISK BUDGET: multiplier={risk_budget.risk_budget_multiplier:.2f}x "
            f"max_position={risk_budget.adjusted_max_position_pct:.1%} "
            f"cash_allocation={risk_budget.recommended_cash_allocation_pct:.1%} "
            f"({risk_budget.regime_rationale})"
        )

        # 11. Position Sizing (constrained by Adaptive Risk Budget)
        avg_loss_magnitude = float(df["adj_close"].pct_change().clip(upper=0).abs().tail(60).mean() or 0.02)
        base_sizing = self.position_sizer.size_position(
            symbol=symbol,
            win_probability=prediction.probability if prediction.predicted_direction == "UP" else 1 - prediction.probability,
            expected_return=prediction.expected_return if prediction.expected_return != 0 else 0.001,
            avg_loss_magnitude=max(avg_loss_magnitude, 0.005),
            conviction_score=conviction,
        )

        # Scale position by adaptive budget multiplier
        scaled_position_pct = min(
            base_sizing.recommended_position_pct * risk_budget.risk_budget_multiplier,
            risk_budget.adjusted_max_position_pct,
        )
        proposed_position_pct = round(scaled_position_pct, 4) if post_uncertainty_decision == "BUY" else 0.0
        audit.append(f"POSITION SIZING: recommended={proposed_position_pct:.1%} (scaled by budget multiplier {risk_budget.risk_budget_multiplier:.2f}x)")

        # 12. Final Deterministic Risk Engine Evaluation (Hard Inviolable Gate)
        final_risk = self.risk_engine.evaluate(
            symbol=symbol,
            df=df,
            proposed_signal=post_uncertainty_decision if post_uncertainty_decision in {"BUY", "SELL", "HOLD"} else "HOLD",
            proposed_position_pct=proposed_position_pct,
            current_portfolio_exposure_pct=current_portfolio_exposure_pct,
            current_daily_pnl_pct=current_daily_pnl_pct,
        )
        audit.append(
            f"RISK ENGINE: status={final_risk.status} VaR95={final_risk.var_95} CVaR95={final_risk.cvar_95} "
            f"max_dd={final_risk.max_drawdown} vol={final_risk.volatility_annualized}"
            + (f" | VIOLATIONS: {'; '.join(final_risk.violations)}" if final_risk.violations else " | no violations")
        )

        # Apply hard gate override
        if final_risk.status == "REJECT":
            final_decision = "HOLD"
            final_position_pct = 0.0
            audit.append("FINAL OVERRIDE: Risk Engine REJECTED trade -> forced HOLD (inviolable risk gate).")
        elif post_uncertainty_decision == "ABSTAIN":
            final_decision = "ABSTAIN"
            final_position_pct = 0.0
            audit.append(f"FINAL: ABSTAIN ({abstention.abstention_reason})")
        else:
            final_decision = post_uncertainty_decision
            final_position_pct = proposed_position_pct
            audit.append(f"FINAL: {final_decision} (position size {final_position_pct:.1%})")

        # 13. Counterfactual Trading Explanation
        counterfactual = self.counterfactual_service.generate_explanation(
            symbol=symbol,
            current_decision=final_decision,
            conviction_score=conviction,
            technical_signal=technical.signal,
            technical_confidence=technical.confidence,
            ml_direction=prediction.predicted_direction,
            ml_probability=prediction.probability,
            sentiment_score=sentiment.sentiment_score,
            regime_label=regime.regime,
            cvar_95=final_risk.cvar_95,
            vol_percentile=vol_percentile,
            confidence_dims=confidence_dims,
        )
        audit.append(f"COUNTERFACTUAL: {counterfactual.summary_text.splitlines()[0]}")

        return TradingDecision(
            symbol=symbol,
            final_decision=final_decision,
            confidence=confidence_dims.composite_confidence,
            position_size_pct=round(final_position_pct, 4),
            data_source=source,
            technical=technical,
            prediction=prediction,
            sentiment=sentiment,
            regime=regime,
            rag_evidence=rag_evidence,
            risk=final_risk,
            position_sizing=base_sizing,
            reasoning_audit_trail=audit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            abstention_reason=abstention.abstention_reason if final_decision == "ABSTAIN" else None,
            selected_agents=routing.selected_agents,
            routing_reason=routing.routing_reason,
            confidence_dimensions=confidence_dims,
            subgraph_context=subgraph,
            critic_report=critic,
            adaptive_risk_budget=risk_budget,
            counterfactual_explanation=counterfactual,
            crash_risk=crash_risk,
            alpha_signal=alpha_result,
            temporal_audit=firewall.get_audit_summary(),
        )

    @staticmethod
    def _combine_signals(
        technical: TechnicalSignal,
        prediction: MLPrediction,
        sentiment: SentimentResult,
        regime: RegimeResult,
    ) -> tuple[float, str]:
        """Deterministic weighted combination -> conviction in [-1, 1].

        Weights: technical 0.30, ML prediction 0.35, sentiment 0.15, regime 0.20.
        """
        tech_component = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(technical.signal, 0.0) * technical.confidence
        ml_component = (1.0 if prediction.predicted_direction == "UP" else -1.0) * (prediction.probability - 0.5) * 2
        sentiment_component = sentiment.sentiment_score
        regime_component = 0.5 if regime.trend_component == "BULL" else (-0.5 if regime.trend_component == "BEAR" else 0.0)
        regime_component *= regime.confidence

        conviction = (
            0.30 * tech_component + 0.35 * ml_component + 0.15 * sentiment_component + 0.20 * regime_component
        )
        conviction = max(-1.0, min(1.0, conviction))

        if conviction > 0.15:
            signal = "BUY"
        elif conviction < -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"
        return conviction, signal

    def _hold_on_insufficient_data(self, symbol: str, source: str, audit: list[str]) -> TradingDecision:
        audit.append("FINAL: ABSTAIN — insufficient validated data to run agents (defensive research policy).")
        now = datetime.now(timezone.utc).isoformat()
        neutral_tech = TechnicalSignal(
            symbol=symbol, signal="HOLD", confidence=0.0, indicators={},
            reasoning="insufficient data", timestamp=now
        )
        neutral_pred = MLPrediction(
            symbol=symbol, predicted_direction="UP", probability=0.5, expected_return=0.0,
            model="none", validation_metrics={}, timestamp=now
        )
        neutral_sent = SentimentResult(
            symbol=symbol, sentiment_score=0.0, label="NEUTRAL", confidence=0.0,
            trend="STABLE", n_headlines=0, headline_scores=[], timestamp=now
        )
        neutral_regime = RegimeResult(
            symbol=symbol, regime="UNKNOWN", confidence=0.0, trend_component="SIDEWAYS",
            volatility_component="NORMAL_VOLATILITY", details={}, timestamp=now
        )
        neutral_rag = RAGResult(
            symbol=symbol, query="", answer="Insufficient evidence in the knowledge base.",
            citations=[], structured_fact=None, verification={"status": "NO_CLAIM_MADE"},
            insufficient_evidence=True, timestamp=now
        )
        neutral_risk = RiskCheckResult(
            symbol=symbol, approved=True, status="PASS", var_95=0.0, cvar_95=0.0,
            max_drawdown=0.0, volatility_annualized=0.0, position_limit_pct=0.0,
            proposed_position_pct=0.0, concentration_ok=True, stop_loss_price=None,
            violations=["insufficient data"], timestamp=now
        )
        neutral_sizing = PositionSize(
            symbol=symbol, kelly_fraction=0.0, kelly_fraction_capped=0.0,
            black_litterman_tilt_weight=0.0, recommended_position_pct=0.0,
            method_used="none", rationale="insufficient data"
        )
        neutral_conf = ConfidenceDimensions(
            signal_confidence=0.0, prediction_confidence=0.0, evidence_quality=0.0,
            market_stability=0.0, risk_confidence=0.0, composite_confidence=0.0,
        )

        return TradingDecision(
            symbol=symbol, final_decision="ABSTAIN", confidence=0.0, position_size_pct=0.0, data_source=source,
            technical=neutral_tech, prediction=neutral_pred, sentiment=neutral_sent, regime=neutral_regime,
            rag_evidence=neutral_rag, risk=neutral_risk, position_sizing=neutral_sizing,
            reasoning_audit_trail=audit, timestamp=now,
            abstention_reason="ABSTAIN: Insufficient validated market data.",
            selected_agents=["risk_engine"],
            routing_reason="Degraded data validation forced safe abstention.",
            confidence_dimensions=neutral_conf,
            temporal_audit="Temporal firewall: Insufficient clean data to enforce.",
        )

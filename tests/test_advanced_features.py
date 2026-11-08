"""Comprehensive Test Suite for 10 Advanced Research Features.

Tests:
  1. Uncertainty-Aware Abstention
  2. Dynamic Agent Router
  3. Temporal Financial RAG & Look-Ahead Prevention
  4. Counterfactual Trading Explanations
  5. Market Stress Testing Agent
  6. Financial Knowledge Graph
  7. Self-Critiquing Trading Agent
  8. Adversarial Financial RAG Testing
  9. Multi-Dimensional Confidence Engine
  10. Adaptive Risk Budget
  11. End-to-End Orchestrator Integration
  12. REST API Endpoints
"""
import pytest
import numpy as np
import pandas as pd

from app.agents.confidence_engine import MultiDimensionalConfidenceEngine, ConfidenceDimensions
from app.agents.uncertainty import UncertaintyEvaluator
from app.agents.agent_router import DynamicAgentRouter
from app.agents.critic_agent import CriticAgent
from app.agents.stress_testing_agent import MarketStressTestingAgent
from app.agents.crash_risk_agent import CrashRiskAgent
from app.agents.alpha_engine import AdaptiveAlphaEngine
from app.data.temporal_firewall import TemporalFirewall
from app.data.pipeline import load_and_prepare
from app.rag.evidence_verifier import EvidenceVerifier, compute_freshness_score
from app.rag.rag_agent import FinancialRAGAgent, INSUFFICIENT_EVIDENCE
from app.rag.knowledge_graph import FinancialKnowledgeGraph
from app.rag.adversarial_evaluator import AdversarialRAGEvaluator
from app.risk.adaptive_risk_budget import AdaptiveRiskBudget
from app.services.counterfactual import CounterfactualService
from app.orchestrator import TradingOrchestrator
from app.api.app import app


# ─── Feature 1: Uncertainty-Aware Abstention ─────────────────────────────────

def test_uncertainty_abstention_triggers_on_high_entropy():
    evaluator = UncertaintyEvaluator(uncertainty_threshold=0.70)
    # Probability = 0.50 means maximum Shannon entropy (1.0)
    entropy = evaluator.calculate_entropy(0.50)
    assert np.isclose(entropy, 1.0, atol=1e-3)

    conf_dims = ConfidenceDimensions(
        signal_confidence=0.5,
        prediction_confidence=0.1,
        evidence_quality=0.8,
        market_stability=0.7,
        risk_confidence=0.8,
        composite_confidence=0.58,
    )

    result = evaluator.evaluate(
        raw_decision="BUY",
        ml_probability=0.50,
        confidence_dims=conf_dims,
        insufficient_rag_evidence=False,
    )
    assert result.should_abstain is True
    assert result.decision == "ABSTAIN"
    assert "High ML prediction uncertainty" in result.abstention_reason


def test_uncertainty_abstention_triggers_on_insufficient_evidence():
    evaluator = UncertaintyEvaluator(min_evidence_quality=0.30)
    conf_dims = ConfidenceDimensions(
        signal_confidence=0.8,
        prediction_confidence=0.8,
        evidence_quality=0.15,  # below 0.30 threshold
        market_stability=0.8,
        risk_confidence=0.8,
        composite_confidence=0.67,
    )

    result = evaluator.evaluate(
        raw_decision="BUY",
        ml_probability=0.75,
        confidence_dims=conf_dims,
        insufficient_rag_evidence=True,
    )
    assert result.should_abstain is True
    assert result.decision == "ABSTAIN"
    assert "Insufficient financial evidence" in result.abstention_reason


# ─── Feature 2: Dynamic Agent Router ─────────────────────────────────────────

def test_dynamic_agent_router_high_volatility():
    router = DynamicAgentRouter()
    decision = router.route(
        symbol="AAPL",
        vol_percentile=0.85,  # High volatility regime
        trend_slope=-0.02,
        n_news_items=5,
    )
    assert "stress_testing_agent" in decision.selected_agents
    assert "critic_agent" in decision.selected_agents
    assert "sentiment_agent" in decision.selected_agents
    assert "temporal_rag_agent" in decision.selected_agents
    assert "High volatility detected" in decision.routing_reason


def test_dynamic_agent_router_degraded_data_safety():
    router = DynamicAgentRouter()
    decision = router.route(
        symbol="XYZ",
        vol_percentile=0.5,
        trend_slope=0.0,
        n_news_items=0,
        is_data_clean=False,
    )
    # On degraded data, only minimal risk engine is activated to safely force hold/abstain
    assert decision.selected_agents == ["risk_engine"]
    assert "Degraded data" in decision.routing_reason


# ─── Feature 3: Temporal Financial RAG ───────────────────────────────────────

def test_temporal_rag_prevents_look_ahead():
    agent = FinancialRAGAgent()
    # AAPL 10-K was published on 2025-11-01
    # Query with as_of_date prior to publication should return zero citations & insufficient evidence
    res_past = agent.query("AAPL", "risk factors", as_of_date="2025-06-01")
    assert len(res_past.citations) == 0
    assert res_past.insufficient_evidence is True
    assert res_past.answer == INSUFFICIENT_EVIDENCE

    # Query with as_of_date after publication should successfully retrieve the grounded filing
    res_future = agent.query("AAPL", "risk factors", as_of_date="2026-01-01")
    assert len(res_future.citations) > 0
    assert res_future.insufficient_evidence is False
    assert res_future.citations[0].published_at == "2025-11-01"


# ─── Feature 4: Counterfactual Trading Explanations ──────────────────────────

def test_counterfactual_explanation_structure():
    service = CounterfactualService()
    conf_dims = ConfidenceDimensions(
        signal_confidence=0.7,
        prediction_confidence=0.6,
        evidence_quality=0.8,
        market_stability=0.7,
        risk_confidence=0.85,
        composite_confidence=0.73,
    )

    exp = service.generate_explanation(
        symbol="AAPL",
        current_decision="BUY",
        conviction_score=0.35,
        technical_signal="BUY",
        technical_confidence=0.7,
        ml_direction="UP",
        ml_probability=0.68,
        sentiment_score=0.25,
        regime_label="NORMAL_VOLATILITY_BULL",
        cvar_95=0.035,
        vol_percentile=0.40,
        confidence_dims=conf_dims,
    )
    assert exp.current_decision == "BUY"
    assert len(exp.primary_reasons_why) > 0
    assert len(exp.what_would_change_decision) >= 3
    # Check that contrastive shifts include volatility and sentiment
    params = [c.parameter for c in exp.what_would_change_decision]
    assert "Asset Volatility" in params
    assert "News Sentiment" in params


# ─── Feature 5: Market Stress Testing Agent ──────────────────────────────────

def test_stress_testing_agent_invariance():
    agent = MarketStressTestingAgent()
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))), index=dates)
    df = pd.DataFrame({"adj_close": prices})

    report = agent.run_stress_test(
        symbol="AAPL",
        df=df,
        portfolio_value=100_000.0,
        cash=50_000.0,
    )
    assert len(report.scenarios) == 7
    # Baseline cash must remain invariant
    assert report.baseline_cash == 50_000.0
    # Black Swan scenario (-20%) must produce greater loss than -5%
    m5 = next(s for s in report.scenarios if s.scenario_id == "MKT_MINUS_5")
    m20 = next(s for s in report.scenarios if s.scenario_id == "MKT_MINUS_20")
    assert m20.portfolio_loss_amount > m5.portfolio_loss_amount
    assert m20.stressed_var_95 > m5.stressed_var_95


# ─── Feature 6: Financial Knowledge Graph ────────────────────────────────────

def test_financial_knowledge_graph():
    kg = FinancialKnowledgeGraph()
    subgraph = kg.get_subgraph("NVDA")
    assert subgraph.symbol == "NVDA"
    assert "Semiconductors" in subgraph.sector
    # Should identify TSMC as supplier
    assert "TSMC" in subgraph.suppliers
    # Should identify MSFT or AMZN as customer
    assert any(c in subgraph.customers for c in ["MSFT", "AMZN", "TSLA"])
    # Should identify export restrictions as risk
    assert any("Export" in r for r in subgraph.risk_factors)


# ─── Feature 7: Self-Critiquing Trading Agent ────────────────────────────────

def test_critic_agent_detects_severe_contradictions():
    critic = CriticAgent()
    from app.agents.technical_agent import TechnicalSignal
    from app.agents.ml_agent import MLPrediction
    from app.agents.sentiment_agent import SentimentResult
    from app.agents.regime_agent import RegimeResult
    from app.rag.rag_agent import RAGResult
    from app.risk.risk_engine import RiskCheckResult

    now = "2026-01-01T00:00:00Z"
    tech = TechnicalSignal(symbol="AAPL", signal="BUY", confidence=0.8, indicators={"rsi_14": 78.5}, reasoning="momentum", timestamp=now)
    # Bearish ML predicting DOWN
    ml = MLPrediction(symbol="AAPL", predicted_direction="DOWN", probability=0.35, expected_return=-0.02, model="xgboost", validation_metrics={}, timestamp=now)
    sent = SentimentResult(symbol="AAPL", sentiment_score=-0.45, label="NEGATIVE", confidence=0.8, trend="DOWN", n_headlines=5, headline_scores=[], timestamp=now)
    regime = RegimeResult(symbol="AAPL", regime="HIGH_VOLATILITY_BEAR", confidence=0.8, trend_component="BEAR", volatility_component="HIGH_VOLATILITY", details={}, timestamp=now)
    rag = RAGResult(symbol="AAPL", query="", answer="Severe supply disruption.", citations=[], verification={"status": "SUPPORTED"}, timestamp=now)
    risk = RiskCheckResult(symbol="AAPL", approved=True, status="PASS", var_95=0.06, cvar_95=0.08, max_drawdown=0.15, volatility_annualized=0.45, position_limit_pct=0.1, proposed_position_pct=0.05, concentration_ok=True, stop_loss_price=None, violations=[], timestamp=now)

    report = critic.critique(
        symbol="AAPL",
        proposed_decision="BUY",
        conviction_score=0.20,
        technical=tech,
        prediction=ml,
        sentiment=sent,
        regime=regime,
        rag_evidence=rag,
        risk=risk,
    )
    # The critic should challenge or revise this contradicted BUY
    assert report.critic_severity > 0.50
    assert report.critic_verdict in {"REVISE_TO_HOLD", "REVISE_TO_ABSTAIN", "CHALLENGED"}
    assert len(report.identified_weaknesses) >= 3


# ─── Feature 8: Adversarial Financial RAG Testing ────────────────────────────

def test_adversarial_rag_evaluator():
    evaluator = AdversarialRAGEvaluator()
    summary = evaluator.run_benchmark_suite()
    assert summary.total_tests == 5
    assert summary.conflicts_detected >= 2
    assert summary.hallucination_prevention_rate == 1.0


# ─── Feature 9: Multi-Dimensional Confidence Engine ──────────────────────────

def test_multi_dimensional_confidence_engine():
    engine = MultiDimensionalConfidenceEngine()
    dims = engine.calculate(
        technical_confidence=0.80,
        ml_probability=0.70,
        rag_verification_status="SUPPORTED",
        rag_citations_count=3,
        regime_confidence=0.85,
        vol_percentile=0.20,
        risk_status="PASS",
        risk_violations=[],
        var_95=0.02,
    )
    assert 0.0 <= dims.signal_confidence <= 1.0
    assert 0.0 <= dims.prediction_confidence <= 1.0
    assert 0.0 <= dims.evidence_quality <= 1.0
    assert 0.0 <= dims.market_stability <= 1.0
    assert 0.0 <= dims.risk_confidence <= 1.0
    assert 0.0 <= dims.composite_confidence <= 1.0
    assert "Research simulation metric" in dims.disclaimer


# ─── Feature 10: Adaptive Risk Budget ────────────────────────────────────────

def test_adaptive_risk_budget_regime_constriction():
    arb = AdaptiveRiskBudget()
    # In a bear + high volatility regime, multiplier should aggressively contract
    budget_bear = arb.calculate_budget(
        symbol="AAPL",
        regime_label="HIGH_VOLATILITY_BEAR",
        vol_percentile=0.90,
    )
    assert budget_bear.risk_budget_multiplier <= 0.50
    assert budget_bear.adjusted_max_position_pct < budget_bear.base_max_position_pct
    assert budget_bear.recommended_cash_allocation_pct >= 0.70

    # In a calm bull regime, multiplier should stay healthy
    budget_bull = arb.calculate_budget(
        symbol="AAPL",
        regime_label="LOW_VOLATILITY_BULL",
        vol_percentile=0.15,
    )
    assert budget_bull.risk_budget_multiplier > budget_bear.risk_budget_multiplier


# ─── End-to-End Orchestrator Integration ─────────────────────────────────────

def test_orchestrator_end_to_end_decision():
    orch = TradingOrchestrator()
    decision = orch.analyze_symbol("AAPL")
    assert decision.final_decision in {"BUY", "SELL", "HOLD", "ABSTAIN"}
    assert len(decision.selected_agents) > 0
    assert decision.confidence_dimensions is not None
    assert decision.critic_report is not None
    assert decision.adaptive_risk_budget is not None
    assert decision.counterfactual_explanation is not None
    assert len(decision.reasoning_audit_trail) > 5


# ─── Flask REST API Verification ─────────────────────────────────────────────

def test_api_endpoints_health_and_research_features():
    client = app.test_client()

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200

    # Dynamic agents
    res = client.get("/api/agents/AAPL")
    assert res.status_code == 200
    assert "selected_agents" in res.json

    # Multi-dimensional confidence
    res = client.get("/api/confidence/AAPL")
    assert res.status_code == 200
    assert "composite_confidence" in res.json

    # Knowledge graph
    res = client.get("/api/graph/AAPL")
    assert res.status_code == 200
    assert "competitors" in res.json

    # Adaptive risk budget
    res = client.get("/api/risk/budget")
    assert res.status_code == 200
    assert "risk_budget_multiplier" in res.json

    # Stress test
    res = client.get("/api/stress-test?symbol=AAPL")
    assert res.status_code == 200
    assert len(res.json.get("scenarios", [])) == 7

    # Counterfactual
    res = client.get("/api/counterfactual/AAPL")
    assert res.status_code == 200
    assert "current_decision" in res.json

    # RAG evaluate
    res = client.get("/api/rag/evaluate")
    assert res.status_code == 200
    assert res.json.get("total_tests") == 5


# ─── Upgrade 1: Temporal Financial Evidence Firewall ─────────────────────────

def test_temporal_firewall_enforces_as_of_date():
    df, _, _ = load_and_prepare("AAPL")
    assert not df.empty

    firewall = TemporalFirewall(as_of_date="2025-06-01")
    truncated_df = firewall.enforce_ohlcv(df, "AAPL")

    # All timestamps must be <= 2025-06-01
    boundary = pd.Timestamp("2025-06-01", tz="UTC")
    idx = truncated_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    assert (idx <= boundary).all()
    assert len(truncated_df) <= len(df)
    assert not firewall.report.is_clean or len(truncated_df) == len(df)

    # Document filtering test
    docs = [
        {"source": "past", "published_at": "2025-01-15T00:00:00Z"},
        {"source": "future", "published_at": "2025-09-01T00:00:00Z"},
    ]
    accepted = firewall.filter_documents(docs, "AAPL")
    assert len(accepted) == 1
    assert accepted[0]["source"] == "past"


# ─── Upgrade 2: Crash & Tail-Risk Prediction Agent ───────────────────────────

def test_crash_risk_agent_tail_metrics():
    df, report, _ = load_and_prepare("AAPL")
    assert report.is_usable

    agent = CrashRiskAgent()
    result = agent.analyze(df, "AAPL")

    assert 0.0 <= result.crash_probability <= 1.0
    assert 0.0 <= result.tail_risk_score <= 1.0
    assert result.risk_tier in {"LOW", "MODERATE", "ELEVATED", "HIGH", "EXTREME"}
    assert isinstance(result.top_risk_drivers, list)
    assert len(result.feature_importances) > 0
    assert result.prediction_horizon_days == 10


# ─── Upgrade 3: Adaptive Alpha Signal Engine ─────────────────────────────────

def test_adaptive_alpha_engine_weighting_and_regimes():
    engine = AdaptiveAlphaEngine()

    # Bullish scenario in normal market
    result_bull = engine.compute_alpha(
        symbol="AAPL",
        technical_signal="BUY",
        technical_confidence=0.85,
        ml_direction="UP",
        ml_probability=0.75,
        sentiment_score=0.60,
        rag_verification_status="SUPPORTED",
        rag_evidence_quality=0.80,
        crash_probability=0.10,
        regime_label="BULL_LOW_VOL",
    )
    assert result_bull.composite_alpha > 0.0
    assert result_bull.directional_bias == "BULLISH"
    assert 0.0 <= result_bull.signal_agreement_pct <= 1.0

    # Total weights must sum to 1.0
    total_weights = sum(c.weight for c in result_bull.signal_contributions)
    assert pytest.approx(total_weights, abs=1e-3) == 1.0

    # In a crisis regime (BEAR_HIGH_VOL), crash risk must receive higher weight than in BULL_LOW_VOL
    result_crisis = engine.compute_alpha(
        symbol="AAPL",
        technical_signal="SELL",
        technical_confidence=0.70,
        ml_direction="DOWN",
        ml_probability=0.80,
        sentiment_score=-0.70,
        crash_probability=0.85,
        regime_label="BEAR_HIGH_VOL",
    )
    crash_w_bull = next(c.weight for c in result_bull.signal_contributions if c.signal_name == "crash_risk")
    crash_w_crisis = next(c.weight for c in result_crisis.signal_contributions if c.signal_name == "crash_risk")
    assert crash_w_crisis > crash_w_bull


# ─── Upgrade 4 & 5: Evidence Verification & Freshness ────────────────────────

def test_evidence_verifier_and_freshness():
    verifier = EvidenceVerifier()
    text = (
        "Apple Inc reported quarterly revenue of 94.9 billion dollars. "
        "Services revenue reached an all-time record of 25.0 billion. "
        "The company expects gross margin between 46 and 47 percent."
    )
    claims = verifier.extract_claims(text)
    assert len(claims) >= 2

    # Freshness scoring
    fresh_score = compute_freshness_score("2026-03-01", as_of_date="2026-03-15")
    old_score = compute_freshness_score("2024-01-01", as_of_date="2026-03-15")
    assert fresh_score > old_score
    assert 0.0 <= fresh_score <= 1.0


# ─── Upgrade 6: Counterfactual Numerical Sensitivity ─────────────────────────

def test_counterfactual_numerical_sensitivity():
    svc = CounterfactualService()
    cf = svc.generate_explanation(
        symbol="AAPL",
        current_decision="BUY",
        conviction_score=0.45,
        technical_signal="BUY",
        technical_confidence=0.80,
        ml_direction="UP",
        ml_probability=0.72,
        sentiment_score=0.55,
        regime_label="BULL_LOW_VOL",
        cvar_95=0.035,
        vol_percentile=0.30,
        confidence_dims=ConfidenceDimensions(
            signal_confidence=0.8, prediction_confidence=0.75,
            evidence_quality=0.8, market_stability=0.7,
            risk_confidence=0.85, composite_confidence=0.78,
        ),
    )
    assert cf.has_sensitivity_metrics
    assert len(cf.flip_conditions) >= 3
    for cond in cf.flip_conditions:
        if cond.distance_to_flip is not None:
            assert cond.distance_to_flip >= 0.0


# ─── Upgrade 7: Historical & Custom Stress Testing ───────────────────────────

def test_historical_and_custom_stress_testing():
    agent = MarketStressTestingAgent()
    df, _, _ = load_and_prepare("AAPL")

    # Historical crisis replay
    hist_report = agent.run_historical_stress("AAPL", df)
    assert len(hist_report.scenarios) == 3
    scenario_ids = [s.scenario_id for s in hist_report.scenarios]
    assert "HIST_GFC_2008" in scenario_ids
    assert "HIST_COVID_2020" in scenario_ids
    assert "HIST_RATE_2022" in scenario_ids

    # Custom stress scenario
    custom_report = agent.run_custom_stress(
        symbol="AAPL",
        df=df,
        price_shock_pct=-0.25,
        vol_multiplier=2.5,
        sentiment_override=-0.90,
        scenario_name="Bespoke Black Swan",
    )
    assert any(s.scenario_name == "Bespoke Black Swan" for s in custom_report.scenarios)


# ─── Research Endpoints API Verification ─────────────────────────────────────

def test_new_research_api_endpoints():
    client = app.test_client()

    # Crash risk
    res = client.get("/api/crash-risk/AAPL")
    assert res.status_code == 200
    assert "crash_probability" in res.json
    assert "tail_risk_score" in res.json

    # Alpha engine
    res = client.get("/api/alpha/AAPL")
    assert res.status_code == 200

    # Historical stress test
    res = client.get("/api/stress-test/historical/AAPL")
    assert res.status_code == 200
    assert len(res.json.get("scenarios", [])) == 3

    # Custom stress test (POST)
    res = client.post("/api/stress-test/custom", json={
        "symbol": "AAPL",
        "price_shock": -0.20,
        "vol_multiplier": 2.0,
        "scenario_name": "API Test Shock",
    })
    assert res.status_code == 200
    assert any(s.get("scenario_name") == "API Test Shock" for s in res.json.get("scenarios", []))

    # Firewall audit
    res = client.get("/api/firewall/audit?symbol=AAPL&as_of_date=2025-06-01")
    assert res.status_code == 200
    assert "accepted_bars" in res.json


# ─── Orchestrator Full Upgrades Integration ──────────────────────────────────

def test_orchestrator_decision_includes_all_upgrades():
    orch = TradingOrchestrator()
    decision = orch.analyze_symbol("AAPL", as_of_date="2025-10-01")

    assert decision.final_decision in {"BUY", "SELL", "HOLD", "ABSTAIN"}
    assert decision.alpha_signal is not None
    assert decision.alpha_signal.composite_alpha is not None
    assert decision.temporal_audit is not None
    assert "Temporal firewall" in decision.temporal_audit
    assert decision.counterfactual_explanation is not None
    assert any("ALPHA ENGINE" in line for line in decision.reasoning_audit_trail)


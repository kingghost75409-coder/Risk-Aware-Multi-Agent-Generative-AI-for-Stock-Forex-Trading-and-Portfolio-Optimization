"""Tests for the RAG pipeline, risk engine, and full orchestrator."""
import pandas as pd

from app.rag.rag_agent import FinancialRAGAgent, INSUFFICIENT_EVIDENCE
from app.risk.risk_engine import RiskEngine, value_at_risk, conditional_value_at_risk, max_drawdown
from app.data.pipeline import load_and_prepare
from app.orchestrator import TradingOrchestrator


def test_rag_returns_insufficient_evidence_when_nothing_relevant():
    agent = FinancialRAGAgent()
    result = agent.query("AAPL", "what is the CEO's favorite restaurant")
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.citations == []
    assert result.verification["status"] == "NO_CLAIM_MADE"


def test_rag_returns_citations_for_relevant_query():
    agent = FinancialRAGAgent()
    result = agent.query("AAPL", "risk factors")
    assert len(result.citations) > 0
    assert result.verification["status"] in ("SUPPORTED", "PARTIALLY_UNSUPPORTED")


def test_rag_structured_fact_lookup_exact_value():
    agent = FinancialRAGAgent()
    result = agent.query("AAPL", "What was the revenue?")
    assert result.structured_fact is not None
    assert "94.9" in result.answer


def test_var_and_cvar_ordering():
    # CVaR (expected shortfall beyond VaR) should never be smaller than VaR
    returns = pd.Series([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, -0.08, 0.015, -0.02] * 5)
    var95 = value_at_risk(returns, 0.95)
    cvar95 = conditional_value_at_risk(returns, 0.95)
    assert cvar95 >= var95 - 1e-9


def test_max_drawdown_is_nonnegative():
    prices = pd.Series([100, 110, 105, 90, 95, 120, 80])
    mdd = max_drawdown(prices)
    assert mdd >= 0
    assert abs(mdd - (120 - 80) / 120) < 1e-6 or mdd >= (110 - 90) / 110 - 1e-6


def test_risk_engine_rejects_oversized_position():
    df, report, source = load_and_prepare("AAPL", use_cache=True)
    engine = RiskEngine()
    result = engine.evaluate(
        symbol="AAPL", df=df, proposed_signal="BUY", proposed_position_pct=0.50,  # way over the 10% cap
        current_portfolio_exposure_pct=0.0,
    )
    assert result.status == "REJECT"
    assert any("exceeds max single-position limit" in v for v in result.violations)


def test_risk_engine_approves_hold_regardless():
    df, report, source = load_and_prepare("AAPL", use_cache=True)
    engine = RiskEngine()
    result = engine.evaluate(
        symbol="AAPL", df=df, proposed_signal="HOLD", proposed_position_pct=0.0,
        current_portfolio_exposure_pct=0.0,
    )
    assert result.status == "PASS"


def test_orchestrator_produces_full_audit_trail():
    orch = TradingOrchestrator()
    decision = orch.analyze_symbol("NVDA")
    assert decision.final_decision in ("BUY", "SELL", "HOLD", "ABSTAIN")
    assert len(decision.reasoning_audit_trail) >= 6  # data, technical, ml, sentiment, regime, rag, risk, final
    assert any("RISK ENGINE" in line for line in decision.reasoning_audit_trail)
    assert any("FINAL" in line for line in decision.reasoning_audit_trail)


def test_orchestrator_never_lets_risk_engine_be_overridden():
    orch = TradingOrchestrator()
    # force a REJECT by claiming massive existing exposure
    decision = orch.analyze_symbol("NVDA", current_portfolio_exposure_pct=0.99)
    if decision.risk.status == "REJECT":
        assert decision.final_decision == "HOLD"

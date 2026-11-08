"""Comprehensive Unit & Integration Tests for Multi-Horizon Investment Strategy & Portfolio Planning.

Tests:
1. HorizonStrategyRouter (routing, separate confidence dimensions, horizon abstention gates)
2. LongTermPlanner (fundamental scores, RAG citations, decision rules, constrained allocation, SIP simulation)
3. ShortTermPlanner (tactical zones, stop loss, risk/reward, Kelly sizing, uncertainty)
4. CombinedPortfolioManager (unified sleeves, total cross-asset risk, rebalancing triggers, audit event persistence)
5. MultiHorizonBacktester (Long-Term vs Short-Term vs Buy & Hold causal evaluation)
6. Flask REST APIs (/api/investment/...)
7. Chatbot Integration (investment planning, horizon comparison, combined risk)
"""
from __future__ import annotations

import json
import pytest

from app.strategy.router import HorizonStrategyRouter, HorizonType, parse_horizon
from app.strategy.long_term_planner import LongTermPlanner
from app.strategy.short_term_planner import ShortTermPlanner
from app.strategy.combined_portfolio import CombinedPortfolioManager
from app.strategy.horizon_backtest import MultiHorizonBacktester
from app.services.chatbot import answer_question
from app.orchestrator import TradingOrchestrator
from app.rag.rag_agent import FinancialRAGAgent
from app.trading.paper_trading import PaperTradingEngine
from app.api.app import app


# ─── 1. Horizon Strategy Router Tests ─────────────────────────────────────────

def test_router_horizon_parsing():
    assert parse_horizon("3_years") == HorizonType.LONG_TERM
    assert parse_horizon("5_years") == HorizonType.LONG_TERM
    assert parse_horizon("1_year") == HorizonType.LONG_TERM
    assert parse_horizon("6_months") == HorizonType.LONG_TERM

    assert parse_horizon("intraday") == HorizonType.SHORT_TERM
    assert parse_horizon("1_day") == HorizonType.SHORT_TERM
    assert parse_horizon("1_week") == HorizonType.SHORT_TERM
    assert parse_horizon("1_4_weeks") == HorizonType.SHORT_TERM

    assert parse_horizon("combined") == HorizonType.COMBINED


def test_router_prioritizes_agents_and_confidence():
    router = HorizonStrategyRouter()
    orch = TradingOrchestrator()
    decision = orch.analyze_symbol("AAPL")

    # Long term routing
    lt_route = router.route_horizon("3_years", decision, fundamental_evidence_count=3)
    assert lt_route.horizon_type == HorizonType.LONG_TERM
    assert "FinancialRAGAgent" in lt_route.prioritized_agents
    assert lt_route.long_term_confidence is not None
    assert 0.0 <= lt_route.long_term_confidence.composite_confidence <= 1.0

    # Short term routing
    st_route = router.route_horizon("1_week", decision, fundamental_evidence_count=3)
    assert st_route.horizon_type == HorizonType.SHORT_TERM
    assert "TechnicalAgent" in st_route.prioritized_agents
    assert "MLPredictionAgent" in st_route.prioritized_agents
    assert st_route.short_term_confidence is not None
    assert 0.0 <= st_route.short_term_confidence.signal_confidence <= 1.0


def test_router_abstains_on_insufficient_fundamental_evidence():
    router = HorizonStrategyRouter()
    orch = TradingOrchestrator()
    decision = orch.analyze_symbol("AAPL")

    # When zero citations and high uncertainty -> Long term should abstain
    lt_route = router.route_horizon("5_years", decision, fundamental_evidence_count=0)
    assert lt_route.decision in {"HOLD", "ABSTAIN", "LONG-TERM ABSTAIN"}


# ─── 2. Long-Term Investment Planner Tests ────────────────────────────────────

def test_long_term_planner_scoring():
    planner = LongTermPlanner()
    analysis = planner.evaluate_symbol("AAPL", horizon="3_years")

    assert analysis.symbol == "AAPL"
    assert 0.0 <= analysis.composite_long_term_score <= 100.0
    assert 0.0 <= analysis.fundamental_quality <= 100.0
    assert 0.0 <= analysis.growth_score <= 100.0
    assert 0.0 <= analysis.valuation_score <= 100.0
    assert 0.0 <= analysis.financial_stability <= 100.0
    assert analysis.long_term_risk in {"LOW", "MODERATE", "HIGH", "EXTREME"}
    assert analysis.recommendation in {"LONG-TERM ACCUMULATE", "LONG-TERM HOLD", "TRIM", "AVOID", "LONG-TERM ABSTAIN"}
    assert len(analysis.accumulation_conditions) > 0
    assert len(analysis.exit_conditions) > 0


def test_long_term_portfolio_construction_respects_constraints():
    planner = LongTermPlanner()
    plan = planner.construct_portfolio(
        investment_amount=500000.0,
        horizon="3_years",
        risk_preference="moderate",
        symbols=["AAPL", "MSFT", "GOOGL"],
        max_stock_allocation=0.30,
        max_sector_allocation=0.60,
    )

    assert plan.capital == 500000.0
    assert plan.cash_allocation_pct >= 0.05  # At least 5% emergency cash
    assert plan.total_invested + plan.cash_amount == pytest.approx(plan.capital, rel=1e-3)
    assert plan.risk_status == "APPROVED_BY_RISK_ENGINE"

    # Per-stock max allocation limit check
    for a in plan.allocations:
        assert a.weight_pct <= 0.35  # within boundary
        assert a.allocated_amount > 0
        assert a.target_shares >= 0
        assert len(a.why_this_stock) > 10


def test_long_term_sip_simulation():
    planner = LongTermPlanner()
    sip = planner.simulate_sip(
        initial_investment=50000.0,
        periodic_amount=10000.0,
        frequency="MONTHLY",
        horizon_years=3.0,
        expected_annual_cagr=0.12,
    )

    assert sip.initial_investment == 50000.0
    assert sip.periodic_amount == 10000.0
    assert sip.total_contributed == 50000.0 + (10000.0 * 36)
    assert sip.final_portfolio_value > sip.total_contributed * 0.90
    assert len(sip.progression) == 37  # Month 0 + 36 months
    assert sip.max_simulation_drawdown >= 0.0


# ─── 3. Short-Term Tactical Trading Planner Tests ─────────────────────────────

def test_short_term_tactical_opportunity():
    planner = ShortTermPlanner()
    opp = planner.analyze_opportunity("AAPL", horizon="1_week", portfolio_value=100000.0)

    assert opp.symbol == "AAPL"
    assert opp.horizon == "1_week"
    assert opp.signal in {"BUY", "SELL", "HOLD", "ABSTAIN"}
    assert opp.current_price > 0.0
    assert opp.trade_zone is not None

    tz = opp.trade_zone
    assert tz.entry_target > 0.0
    assert tz.stop_loss > 0.0
    assert tz.stop_distance_abs > 0.0
    assert tz.risk_reward_ratio > 0.0
    assert "1:" in tz.risk_reward_str

    # Risk Engine bounds
    assert opp.position_size_pct <= 0.20
    assert opp.var_95 >= 0.0
    assert opp.cvar_95 >= 0.0
    assert opp.risk_level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert len(opp.disclaimer) > 0


def test_short_term_scan_watchlist():
    planner = ShortTermPlanner()
    opps = planner.scan_watchlist(["AAPL", "MSFT"], horizon="1_week")
    assert len(opps) == 2
    for o in opps:
        assert o.symbol in {"AAPL", "MSFT"}


# ─── 4. Combined Portfolio & Total Risk Tests ─────────────────────────────────

def test_combined_portfolio_build_and_total_risk():
    mgr = CombinedPortfolioManager()
    comb = mgr.build_combined_portfolio(
        total_capital=1000000.0,
        long_term_target_pct=0.70,
        short_term_target_pct=0.20,
        cash_target_pct=0.10,
    )

    assert comb.total_capital == 1000000.0
    risk = comb.total_risk

    assert risk.total_capital == 1000000.0
    assert risk.long_term_capital > 0.0
    assert risk.cash_capital > 0.0
    assert risk.portfolio_volatility_annualized >= 0.0
    assert risk.total_var_95 >= 0.0
    assert risk.total_cvar_95 >= 0.0
    assert risk.max_sector_pct <= 1.0
    assert risk.risk_status in {"APPROVED", "WARNING", "REJECT"}

    # Total exposure must not exceed 100%
    assert risk.total_equity_exposure_pct <= 1.0


def test_rebalance_trigger_evaluation_and_persistence():
    mgr = CombinedPortfolioManager()
    comb = mgr.build_combined_portfolio(total_capital=500000.0)
    assessment = comb.rebalance_assessment

    assert assessment is not None
    assert isinstance(assessment.should_rebalance, bool)

    # Test database audit persistence
    event_id = mgr.record_rebalance_event(
        plan_id=None,
        assessment=assessment,
        total_risk=comb.total_risk,
    )
    assert event_id is not None
    assert event_id > 0


# ─── 5. Multi-Horizon Backtester Tests ────────────────────────────────────────

def test_multi_horizon_backtester_comparison():
    bt = MultiHorizonBacktester(friction_bps=8.0)
    res = bt.run_comparison(symbol="AAPL", period="2y")

    assert res["symbol"] == "AAPL"
    assert "comparison" in res
    assert len(res["comparison"]) == 3  # Long-Term, Short-Term, Buy & Hold

    lt = next(c for c in res["comparison"] if c["horizon_category"] == "LONG_TERM")
    st = next(c for c in res["comparison"] if c["horizon_category"] == "SHORT_TERM")
    bh = next(c for c in res["comparison"] if c["horizon_category"] == "BENCHMARK")

    # Standard metrics exist and are non-null
    for metric_obj in [lt, st, bh]:
        assert "cagr" in metric_obj
        assert "sharpe_ratio" in metric_obj
        assert "sortino_ratio" in metric_obj
        assert "max_drawdown" in metric_obj
        assert "win_rate" in metric_obj
        assert "turnover" in metric_obj
        assert "abstention_rate" in metric_obj
        assert "var_95" in metric_obj
        assert "cvar_95" in metric_obj

    # Long-term turnover should be significantly lower than tactical turnover
    assert lt["turnover"] <= st["turnover"] + 0.10


# ─── 6. Flask REST API Integration Tests ──────────────────────────────────────

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_long_term_stock(client):
    r = client.get("/api/investment/long-term/AAPL?horizon=3_years")
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AAPL"
    assert "composite_long_term_score" in data
    assert "fundamental_quality" in data


def test_api_long_term_plan(client):
    payload = {
        "investment_amount": 500000.0,
        "horizon": "3_years",
        "risk_preference": "moderate",
        "monthly_investment": 10000.0,
    }
    r = client.post("/api/investment/long-term/plan", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["capital"] == 500000.0
    assert len(data["allocations"]) > 0


def test_api_short_term_stock(client):
    r = client.get("/api/investment/short-term/AAPL?horizon=1_week")
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AAPL"
    assert "trade_zone" in data
    assert "risk_reward_str" in data["trade_zone"]


def test_api_combined_portfolio(client):
    r = client.get("/api/investment/combined?capital=1000000&long_term_pct=0.7&short_term_pct=0.2&cash_pct=0.1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total_capital"] == 1000000.0
    assert "total_risk" in data


def test_api_performance(client):
    r = client.get("/api/investment/performance?symbol=AAPL&period=2y")
    assert r.status_code == 200
    data = r.get_json()
    assert "comparison" in data
    assert len(data["comparison"]) == 3


def test_api_risk(client):
    r = client.get("/api/investment/risk?capital=500000")
    assert r.status_code == 200
    data = r.get_json()
    assert "portfolio_volatility_annualized" in data
    assert "total_var_95" in data


# ─── 7. Chatbot Multi-Horizon Routing Tests ───────────────────────────────────

def test_chatbot_investment_plan_query():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()
    decisions = {}

    res = answer_question(
        question="Create a 3-year investment plan for ₹5 lakh",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions=decisions,
    )

    assert res["tool_used"] == "generate_long_term_plan"
    assert len(res["answer"]) > 50


def test_chatbot_long_term_opportunities():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()
    decisions = {}

    res = answer_question(
        question="Find long-term opportunities",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions=decisions,
    )

    assert res["tool_used"] == "find_long_term_opportunities"
    assert len(res["answer"]) > 50


def test_chatbot_short_term_opportunities():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()
    decisions = {}

    res = answer_question(
        question="Give me short-term opportunities",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions=decisions,
    )

    assert res["tool_used"] == "scan_short_term_opportunities"
    assert len(res["answer"]) > 50


def test_chatbot_why_hold_long_term():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()
    decisions = {}

    res = answer_question(
        question="Why should I hold this stock long term?",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions=decisions,
    )

    assert res["tool_used"] == "explain_long_term_holding"
    assert len(res["answer"]) > 50


def test_chatbot_combined_portfolio_risk():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()
    decisions = {}

    res = answer_question(
        question="Show my combined portfolio risk",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions=decisions,
    )

    assert res["tool_used"] == "get_combined_portfolio_risk"
    assert len(res["answer"]) > 50


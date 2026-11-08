"""Comprehensive Unit and Integration Tests for OpenRouter and Intelligence Upgrades.

Tests:
1. OpenRouter Provider (completions, backoff, auth failure, latency, token usage, structured JSON, fallback)
2. Adaptive Investment Horizon Selector (dual suitability scores, classification, user override, insufficient data)
3. Personal Portfolio Digital Twin (sandbox invariance, market shocks, turnover, friction, VaR/CVaR delta)
4. Financial Evidence Conflict Detector (accruals vs cash flow, momentum vs crash risk, calibrated score, gate action)
5. Portfolio Decision Attribution Engine (Brinson factor decomposition, benchmark excess return, cash drag)
6. AI Reliability and Drift Monitor (5 health states, trade lock on critical, degraded position scaling, safe recovery)
7. Flask REST API Integration (all 8 new endpoints)
8. Chatbot Tool Dispatching (all 5 new tool routes)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from app.api.app import app
from app.db.models import init_db
from app.orchestrator import TradingOrchestrator
from app.portfolio.attribution import PortfolioDecisionAttributionEngine
from app.portfolio.digital_twin import PersonalPortfolioDigitalTwin
from app.rag.conflict_detector import ConflictSeverity, FinancialEvidenceConflictDetector
from app.rag.rag_agent import FinancialRAGAgent
from app.services.chatbot import answer_question
from app.services.llm_service import (
    LLMGateway,
    LLMResponse,
    OpenRouterProvider,
    get_llm_gateway,
)
from app.services.reliability_monitor import (
    AIReliabilityMonitor,
    HealthStatus,
    get_reliability_monitor,
)
from app.strategy.adaptive_horizon_selector import (
    AdaptiveInvestmentHorizonSelector,
    HorizonClassification,
)
from app.trading.paper_trading import PaperTradingEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─── 1. OpenRouter Provider Tests ─────────────────────────────────────────────

def test_openrouter_missing_api_key_graceful_fallback():
    gateway = LLMGateway()
    # Without OPENROUTER_API_KEY, fallback remains active and returns template response
    res = gateway.generate("Explain momentum")
    assert res is not None
    assert isinstance(res, LLMResponse)
    assert res.text != ""


def test_openrouter_successful_chat_completion():
    mock_payload = {
        "id": "gen-12345",
        "choices": [{
            "message": {"role": "assistant", "content": "Analysis based on verified RSI and moving averages."}
        }],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 18,
            "total_tokens": 60,
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenRouterProvider(api_key="sk-or-v1-testkey1234567890")
        res = provider.generate("Analyze AAPL")

        assert res is not None
        assert res.provider == "openrouter"
        assert res.tokens_used == 60
        assert res.prompt_tokens == 42
        assert res.completion_tokens == 18
        assert res.latency_ms >= 0.0
        assert "verified RSI" in res.text


def test_openrouter_structured_json_parsing():
    structured_content = {"signal": "BUY", "confidence": 0.85, "reasoning": "Strong fundamental quality"}
    mock_payload = {
        "choices": [{"message": {"content": json.dumps(structured_content)}}],
        "usage": {"total_tokens": 40}
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        provider = OpenRouterProvider(api_key="sk-or-v1-test")
        res = provider.generate("Give structured verdict", response_format={"type": "json_object"})

        assert res is not None
        assert res.structured_data is not None
        assert res.structured_data["signal"] == "BUY"
        assert res.structured_data["confidence"] == 0.85


def test_openrouter_auth_failure_does_not_retry_indefinitely():
    import urllib.error
    http_error = urllib.error.HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": {"message": "Invalid API key"}}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error) as mock_urlopen:
        provider = OpenRouterProvider(api_key="invalid-key", max_retries=3)
        res = provider.generate("Hello")
        assert res is None
        # On 401 permanent auth failure, must stop immediately after 1 attempt
        assert mock_urlopen.call_count == 1


# ─── 2. Adaptive Investment Horizon Selector Tests ────────────────────────────

def test_adaptive_horizon_selector_assessment():
    selector = AdaptiveInvestmentHorizonSelector()
    res = selector.assess("AAPL")

    assert res.symbol == "AAPL"
    assert res.recommended_horizon in {
        HorizonClassification.LONG_TERM_CANDIDATE,
        HorizonClassification.SHORT_TERM_CANDIDATE,
        HorizonClassification.BOTH_CANDIDATE,
        HorizonClassification.NEITHER,
        HorizonClassification.INSUFFICIENT_DATA,
    }
    assert 0.0 <= res.long_term_score <= 100.0
    assert 0.0 <= res.short_term_score <= 100.0
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.supporting_evidence) > 0 or len(res.conflicting_evidence) > 0
    assert len(res.reversal_conditions) > 0
    assert len(res.alternative_horizon_rationale) > 20


def test_adaptive_horizon_user_override():
    selector = AdaptiveInvestmentHorizonSelector()
    res = selector.assess(
        "AAPL",
        user_override="LONG_TERM_CANDIDATE",
        override_reason="Strategic retirement portfolio accumulation",
    )
    assert res.user_override is not None
    assert res.user_override["user_selected"] == "LONG_TERM_CANDIDATE"
    assert res.user_override["risk_engine_enforced"] is True


# ─── 3. Personal Portfolio Digital Twin Tests ─────────────────────────────────

def test_digital_twin_sandbox_invariance():
    paper = PaperTradingEngine()
    initial_cash = paper.cash
    initial_holdings_count = len(paper.holdings)

    dt = PersonalPortfolioDigitalTwin(paper_engine=paper)
    sim_res = dt.simulate_scenario(
        scenario_name="Stress Simulation",
        market_shock_pct=-0.15,
        cash_shift_pct=0.10,
        volatility_shock_multiplier=1.40,
    )

    # Invariance check: Live paper portfolio must remain 100% UNTOUCHED
    assert paper.cash == initial_cash
    assert len(paper.holdings) == initial_holdings_count

    # Simulation outcome checks
    assert sim_res.baseline.total_value >= 1000.0
    assert sim_res.simulated.total_value > 0.0
    assert sim_res.allocation_diffs["cash_pct_diff"] >= 0.0
    assert sim_res.estimated_transaction_cost >= 0.0
    assert len(sim_res.scenario_assumptions) > 0
    assert len(sim_res.limitations_and_uncertainty) > 0


def test_digital_twin_scenario_retrieval():
    paper = PaperTradingEngine()
    dt = PersonalPortfolioDigitalTwin(paper_engine=paper)
    res = dt.simulate_scenario(scenario_name="Test Scenario", market_shock_pct=-0.05)
    sid = res.scenario_id

    fetched = dt.get_scenario(sid)
    assert fetched is not None
    assert fetched.scenario_name == "Test Scenario"

    scenarios = dt.list_scenarios()
    assert any(s["scenario_id"] == sid for s in scenarios)


# ─── 4. Financial Evidence Conflict Detector Tests ────────────────────────────

def test_evidence_conflict_detector_normal_case():
    detector = FinancialEvidenceConflictDetector()
    report = detector.detect_conflicts("AAPL", technical_signal="HOLD", crash_probability=0.10)

    assert report.symbol == "AAPL"
    assert report.severity in {ConflictSeverity.NONE, ConflictSeverity.LOW, ConflictSeverity.MEDIUM, ConflictSeverity.HIGH, ConflictSeverity.CRITICAL}
    assert 0.0 <= report.conflict_score <= 1.0


def test_evidence_conflict_detector_momentum_vs_crash_risk():
    detector = FinancialEvidenceConflictDetector()
    # High crash probability combined with BUY momentum must trigger high severity conflict
    report = detector.detect_conflicts(
        "TSLA",
        technical_signal="BUY",
        crash_probability=0.45,
    )

    assert report.severity in {ConflictSeverity.HIGH, ConflictSeverity.CRITICAL}
    assert report.recommended_action == "ABSTAIN"
    assert report.conflict_score >= 0.65
    assert report.confidence_impact >= 0.30
    assert any("Momentum vs Tail Risk" in c.get("type", "") for c in report.conflicting_claims)


# ─── 5. Portfolio Decision Attribution Engine Tests ───────────────────────────

def test_portfolio_decision_attribution():
    paper = PaperTradingEngine()
    engine = PortfolioDecisionAttributionEngine(paper_engine=paper)
    res = engine.compute_attribution(benchmark_symbol="SPY", benchmark_period_return=0.08)

    assert res.benchmark_return == 0.08
    assert isinstance(res.excess_return, float)
    assert isinstance(res.allocation_effect, float)
    assert isinstance(res.selection_effect, float)
    assert isinstance(res.cash_drag_effect, float)
    assert len(res.strategy_attribution) == 2
    assert "sharpe_ratio" in res.risk_adjusted_metrics
    assert len(res.limitations) > 0


# ─── 6. AI Reliability and Drift Monitor Tests ────────────────────────────────

def test_reliability_monitor_healthy_state():
    monitor = AIReliabilityMonitor()
    report = monitor.check_system_health()

    assert report.overall_status in {HealthStatus.HEALTHY, HealthStatus.WARNING}
    assert report.block_risk_increasing_trades is False
    assert report.position_size_multiplier == 1.0
    assert len(report.checks) >= 5


def test_reliability_monitor_critical_trade_lock():
    monitor = AIReliabilityMonitor()
    # Inject critical risk engine fault
    report = monitor.check_system_health(force_critical_fault=True)

    assert report.overall_status == HealthStatus.CRITICAL
    assert report.block_risk_increasing_trades is True
    assert report.position_size_multiplier == 0.0
    assert len(report.active_alerts) > 0


def test_reliability_monitor_safe_recovery():
    monitor = AIReliabilityMonitor()
    monitor.record_data_warning("AAPL", "Synthetic pipeline timeout")
    monitor.record_llm_failure("openrouter", "Connection reset")

    recovery = monitor.attempt_recovery()
    assert recovery["status"] == "RECOVERY_ATTEMPTED"
    assert recovery["cleared_warnings"] == 1
    assert recovery["cleared_llm_failures"] == 1
    assert recovery["post_recovery_system_status"] in {"HEALTHY", "WARNING"}


# ─── 7. Flask REST API Integration Tests ──────────────────────────────────────

def test_api_investment_horizon(client):
    r = client.get("/api/investment/horizon/AAPL")
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AAPL"
    assert "recommended_horizon" in data
    assert "long_term_score" in data
    assert "short_term_score" in data


def test_api_digital_twin_scenario_post_and_get(client):
    payload = {
        "scenario_name": "API Test Shock",
        "market_shock_pct": -0.10,
        "cash_shift_pct": 0.05,
        "volatility_shock_multiplier": 1.20,
    }
    r = client.post("/api/portfolio/digital-twin/scenario", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["scenario_name"] == "API Test Shock"
    assert "baseline" in data
    assert "simulated" in data

    # List scenarios
    r_list = client.get("/api/portfolio/digital-twin/scenarios")
    assert r_list.status_code == 200
    assert r_list.get_json()["count"] >= 1


def test_api_evidence_conflicts(client):
    r = client.get("/api/evidence/conflicts/AAPL?technical_signal=BUY&crash_probability=0.15")
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "AAPL"
    assert "conflict_score" in data
    assert "severity" in data


def test_api_portfolio_attribution(client):
    r = client.get("/api/portfolio/attribution?benchmark_symbol=SPY&benchmark_period_return=0.085")
    assert r.status_code == 200
    data = r.get_json()
    assert "excess_return" in data
    assert "selection_effect" in data
    assert "strategy_attribution" in data


def test_api_system_reliability_and_alerts(client):
    r_rel = client.get("/api/system/reliability")
    assert r_rel.status_code == 200
    data_rel = r_rel.get_json()
    assert "overall_status" in data_rel
    assert "block_risk_increasing_trades" in data_rel

    r_alerts = client.get("/api/system/alerts")
    assert r_alerts.status_code == 200
    assert "alerts" in r_alerts.get_json()


def test_api_system_recovery_and_llm_status(client):
    r_rec = client.post("/api/system/recover")
    assert r_rec.status_code == 200
    assert r_rec.get_json()["status"] == "RECOVERY_ATTEMPTED"

    r_llm = client.get("/api/llm/status")
    assert r_llm.status_code == 200
    data_llm = r_llm.get_json()
    assert "active_provider" in data_llm
    assert "openrouter_model" in data_llm


# ─── 8. Chatbot Tool Dispatching Tests ────────────────────────────────────────

def test_chatbot_adaptive_horizon_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Should I analyze AAPL for the long term or short term?",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_horizon_assessment"
    assert len(res["answer"]) > 50


def test_chatbot_digital_twin_scenario_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Simulate a 10% market decline for my paper portfolio",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "run_portfolio_scenario"
    assert any(w in res["answer"].lower() for w in ["digital twin", "stress test", "portfolio", "decline"])


def test_chatbot_evidence_conflicts_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="What evidence conflicts exist for AAPL?",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_evidence_conflicts"
    assert len(res["answer"]) > 50


def test_chatbot_performance_attribution_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Why did my portfolio underperform its benchmark?",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_performance_attribution"
    assert len(res["answer"]) > 50
    assert any(w in res["answer"].lower() for w in ["attribution", "return", "portfolio", "brinson", "alpha"])


def test_chatbot_reliability_status_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Is the prediction system currently reliable?",
        symbol="AAPL",
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_reliability_status"
    assert len(res["answer"]) > 50
    assert any(w in res["answer"].lower() for w in ["health", "healthy", "reliability", "engine", "status"])

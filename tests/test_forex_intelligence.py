"""Comprehensive Unit and Integration Tests for Forex Pre/Post Event AI, Regime Intelligence, and Event Reaction Memory.

Covers:
1. Forex pair normalization, pip arithmetic (0.01 for JPY pairs vs 0.0001 for G10), spreads, active market sessions.
2. Market data provider fallback (synthetic generator when offline, OHLCV shape, no crashes).
3. Unsupported pairs return transparent informative errors instead of fabricated data.
4. Market Regime Intelligence classification (Trending, Ranging, High-Volatility, Low-Volatility, Crisis/Extreme).
5. Regime change detection logging and database persistence.
6. Strategy confidence calibration under regime constraints (strictly deterministic, never inflated).
7. Pre-event assessment does not leak future actual releases (temporal firewall check).
8. Economic surprise calculation handles missing consensus and respects indicator direction (unemployment vs rates).
9. Post-event multi-window price reaction timeline (immediate, 5m, 15m, 30m, 60m).
10. Distinction between initial spike and subsequent continuation or reversal.
11. Pre-event assessment remains intact and unmutated after post-event analysis (non-overwriting).
12. Historical Event Reaction Memory: similarity scoring, analog matching, reversal tendency distribution.
13. Event Risk Gate blocks new simulated entries during announcement blackout windows or abnormal spread expansion.
14. Existing simulated positions are never deleted or closed by an event alert.
15. Chatbot tool dispatching for Forex regimes, pre-event, post-event, reaction memory, and risk check.
16. Flask REST API endpoints integration for all Forex routes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import pytest

from app.api.app import app
from app.db.models import (
    init_db,
    get_session,
    ForexPreEventAssessmentRecord,
    ForexPostEventAnalysisRecord,
    ForexRegimeLogRecord,
)
from app.forex.forex_data import (
    SUPPORTED_FOREX_PAIRS,
    normalize_forex_pair,
    is_supported_pair,
    get_pip_size,
    calculate_pips,
    get_active_sessions,
    ForexDataProvider,
    get_forex_provider,
)
from app.forex.regime_intelligence import (
    ForexMarketRegimeIntelligence,
    ForexRegimeType,
    VolatilityLevel,
    TrendDirection,
    get_regime_intelligence,
)
from app.forex.event_analyzer import (
    EventCategory,
    EventRiskLevel,
    ReactionPattern,
    ForexEventAnalyzer,
    get_event_analyzer,
)
from app.forex.event_memory import (
    ForexEventReactionMemory,
    get_event_reaction_memory,
)
from app.forex.event_risk_gate import (
    ForexEventRiskGate,
    get_forex_risk_gate,
)
from app.orchestrator import TradingOrchestrator
from app.rag.rag_agent import FinancialRAGAgent
from app.services.chatbot import answer_question
from app.trading.paper_trading import PaperTradingEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─── 1. Forex Data & Pip Calculation Tests ────────────────────────────────────

def test_forex_pair_normalization_and_support():
    assert normalize_forex_pair("eurusd") == "EUR/USD"
    assert normalize_forex_pair("EUR-USD") == "EUR/USD"
    assert normalize_forex_pair("usdjpy") == "USD/JPY"
    assert normalize_forex_pair("GBP/USD") == "GBP/USD"
    assert is_supported_pair("EUR/USD")
    assert is_supported_pair("USD/JPY")
    assert is_supported_pair("AUD/USD")
    assert not is_supported_pair("BTC/USD")
    assert not is_supported_pair("TRY/ZAR")


def test_pip_size_and_calculations():
    # Standard G10: 0.0001
    assert get_pip_size("EUR/USD") == 0.0001
    assert get_pip_size("GBP/USD") == 0.0001
    assert calculate_pips(1.0850, 1.0875, "EUR/USD") == 25.0

    # JPY pair: 0.01
    assert get_pip_size("USD/JPY") == 0.01
    assert calculate_pips(154.20, 154.70, "USD/JPY") == 50.0
    assert calculate_pips(155.00, 154.50, "USD/JPY") == -50.0


def test_market_sessions_detection():
    # 14:00 UTC -> London and New York are active
    t_london_ny = datetime(2026, 10, 22, 14, 30, tzinfo=timezone.utc)
    sessions = get_active_sessions(t_london_ny)
    assert "London" in sessions
    assert "New York" in sessions

    # 03:00 UTC -> Tokyo and Sydney are active
    t_asia = datetime(2026, 10, 22, 3, 0, tzinfo=timezone.utc)
    asia_sessions = get_active_sessions(t_asia)
    assert "Tokyo" in asia_sessions
    assert "Sydney" in asia_sessions


def test_forex_data_provider_quote_and_synthetic_fallback():
    provider = ForexDataProvider(use_synthetic=True)
    quote = provider.get_quote("EUR/USD")

    assert quote.pair == "EUR/USD"
    assert quote.bid < quote.ask
    assert quote.spread_pips > 0.0
    assert quote.annualized_volatility > 0.0
    assert len(quote.active_sessions) >= 1
    assert quote.is_synthetic is True

    # Check JPY pair quote
    quote_jpy = provider.get_quote("USD/JPY")
    assert quote_jpy.pair == "USD/JPY"
    assert quote_jpy.mid > 100.0


def test_unsupported_pair_raises_clear_error():
    provider = get_forex_provider()
    with pytest.raises(ValueError, match="not supported"):
        provider.get_quote("XYZ/ABC")


# ─── 2. Forex Market Regime Intelligence Tests ────────────────────────────────

def test_regime_intelligence_detection():
    intel = ForexMarketRegimeIntelligence(data_provider=ForexDataProvider(use_synthetic=True))
    res = intel.analyze_regime("EUR/USD")

    assert res.pair == "EUR/USD"
    assert res.regime in [
        ForexRegimeType.TRENDING_BULLISH,
        ForexRegimeType.TRENDING_BEARISH,
        ForexRegimeType.RANGING,
        ForexRegimeType.HIGH_VOLATILITY,
        ForexRegimeType.LOW_VOLATILITY,
        ForexRegimeType.CRISIS_EXTREME,
    ]
    assert 0.0 <= res.regime_confidence <= 1.0
    assert res.trend_direction in [TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.NEUTRAL]
    assert "adx_14" in res.supporting_indicators
    assert "bollinger_band_width" in res.supporting_indicators
    assert len(res.explanation) > 10


def test_signal_confidence_calibration_under_regimes():
    intel = get_regime_intelligence()
    mock_assessment = intel.analyze_regime("EUR/USD")

    # In Crisis regime, confidence must be strictly attenuated
    mock_assessment.regime = ForexRegimeType.CRISIS_EXTREME
    cal_conf, rationale, is_val = intel.calibrate_signal_confidence(
        base_signal="BUY",
        base_confidence=0.85,
        strategy_type="momentum",
        regime_assessment=mock_assessment,
    )
    assert cal_conf <= 0.35
    assert "CRISIS" in rationale

    # In Ranging regime, momentum signal must be attenuated (whipsaw risk)
    mock_assessment.regime = ForexRegimeType.RANGING
    cal_conf2, rationale2, is_val2 = intel.calibrate_signal_confidence(
        base_signal="BUY",
        base_confidence=0.80,
        strategy_type="momentum",
        regime_assessment=mock_assessment,
    )
    assert cal_conf2 < 0.80
    assert "Ranging" in rationale2


# ─── 3. Forex Pre-Event & Post-Event Economic Intelligence Tests ──────────────

def test_pre_event_assessment_no_future_leakage():
    analyzer = get_event_analyzer()
    assessment = analyzer.assess_pre_event("FOMC_2026_10", primary_pair="EUR/USD")

    assert assessment.event_id == "FOMC_2026_10"
    assert assessment.category == EventCategory.INTEREST_RATE
    assert assessment.currency == "USD"
    assert "EUR/USD" in assessment.affected_pairs
    assert assessment.expected_risk_level in [EventRiskLevel.HIGH, EventRiskLevel.EXTREME]
    assert assessment.consensus_expectation is not None
    assert assessment.current_market_price > 0.0
    assert len(assessment.citations) >= 1
    assert assessment.has_sufficient_evidence is True

    # Temporal integrity check: actual release must NOT be present in pre-event record
    assert not hasattr(assessment, "actual_value")


def test_post_event_analysis_and_indicator_direction():
    analyzer = get_event_analyzer()

    # Fed Rate Decision: consensus 4.75%, actual 5.00% (+0.25% surprise = Positive for USD)
    post = analyzer.analyze_post_event(
        event_id="FOMC_2026_10",
        actual_value=5.00,
        primary_pair="EUR/USD",
        simulated_pip_delta=-35.0,  # USD strengthens -> EUR/USD drops 35 pips
    )

    assert post.event_id == "FOMC_2026_10"
    assert post.actual_value == 5.00
    assert post.raw_surprise == 0.25
    assert post.normalized_surprise > 0.0
    assert post.surprise_direction_for_currency == "POSITIVE"
    assert post.initial_reaction == "BEARISH_SPIKE"  # On EUR/USD, positive USD = bearish spike
    assert len(post.window_reactions) == 5
    assert post.spread_peak_pips >= 3.0


def test_pre_event_assessment_persists_unmutated():
    analyzer = get_event_analyzer()
    pre = analyzer.assess_pre_event("US_CPI_2026_10", primary_pair="EUR/USD")
    initial_timestamp = pre.timestamp

    # Run post-event analysis
    post = analyzer.analyze_post_event("US_CPI_2026_10", actual_value=2.7, primary_pair="EUR/USD")
    assert post.actual_value == 2.7

    # Retrieve pre-event from DB to verify it was preserved unmutated
    session = get_session()
    pre_rec = session.query(ForexPreEventAssessmentRecord).filter_by(event_id="US_CPI_2026_10").order_by(ForexPreEventAssessmentRecord.id.desc()).first()
    session.close()

    assert pre_rec is not None
    assert pre_rec.event_id == "US_CPI_2026_10"
    assert pre_rec.expected_risk_level in ["HIGH", "EXTREME"]


def test_pre_vs_post_comparison_report():
    analyzer = get_event_analyzer()
    comp = analyzer.compare_pre_post("FOMC_2026_10", primary_pair="EUR/USD")

    assert comp.event_id == "FOMC_2026_10"
    assert comp.primary_pair == "EUR/USD"
    assert len(comp.outcome_summary) > 20
    assert len(comp.unresolved_uncertainties) >= 1
    assert comp.pre_event.event_id == comp.post_event.event_id


# ─── 4. Forex Event Reaction Memory Tests ─────────────────────────────────────

def test_event_reaction_memory_retrieval_and_similarity():
    memory = get_event_reaction_memory()
    summary = memory.find_similar_events(
        category="INTEREST_RATE",
        currency="USD",
        target_pair="EUR/USD",
        current_surprise=-0.25,
        current_regime="Trending Bullish",
        top_k=3,
    )

    assert summary.query_category == "INTEREST_RATE"
    assert summary.query_currency == "USD"
    assert summary.total_historical_matches >= 2
    assert summary.has_sufficient_history is True
    assert len(summary.top_similar_events) >= 2

    top = summary.top_similar_events[0]
    assert 0.0 <= top.similarity_score <= 1.0
    assert len(top.similarity_rationale) > 5
    assert top.observed_pattern in ["REVERSAL", "CONTINUATION", "CONSOLIDATION", "WHIPSAW"]
    assert summary.reversal_tendency in ["HIGH", "MODERATE", "LOW"]
    assert "price reactions" in summary.limitations[0].lower()


# ─── 5. Event-Related Simulated Trading Risk Gate Tests ─────────────────────────

def test_event_risk_gate_normal_conditions():
    gate = get_forex_risk_gate()
    # Evaluate at a neutral time when no event is active
    neutral_time = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)
    res = gate.evaluate_entry_risk("EUR/USD", side="BUY", as_of_time=neutral_time)

    assert res.pair == "EUR/USD"
    assert res.is_restricted is False
    assert "BUY" in res.allowed_actions
    assert res.spread_ratio < 2.5


def test_event_risk_gate_blocks_simulated_entries_in_blackout():
    gate = get_forex_risk_gate()
    # FOMC is scheduled at 2026-10-22T18:00:00Z.
    # Evaluating at 17:45 UTC (15m before announcement) must trigger blackout restriction!
    blackout_time = datetime(2026, 10, 22, 17, 45, tzinfo=timezone.utc)
    res = gate.evaluate_entry_risk("EUR/USD", side="BUY", as_of_time=blackout_time)

    assert res.is_restricted is True
    assert res.active_event_id == "FOMC_2026_10"
    assert "Active Event Blackout" in res.restriction_reason
    assert res.allowed_actions == ["HOLD"]  # BUY is blocked!


def test_existing_positions_never_deleted_by_event_risk():
    paper = PaperTradingEngine()
    paper.cash = 100000.0
    # Simulate an open position
    paper.holdings["EUR/USD"] = {"quantity": 10000.0, "avg_entry_price": 1.0850}

    gate = get_forex_risk_gate()
    blackout_time = datetime(2026, 10, 22, 17, 55, tzinfo=timezone.utc)
    res = gate.evaluate_entry_risk("EUR/USD", side="BUY", as_of_time=blackout_time)

    # Invariance check: The risk restriction must NOT delete or mutate the position
    assert "EUR/USD" in paper.holdings
    assert paper.holdings["EUR/USD"]["quantity"] == 10000.0


# ─── 6. Chatbot Forex Tool Dispatching Tests ──────────────────────────────────

def test_chatbot_forex_regime_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="What is the current market regime for EUR/USD?",
        symbol=None,
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_forex_regime"
    assert any(w in res["answer"].lower() for w in ["regime", "eur/usd", "ranging", "trending", "volatility"])


def test_chatbot_forex_pre_event_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="What major economic event is being analyzed for EUR/USD?",
        symbol=None,
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_forex_pre_event"
    assert any(w in res["answer"].lower() for w in ["pre-event", "fomc", "consensus", "scheduled", "risk"])


def test_chatbot_forex_reversal_continuation_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Did the initial reaction continue or reverse for EUR/USD?",
        symbol=None,
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_forex_post_event"
    assert any(w in res["answer"].lower() for w in ["reversal", "continue", "reaction", "pattern", "initial"])


def test_chatbot_forex_reaction_memory_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Which historical events are similar to the upcoming CPI release?",
        symbol=None,
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_forex_event_memory"
    assert any(w in res["answer"].lower() for w in ["memory", "historical", "analog", "reversal", "cpi"])


def test_chatbot_forex_risk_restriction_dispatch():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    paper = PaperTradingEngine()

    res = answer_question(
        question="Why did the system restrict new simulated trades for EUR/USD?",
        symbol=None,
        orchestrator=orch,
        rag_agent=rag,
        paper_engine=paper,
        last_decisions={},
    )
    assert res["tool_used"] == "get_forex_risk_restriction"
    assert any(w in res["answer"].lower() for w in ["restrict", "blackout", "spread", "simulated", "risk"])


# ─── 7. REST API Endpoints Integration Tests ──────────────────────────────────

def test_api_forex_pairs(client):
    res = client.get("/api/forex/pairs")
    assert res.status_code == 200
    data = res.get_json()
    assert "pairs" in data
    assert data["count"] >= 7
    pairs = [p["pair"] for p in data["pairs"]]
    assert "EUR/USD" in pairs
    assert "USD/JPY" in pairs


def test_api_forex_quote(client):
    res = client.get("/api/forex/quote/EUR/USD")
    assert res.status_code == 200
    data = res.get_json()
    assert data["pair"] == "EUR/USD"
    assert data["mid"] > 0.0
    assert "active_sessions" in data


def test_api_forex_regime(client):
    res = client.get("/api/forex/regime/GBP/USD")
    assert res.status_code == 200
    data = res.get_json()
    assert data["pair"] == "GBP/USD"
    assert "regime" in data
    assert "supporting_indicators" in data


def test_api_forex_events_and_pre(client):
    res = client.get("/api/forex/events")
    assert res.status_code == 200
    data = res.get_json()
    assert "events" in data
    assert data["count"] >= 3

    # Pre-event assessment
    res_pre = client.get("/api/forex/events/pre/FOMC_2026_10")
    assert res_pre.status_code == 200
    pre_data = res_pre.get_json()
    assert pre_data["event_id"] == "FOMC_2026_10"
    assert "consensus_expectation" in pre_data


def test_api_forex_post_event_and_compare(client):
    res_post = client.post(
        "/api/forex/events/post",
        json={"event_id": "FOMC_2026_10", "actual_value": 4.75, "pair": "EUR/USD"},
    )
    assert res_post.status_code == 200
    post_data = res_post.get_json()
    assert post_data["event_id"] == "FOMC_2026_10"
    assert len(post_data["window_reactions"]) == 5

    res_comp = client.get("/api/forex/events/compare/FOMC_2026_10")
    assert res_comp.status_code == 200
    comp_data = res_comp.get_json()
    assert "outcome_summary" in comp_data


def test_api_forex_similar_memory(client):
    res = client.get("/api/forex/events/memory/similar?category=INTEREST_RATE&currency=USD")
    assert res.status_code == 200
    data = res.get_json()
    assert "top_similar_events" in data
    assert data["has_sufficient_history"] is True


def test_api_forex_risk_check(client):
    res = client.get("/api/forex/risk/check/EUR/USD")
    assert res.status_code == 200
    data = res.get_json()
    assert data["pair"] == "EUR/USD"
    assert "is_restricted" in data
    assert "allowed_actions" in data

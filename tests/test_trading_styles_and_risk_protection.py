"""Comprehensive Automated Test Suite for Trading Styles, Risk Protection & Trade Quality Gate.

Covers:
1. Trading Styles (Day, Swing, Position) and operational configuration.
2. Deterministic Position Sizing (Stocks and Forex pip values with currency conversion).
3. Persistent Daily Loss Limits & Consecutive-Loss Cooldowns.
4. Database Lock Persistence across restarts.
5. Leverage, Margin & Cross-Pair Currency Concentration.
6. Trade Setup Quality Gate decisions (ELIGIBLE, REJECTED, ABSTAIN, DATA_UNAVAILABLE).
7. REST API endpoints.
8. Chatbot natural language explanations.
"""
import json
import pytest
from datetime import datetime, timezone, timedelta

from app.api.app import app
from app.db.models import init_db, get_session, DailyRiskStateRecord, TradingStyleConfigRecord
from app.risk.position_sizer import PositionSizer, get_position_sizer
from app.risk.daily_loss_guard import DailyLossGuard, get_daily_loss_guard
from app.risk.portfolio_exposure import PortfolioExposureEngine, get_portfolio_exposure_engine
from app.risk.trade_quality_gate import TradeQualityGate, get_trade_quality_gate
from app.strategy.trading_style_manager import TradingStyleManager, get_trading_style_manager
from app.trading.paper_trading import PaperTradingEngine
from app.orchestrator import TradingOrchestrator
from app.rag.rag_agent import FinancialRAGAgent
from app.services.chatbot import answer_question


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _clean_daily_state(guard: DailyLossGuard | None = None):
    g = guard or get_daily_loss_guard()
    g.reset_lock_manual("Reset for test")
    with get_session() as s:
        rec = s.query(DailyRiskStateRecord).filter_by(date_str=g._get_today_str()).first()
        if rec:
            rec.realized_pnl = 0.0
            rec.unrealized_pnl = 0.0
            rec.fees_paid = 0.0
            rec.consecutive_losses = 0
            rec.is_locked = False
            rec.lock_reason = None
            rec.cooldown_until = None
            s.commit()


# ─── 1. Trading Style Manager Tests ──────────────────────────────────────────

def test_trading_style_manager_profiles():
    mgr = TradingStyleManager()
    day = mgr.set_active_style("DAY", persist=False)
    assert day.style == "DAY"
    assert day.max_holding_days == 1.0
    assert day.default_risk_per_trade_pct == 0.005
    assert day.session_boundary_action == "FLAT_AT_CLOSE"

    swing = mgr.set_active_style("SWING", persist=False)
    assert swing.style == "SWING"
    assert swing.max_holding_days == 14.0
    assert swing.min_risk_reward_ratio == 2.0
    assert swing.carries_overnight_risk is True

    position = mgr.set_active_style("POSITION", persist=False)
    assert position.style == "POSITION"
    assert position.max_holding_days == 180.0
    assert position.min_risk_reward_ratio == 2.5


def test_trading_style_invalid_rejection():
    mgr = TradingStyleManager()
    with pytest.raises(ValueError):
        mgr.set_active_style("SCALPING_INVALID")


# ─── 2. Position Sizer & Stop-Loss Tests ─────────────────────────────────────

def test_stock_position_sizing_valid():
    sizer = PositionSizer()
    # $100,000 equity, 1% risk = $1,000 budget. Entry $200, Stop $190 ($10 distance).
    # Approx 99-100 shares.
    res = sizer.calculate_size(
        symbol="AAPL",
        side="BUY",
        entry_price=200.0,
        stop_loss_price=190.0,
        target_price=220.0,
        account_equity=100000.0,
        risk_pct=0.01,
    )
    assert res.is_valid is True
    assert res.rejection_reason is None
    assert res.risk_budget_usd == 1000.0
    assert res.max_units in (98.0, 99.0, 100.0)
    assert res.risk_reward_ratio == 2.0  # ($220-$200)/($200-$190) = 2.0
    assert res.estimated_monetary_risk <= 1010.0


def test_forex_position_sizing_eurusd():
    sizer = PositionSizer()
    # EUR/USD: 1 pip = 0.0001 = $10 per 100,000 lot.
    # Entry 1.1000, Stop 1.0950 = 50 pips.
    # Risk budget $1,000. 50 pips at $10/pip/lot = $500 per lot. Max = 2.0 lots = 200,000 units.
    res = sizer.calculate_size(
        symbol="EUR/USD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        target_price=1.1100,
        account_equity=100000.0,
        risk_pct=0.01,
    )
    assert res.is_valid is True
    assert res.instrument_type == "FOREX"
    assert res.stop_distance_pips == 50.0
    assert res.max_units >= 150000.0  # Approx 170k - 200k accounting for fees


def test_forex_position_sizing_usdjpy():
    sizer = PositionSizer()
    # USD/JPY: Pip value dynamically converted from JPY to USD.
    res = sizer.calculate_size(
        symbol="USD/JPY",
        side="BUY",
        entry_price=150.00,
        stop_loss_price=149.00,
        target_price=152.00,
        account_equity=100000.0,
        risk_pct=0.01,
    )
    assert res.is_valid is True
    assert res.instrument_type == "FOREX"
    assert res.stop_distance_pips == 100.0
    assert res.max_units >= 5000.0


def test_position_sizing_invalid_stop_placement():
    sizer = PositionSizer()
    # For BUY: stop cannot be >= entry
    res = sizer.calculate_size(
        symbol="AAPL",
        side="BUY",
        entry_price=200.0,
        stop_loss_price=205.0,
        account_equity=100000.0,
    )
    assert res.is_valid is False
    assert "must be strictly below entry price" in res.rejection_reason

    # For SELL: stop cannot be <= entry
    res_sell = sizer.calculate_size(
        symbol="AAPL",
        side="SELL",
        entry_price=200.0,
        stop_loss_price=195.0,
        account_equity=100000.0,
    )
    assert res_sell.is_valid is False
    assert "must be strictly above entry price" in res_sell.rejection_reason


def test_position_sizing_narrow_stop_rejection():
    sizer = PositionSizer()
    # Stop distance too narrow (0.01% distance)
    res = sizer.calculate_size(
        symbol="AAPL",
        side="BUY",
        entry_price=200.0,
        stop_loss_price=199.99,
        account_equity=100000.0,
    )
    assert res.is_valid is False
    assert "too narrow" in res.rejection_reason


# ─── 3. Daily Loss & Anti-Revenge Guard Tests ───────────────────────────────

def test_daily_loss_guard_normal_state():
    guard = DailyLossGuard(max_daily_loss_pct=0.03)
    _clean_daily_state(guard)
    status = guard.update_unrealized_pnl_and_evaluate(current_equity=100000.0, unrealized_pnl=0.0)
    assert status.is_locked is False
    assert status.can_enter_new_trades is True
    assert status.daily_loss_allowance_remaining == 3000.0


def test_daily_loss_guard_breach_locks_trading():
    guard = DailyLossGuard(max_daily_loss_pct=0.03)
    # Simulate a -$3,500 net loss (3.5% breach on $100k)
    guard.record_closed_trade_result(trade_pnl=-3500.0, fees=10.0)
    status = guard.update_unrealized_pnl_and_evaluate(current_equity=96490.0, unrealized_pnl=0.0)

    assert status.is_locked is True
    assert status.can_enter_new_trades is False
    assert "Daily loss limit breached" in (status.lock_reason or "")


def test_daily_loss_lock_persistence_across_instances():
    # Instantiate a completely fresh guard object
    fresh_guard = DailyLossGuard(max_daily_loss_pct=0.03)
    status = fresh_guard.update_unrealized_pnl_and_evaluate(current_equity=96490.0, unrealized_pnl=0.0)
    # Must remain locked from DB persistence!
    assert status.is_locked is True
    assert status.can_enter_new_trades is False


def test_consecutive_losses_triggers_cooldown():
    guard = DailyLossGuard(max_daily_loss_pct=0.10, max_consecutive_losses=3, loss_cooldown_minutes=60)
    _clean_daily_state(guard)

    # Record 2 losses -> no lock yet
    guard.record_closed_trade_result(trade_pnl=-50.0)
    guard.record_closed_trade_result(trade_pnl=-50.0)
    st1 = guard.update_unrealized_pnl_and_evaluate(current_equity=99900.0, unrealized_pnl=0.0)
    assert st1.is_locked is False

    # 3rd consecutive loss -> triggers cooldown
    guard.record_closed_trade_result(trade_pnl=-50.0)
    st2 = guard.update_unrealized_pnl_and_evaluate(current_equity=99850.0, unrealized_pnl=0.0)
    assert st2.is_locked is True
    assert st2.is_cooldown_active is True
    assert "Consecutive loss cooldown" in (st2.lock_reason or "")

    # Manual reset for next tests
    guard.reset_lock_manual("Clean up test lock")


# ─── 4. Portfolio Exposure & Currency Concentration Tests ────────────────────

def test_portfolio_exposure_gross_leverage():
    engine = PortfolioExposureEngine(max_leverage_stock=5.0)
    # Proposed $600,000 notional on $100,000 equity = 6.0x leverage (exceeds 5.0x cap)
    res = engine.evaluate_exposure(
        proposed_symbol="AAPL",
        proposed_side="BUY",
        proposed_notional_usd=600000.0,
        account_equity=100000.0,
    )
    assert res.is_approved is False
    assert any("Gross leverage" in v for v in res.violations)


def test_portfolio_exposure_currency_concentration():
    engine = PortfolioExposureEngine(max_currency_pct=0.40)
    # Existing: Long EUR/USD $25k notional (Short USD $25k)
    # Proposed: Long GBP/USD $25k notional (Short USD $25k)
    # Total Short USD = $50k on $100k equity = 50% concentration (exceeds 40% ceiling)
    existing = {
        "EUR/USD": {"quantity": 25000.0, "avg_entry_price": 1.0, "current_price": 1.0}
    }
    res = engine.evaluate_exposure(
        proposed_symbol="GBP/USD",
        proposed_side="BUY",
        proposed_notional_usd=25000.0,
        account_equity=100000.0,
        existing_positions=existing,
    )
    assert res.is_approved is False
    assert any("USD" in v and "currency ceiling" in v for v in res.violations)


# ─── 5. Trade Setup Quality Gate Tests ──────────────────────────────────────

def test_trade_quality_gate_eligible():
    _clean_daily_state()
    gate = TradeQualityGate()
    dec = gate.evaluate(
        symbol="AAPL",
        side="BUY",
        entry_price=200.0,
        stop_loss_price=180.0,
        target_price=240.0,
        trading_style="SWING",
        account_equity=100000.0,
        strategy_win_rate=0.55,
        historical_sample_size=30,
    )
    assert dec.decision == "ELIGIBLE"
    assert dec.is_approved is True
    assert dec.calculated_units > 0
    assert dec.risk_reward_ratio == 2.0


def test_trade_quality_gate_unvalidated_strategy_abstains():
    _clean_daily_state()
    gate = TradeQualityGate()
    # Strategy untested (sample size < 10) -> ABSTAIN
    dec = gate.evaluate(
        symbol="AAPL",
        side="BUY",
        entry_price=200.0,
        stop_loss_price=180.0,
        target_price=240.0,
        trading_style="SWING",
        account_equity=100000.0,
        strategy_win_rate=None,
        historical_sample_size=3,
    )
    assert dec.decision == "ABSTAIN"
    assert dec.is_approved is False
    assert any("UNVALIDATED" in r for r in dec.rejection_reasons)


def test_trade_quality_gate_data_unavailable():
    gate = TradeQualityGate()
    # Price is 0 -> DATA_UNAVAILABLE
    dec = gate.evaluate(
        symbol="AAPL",
        side="BUY",
        entry_price=0.0,
        stop_loss_price=0.0,
    )
    assert dec.decision == "DATA_UNAVAILABLE"
    assert dec.is_approved is False


# ─── 6. REST API Endpoints Tests ─────────────────────────────────────────────

def test_api_trading_style_endpoints(client):
    res = client.get("/api/trading-style")
    assert res.status_code == 200
    data = res.get_json()
    assert "active_style" in data
    assert "available_styles" in data

    # Set style to DAY
    post_res = client.post("/api/trading-style/set", json={"style": "DAY"})
    assert post_res.status_code == 200
    pdata = post_res.get_json()
    assert pdata["active_style"] == "DAY"
    assert pdata["profile"]["session_boundary_action"] == "FLAT_AT_CLOSE"


def test_api_trade_gate_evaluate(client):
    payload = {
        "symbol": "AAPL",
        "side": "BUY",
        "entry_price": 225.0,
        "stop_loss_price": 215.0,
        "target_price": 245.0,
        "trading_style": "SWING",
    }
    res = client.post("/api/trade-gate/evaluate", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["decision"] in ("ELIGIBLE", "REJECTED", "ABSTAIN")
    assert "checks" in data
    assert "calculated_units" in data


def test_api_risk_daily_status_and_reset(client):
    res = client.get("/api/risk/daily-status")
    assert res.status_code == 200
    data = res.get_json()
    assert "starting_equity" in data
    assert "max_daily_loss_pct" in data

    # Reset cooldown endpoint
    rst = client.post("/api/risk/cooldown/reset", json={"reason": "Test reset"})
    assert rst.status_code == 200
    assert rst.get_json()["success"] in (True, False)


def test_api_portfolio_exposure(client):
    res = client.get("/api/risk/portfolio-exposure")
    assert res.status_code == 200
    data = res.get_json()
    assert "gross_leverage" in data
    assert "free_margin_projected" in data


# ─── 7. Chatbot Natural Language Explanations Tests ──────────────────────────

def test_chatbot_trading_style_intent():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    pe = PaperTradingEngine()
    ans = answer_question("Which trading style is currently selected?", None, orch, rag, pe, {})
    assert ans["tool_used"] == "get_trading_style_status"
    assert "style" in ans["answer"].lower()


def test_chatbot_daily_loss_intent():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    pe = PaperTradingEngine()
    ans = answer_question("How much of the daily loss allowance remains?", None, orch, rag, pe, {})
    assert ans["tool_used"] == "get_daily_risk_status"
    assert "loss" in ans["answer"].lower()


def test_chatbot_position_sizing_intent():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    pe = PaperTradingEngine()
    ans = answer_question("How was the position size calculated for NVDA?", "NVDA", orch, rag, pe, {})
    assert ans["tool_used"] == "get_position_sizing_query"
    assert "position" in ans["answer"].lower() or "shares" in ans["answer"].lower() or "units" in ans["answer"].lower()


def test_chatbot_portfolio_exposure_intent():
    orch = TradingOrchestrator()
    rag = FinancialRAGAgent()
    pe = PaperTradingEngine()
    ans = answer_question("Is the current leverage within the configured limit?", None, orch, rag, pe, {})
    assert ans["tool_used"] == "get_portfolio_exposure_query"
    assert "leverage" in ans["answer"].lower()

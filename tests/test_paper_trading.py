"""Tests for the paper trading engine."""
from app.trading.paper_trading import PaperTradingEngine


def test_buy_reduces_cash_and_creates_position():
    engine = PaperTradingEngine(starting_cash=100_000)
    trade = engine.execute("AAPL", "BUY", price=100.0, position_pct=0.10, reason="test", risk_status="PASS")
    assert trade.side == "BUY"
    assert trade.quantity > 0
    assert engine.cash < 100_000
    assert engine.holdings["AAPL"]["quantity"] == trade.quantity


def test_sell_realizes_pnl():
    engine = PaperTradingEngine(starting_cash=100_000)
    engine.execute("AAPL", "BUY", price=100.0, position_pct=0.10, reason="entry", risk_status="PASS")
    trade = engine.execute("AAPL", "SELL", price=120.0, position_pct=0.0, reason="exit", risk_status="PASS")
    assert trade.side == "SELL"
    assert engine.realized_pnl > 0  # bought at 100, sold at 120 -> profit


def test_hold_creates_no_position_change():
    engine = PaperTradingEngine(starting_cash=100_000)
    trade = engine.execute("AAPL", "HOLD", price=100.0, position_pct=0.0, reason="test", risk_status="PASS")
    assert trade.side == "HOLD"
    assert engine.cash == 100_000


def test_reject_status_forces_no_trade():
    engine = PaperTradingEngine(starting_cash=100_000)
    trade = engine.execute("AAPL", "BUY", price=100.0, position_pct=0.10, reason="test", risk_status="REJECT")
    assert trade.side == "HOLD"
    assert engine.cash == 100_000


def test_cannot_sell_without_position():
    engine = PaperTradingEngine(starting_cash=100_000)
    trade = engine.execute("AAPL", "SELL", price=100.0, position_pct=0.0, reason="test", risk_status="PASS")
    assert trade is None


def test_portfolio_state_reflects_unrealized_pnl():
    engine = PaperTradingEngine(starting_cash=100_000)
    engine.execute("AAPL", "BUY", price=100.0, position_pct=0.10, reason="entry", risk_status="PASS")
    state = engine.get_state({"AAPL": 110.0})
    assert state.positions["AAPL"].unrealized_pnl > 0
    assert state.total_portfolio_value > 0

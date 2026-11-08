"""Paper Trading Engine.

Simulates BUY/SELL/HOLD execution with transaction costs and slippage.
No real money, no real brokerage connection — this is explicitly a
research / paper-trading simulator (see app/api routes for the disclosure
banner text surfaced to the UI).
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from app.config import settings


class Trade(BaseModel):
    trade_id: str
    symbol: str
    side: str  # BUY / SELL / HOLD
    quantity: float
    price: float
    executed_price: float  # after slippage
    transaction_cost: float
    timestamp: str
    reason: str
    risk_status: str


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioState(BaseModel):
    cash: float
    positions: dict[str, Position]
    total_market_value: float
    total_portfolio_value: float
    realized_pnl: float
    trade_count: int


class PaperTradingEngine:
    """In-memory paper trading book. `app/db/models.py` persists trades to
    SQLite for the dashboard's trade history; this class holds live state
    for a single running process."""

    def __init__(self, starting_cash: float | None = None) -> None:
        self.cash = starting_cash if starting_cash is not None else settings.starting_cash
        self.holdings: dict[str, dict] = {}  # symbol -> {quantity, avg_entry_price}
        self.realized_pnl = 0.0
        self.trade_log: list[Trade] = []
        self._trade_counter = 0

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        ts = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
        return f"T{ts}_{self._trade_counter:03d}"

    def execute(self, symbol: str, side: str, price: float, position_pct: float,
                reason: str, risk_status: str) -> Trade | None:
        symbol = symbol.upper()
        if side == "HOLD" or risk_status != "PASS":
            trade = Trade(
                trade_id=self._next_trade_id(), symbol=symbol, side="HOLD", quantity=0.0,
                price=price, executed_price=price, transaction_cost=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason, risk_status=risk_status,
            )
            self.trade_log.append(trade)
            return trade

        slippage_mult = 1 + (settings.slippage_bps / 10_000) * (1 if side == "BUY" else -1)
        executed_price = round(price * slippage_mult, 4)

        if side == "BUY":
            # Check Daily Loss Guard for active locks
            from app.risk.daily_loss_guard import get_daily_loss_guard
            daily_guard = get_daily_loss_guard()
            status = daily_guard.update_unrealized_pnl_and_evaluate(self.total_portfolio_value(), 0.0)
            if not status.can_enter_new_trades:
                logger.warning("Paper trade blocked by Daily Loss Guard: %s", status.lock_reason)
                return None

            budget = self.total_portfolio_value() * position_pct
            budget = min(budget, self.cash)
            if budget <= 0:
                return None
            quantity = round(budget / executed_price, 6)
            cost = executed_price * quantity
            fee = cost * (settings.transaction_cost_bps / 10_000)
            if cost + fee > self.cash:
                quantity = round((self.cash / (1 + settings.transaction_cost_bps / 10_000)) / executed_price, 6)
                cost = executed_price * quantity
                fee = cost * (settings.transaction_cost_bps / 10_000)

            self.cash -= (cost + fee)
            pos = self.holdings.setdefault(symbol, {"quantity": 0.0, "avg_entry_price": 0.0})
            new_qty = pos["quantity"] + quantity
            pos["avg_entry_price"] = (
                (pos["avg_entry_price"] * pos["quantity"] + executed_price * quantity) / new_qty
                if new_qty > 0 else executed_price
            )
            pos["quantity"] = new_qty

            trade = Trade(
                trade_id=self._next_trade_id(), symbol=symbol, side="BUY", quantity=quantity,
                price=price, executed_price=executed_price, transaction_cost=round(fee, 4),
                timestamp=datetime.now(timezone.utc).isoformat(), reason=reason, risk_status=risk_status,
            )

        elif side == "SELL":
            pos = self.holdings.get(symbol)
            if not pos or pos["quantity"] <= 0:
                return None
            quantity = pos["quantity"]
            proceeds = executed_price * quantity
            fee = proceeds * (settings.transaction_cost_bps / 10_000)
            self.cash += (proceeds - fee)
            realized = (executed_price - pos["avg_entry_price"]) * quantity - fee
            self.realized_pnl += realized
            self.holdings[symbol] = {"quantity": 0.0, "avg_entry_price": 0.0}

            # Record closed trade result in Daily Loss Guard
            try:
                from app.risk.daily_loss_guard import get_daily_loss_guard
                get_daily_loss_guard().record_closed_trade_result(trade_pnl=realized, fees=fee)
            except Exception as guard_exc:
                pass

            trade = Trade(
                trade_id=self._next_trade_id(), symbol=symbol, side="SELL", quantity=quantity,
                price=price, executed_price=executed_price, transaction_cost=round(fee, 4),
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason + f" | realized_pnl={realized:.2f}", risk_status=risk_status,
            )
        else:
            return None

        self.trade_log.append(trade)
        return trade

    def execute_order(self, symbol: str, side: str, quantity: float, price: float,
                      reason: str = "Manual Paper Trade", risk_status: str = "PASS") -> Trade | None:
        """Execute a manual paper trade order with specific quantity and price."""
        symbol = symbol.upper()
        side = side.upper()
        if quantity <= 0 or price <= 0:
            return None

        slippage_mult = 1 + (settings.slippage_bps / 10_000) * (1 if side == "BUY" else -1)
        executed_price = round(price * slippage_mult, 4)

        if side == "BUY":
            # Check Daily Loss Guard for active locks
            from app.risk.daily_loss_guard import get_daily_loss_guard
            daily_guard = get_daily_loss_guard()
            status = daily_guard.update_unrealized_pnl_and_evaluate(self.total_portfolio_value(), 0.0)
            if not status.can_enter_new_trades:
                logger.warning("Manual order blocked by Daily Loss Guard: %s", status.lock_reason)
                return None

            cost = executed_price * quantity
            fee = cost * (settings.transaction_cost_bps / 10_000)
            if cost + fee > self.cash:
                return None
            self.cash -= (cost + fee)
            pos = self.holdings.setdefault(symbol, {"quantity": 0.0, "avg_entry_price": 0.0})
            new_qty = pos["quantity"] + quantity
            pos["avg_entry_price"] = (
                (pos["avg_entry_price"] * pos["quantity"] + executed_price * quantity) / new_qty
                if new_qty > 0 else executed_price
            )
            pos["quantity"] = new_qty
            trade = Trade(
                trade_id=self._next_trade_id(), symbol=symbol, side="BUY", quantity=quantity,
                price=price, executed_price=executed_price, transaction_cost=round(fee, 4),
                timestamp=datetime.now(timezone.utc).isoformat(), reason=reason, risk_status=risk_status,
            )
        elif side == "SELL":
            pos = self.holdings.get(symbol)
            if not pos or pos["quantity"] < quantity:
                return None
            proceeds = executed_price * quantity
            fee = proceeds * (settings.transaction_cost_bps / 10_000)
            self.cash += (proceeds - fee)
            realized = (executed_price - pos["avg_entry_price"]) * quantity - fee
            self.realized_pnl += realized
            pos["quantity"] -= quantity

            # Record closed trade result in Daily Loss Guard
            try:
                from app.risk.daily_loss_guard import get_daily_loss_guard
                get_daily_loss_guard().record_closed_trade_result(trade_pnl=realized, fees=fee)
            except Exception as guard_exc:
                pass

            trade = Trade(
                trade_id=self._next_trade_id(), symbol=symbol, side="SELL", quantity=quantity,
                price=price, executed_price=executed_price, transaction_cost=round(fee, 4),
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason + f" | realized_pnl={realized:.2f}", risk_status=risk_status,
            )
        else:
            return None

        self.trade_log.append(trade)
        return trade

    def total_portfolio_value(self, current_prices: dict[str, float] | None = None) -> float:
        market_value = 0.0
        prices = current_prices or {}
        for sym, pos in self.holdings.items():
            px = prices.get(sym, pos["avg_entry_price"])
            market_value += pos["quantity"] * px
        return self.cash + market_value

    def get_state(self, current_prices: dict[str, float] | None = None) -> PortfolioState:
        positions = {}
        market_value_total = 0.0
        prices = current_prices or {}
        for sym, pos in self.holdings.items():
            if pos["quantity"] <= 0:
                continue
            px = prices.get(sym, pos["avg_entry_price"])
            mv = pos["quantity"] * px
            unrealized = (px - pos["avg_entry_price"]) * pos["quantity"]
            unrealized_pct = (px / pos["avg_entry_price"] - 1) if pos["avg_entry_price"] else 0.0
            positions[sym] = Position(
                symbol=sym, quantity=round(pos["quantity"], 6), avg_entry_price=round(pos["avg_entry_price"], 4),
                current_price=round(px, 4), market_value=round(mv, 2),
                unrealized_pnl=round(unrealized, 2), unrealized_pnl_pct=round(unrealized_pct, 4),
            )
            market_value_total += mv

        return PortfolioState(
            cash=round(self.cash, 2), positions=positions,
            total_market_value=round(market_value_total, 2),
            total_portfolio_value=round(self.cash + market_value_total, 2),
            realized_pnl=round(self.realized_pnl, 2),
            trade_count=len([t for t in self.trade_log if t.side != "HOLD"]),
        )

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Convenience dictionary representation of current portfolio state."""
        st = self.get_state()
        return {
            "cash": st.cash,
            "total_portfolio_value": st.total_portfolio_value,
            "total_market_value": st.total_market_value,
            "realized_pnl": st.realized_pnl,
            "unrealized_pnl": sum(p.unrealized_pnl for p in st.positions.values()),
            "trade_count": st.trade_count,
            "positions": {sym: p.model_dump() for sym, p in st.positions.items()},
        }

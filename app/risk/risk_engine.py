"""Deterministic Risk Engine.

This is a hard gate: it can override every AI signal. The LLM/agents never
have authority to bypass it. Every check here is a plain numeric rule —
nothing probabilistic or model-based.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.config import settings


class RiskCheckResult(BaseModel):
    symbol: str
    approved: bool
    status: str  # PASS / REJECT
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility_annualized: float
    position_limit_pct: float
    proposed_position_pct: float
    concentration_ok: bool
    stop_loss_price: float | None
    violations: list[str]
    timestamp: str


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR — the loss threshold not expected to be exceeded with
    the given confidence, expressed as a positive fraction of capital."""
    if len(returns) < 10:
        return 0.0
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall — average loss beyond the VaR threshold."""
    if len(returns) < 10:
        return 0.0
    var = -value_at_risk(returns, confidence)  # negative threshold
    tail = returns[returns <= var]
    if tail.empty:
        return abs(var)
    return float(-tail.mean())


def max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max
    return float(-drawdown.min())


class RiskEngine:
    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        proposed_signal: str,
        proposed_position_pct: float,
        current_portfolio_exposure_pct: float,
        current_daily_pnl_pct: float = 0.0,
    ) -> RiskCheckResult:
        close = df["adj_close"]
        returns = close.pct_change().dropna()

        var95 = value_at_risk(returns, settings.var_confidence)
        cvar95 = conditional_value_at_risk(returns, settings.var_confidence)
        mdd = max_drawdown(close)
        vol_annualized = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        violations: list[str] = []

        if proposed_position_pct > settings.max_position_pct:
            violations.append(
                f"proposed position {proposed_position_pct:.1%} exceeds max single-position limit "
                f"{settings.max_position_pct:.1%}"
            )

        projected_exposure = current_portfolio_exposure_pct + proposed_position_pct
        concentration_ok = projected_exposure <= settings.max_concentration_pct
        if not concentration_ok and proposed_signal == "BUY":
            violations.append(
                f"projected sector/portfolio concentration {projected_exposure:.1%} exceeds limit "
                f"{settings.max_concentration_pct:.1%}"
            )

        if vol_annualized > settings.max_portfolio_volatility:
            violations.append(
                f"asset annualized volatility {vol_annualized:.1%} exceeds max allowed "
                f"{settings.max_portfolio_volatility:.1%}"
            )

        if current_daily_pnl_pct <= -settings.max_daily_loss_pct:
            violations.append(
                f"daily loss {current_daily_pnl_pct:.1%} has already breached max daily loss "
                f"{-settings.max_daily_loss_pct:.1%} — no new risk-increasing trades allowed"
            )

        approved = len(violations) == 0 or proposed_signal == "HOLD"
        stop_loss_price = None
        if proposed_signal == "BUY" and approved:
            last_price = float(close.iloc[-1])
            stop_loss_price = round(last_price * (1 - settings.stop_loss_pct), 2)

        return RiskCheckResult(
            symbol=symbol,
            approved=approved,
            status="PASS" if approved else "REJECT",
            var_95=round(var95, 4),
            cvar_95=round(cvar95, 4),
            max_drawdown=round(mdd, 4),
            volatility_annualized=round(vol_annualized, 4),
            position_limit_pct=settings.max_position_pct,
            proposed_position_pct=round(proposed_position_pct, 4),
            concentration_ok=concentration_ok,
            stop_loss_price=stop_loss_price,
            violations=violations,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

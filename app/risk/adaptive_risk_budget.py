"""Adaptive Risk Budget Module.

Research concept: Static risk limits (e.g. constant 10% max position, 30% exposure)
fail to adapt to changing volatility regimes and portfolio drawdowns.

The Adaptive Risk Budget dynamically recalculates:
  1. Base Risk Budget Multiplier [0.2x to 1.2x]
  2. Adjusted Max Position Size (scales down in high-vol/bear regimes)
  3. Maximum Portfolio Exposure Limit
  4. Mandatory Cash Allocation Buffer
  5. Maximum Daily Turnover Cap

SAFETY RULE: The deterministic hard constraints (base settings.max_position_pct,
VaR, CVaR, concentration bounds) form an absolute upper ceiling. The Adaptive Risk
Budget can only CONSTRICT risk during danger, never expand it beyond hard limits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.config import settings


class AdaptiveRiskBudgetResult(BaseModel):
    symbol: str
    regime: str
    risk_budget_multiplier: float = Field(ge=0.1, le=1.2, description="Dynamic scaling factor applied to baseline risk")
    base_max_position_pct: float
    adjusted_max_position_pct: float
    base_max_exposure_pct: float
    adjusted_max_exposure_pct: float
    recommended_cash_allocation_pct: float
    daily_turnover_limit_pct: float
    regime_rationale: str
    hard_constraints_active: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AdaptiveRiskBudget:
    """Computes regime-aware, state-contingent risk parameters."""

    def calculate_budget(
        self,
        symbol: str,
        regime_label: str,
        vol_percentile: float,
        current_drawdown_pct: float = 0.0,
        current_daily_loss_pct: float = 0.0,
    ) -> AdaptiveRiskBudgetResult:
        base_pos = settings.max_position_pct
        base_exp = settings.max_concentration_pct

        multiplier = 1.0
        reasons: list[str] = []

        # 1. Regime-based scaling
        regime_upper = regime_label.upper()
        if "BEAR" in regime_upper and "HIGH_VOLATILITY" in regime_upper:
            multiplier *= 0.40
            reasons.append("Bear market + High volatility: extreme risk constriction (0.40x)")
        elif "BEAR" in regime_upper:
            multiplier *= 0.60
            reasons.append("Bear market trend: risk budget reduced (0.60x)")
        elif "HIGH_VOLATILITY" in regime_upper:
            multiplier *= 0.65
            reasons.append("High volatility regime: sizing reduced (0.65x)")
        elif "BULL" in regime_upper and "LOW_VOLATILITY" in regime_upper:
            multiplier *= 1.10
            reasons.append("Favorable Bull + Low volatility: slight budget expansion within hard ceilings (1.10x)")
        elif "SIDEWAYS" in regime_upper:
            multiplier *= 0.80
            reasons.append("Range-bound sideways market: moderate risk compression (0.80x)")
        else:
            multiplier *= 0.90
            reasons.append("Neutral/standard regime: baseline conservative budget (0.90x)")

        # 2. Drawdown dampener
        if current_drawdown_pct > 0.05:
            multiplier *= 0.75
            reasons.append(f"Recent portfolio drawdown ({current_drawdown_pct:.1%}) triggers circuit dampener")

        # 3. Daily loss proximity
        if current_daily_loss_pct < -0.015:
            multiplier *= 0.50
            reasons.append(f"Daily loss ({current_daily_loss_pct:.1%}) approaching daily limit: protective contraction")

        multiplier = max(0.20, min(1.15, multiplier))

        # Adjusted limits (never exceeding base hard limits)
        adjusted_pos = round(min(base_pos, base_pos * multiplier), 4)
        adjusted_exp = round(min(base_exp, base_exp * multiplier), 4)

        # Cash buffer scales inversely to risk budget
        cash_buffer = round(max(0.10, 1.0 - adjusted_exp), 4)
        turnover_cap = round(max(0.05, 0.20 * multiplier), 4)

        return AdaptiveRiskBudgetResult(
            symbol=symbol.upper(),
            regime=regime_label,
            risk_budget_multiplier=round(multiplier, 3),
            base_max_position_pct=base_pos,
            adjusted_max_position_pct=adjusted_pos,
            base_max_exposure_pct=base_exp,
            adjusted_max_exposure_pct=adjusted_exp,
            recommended_cash_allocation_pct=cash_buffer,
            daily_turnover_limit_pct=turnover_cap,
            regime_rationale="; ".join(reasons),
            hard_constraints_active=True,
        )

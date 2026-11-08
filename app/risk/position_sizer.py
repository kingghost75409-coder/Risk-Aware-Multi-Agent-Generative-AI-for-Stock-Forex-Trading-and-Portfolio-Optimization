"""Deterministic Position Sizing & Stop-Loss Validation Service.

Calculates the mathematically permitted size of a proposed simulated position
based on:
1. Account equity & configured risk percentage (default 1%, max 2%)
2. Distance from entry price to stop-loss price
3. Instrument specifications (Stock shares vs Forex lots/pips)
4. Account-currency conversion for Forex pip values
5. Trading friction (estimated spread, commission, slippage)

INVARIANTS:
- A stop-loss does not guarantee exact loss amount due to gaps and slippage;
  all estimated risk amounts are labeled and treated as estimates.
- Stop-loss placement must be mathematically sound:
  - For BUY: stop_loss < entry (and stop distance > spread)
  - For SELL: stop_loss > entry (and stop distance > spread)
- Forex pip value is dynamically derived, never a hardcoded constant.
"""
from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from app.config import settings
from app.forex.forex_data import (
    is_supported_pair,
    normalize_forex_pair,
    get_pip_size,
    get_forex_provider,
    _TYPICAL_SPREAD_PIPS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PositionSizingResult(BaseModel):
    symbol: str
    instrument_type: str  # "STOCK" or "FOREX"
    side: str  # "BUY" or "SELL"
    is_valid: bool
    rejection_reason: str | None = None
    account_equity: float
    risk_pct: float
    risk_budget_usd: float
    entry_price: float
    stop_loss_price: float
    target_price: float | None = None
    stop_distance: float
    stop_distance_pct: float
    stop_distance_pips: float | None = None
    risk_reward_ratio: float | None = None
    estimated_loss_per_unit: float
    max_units: float  # Shares for stocks, contract units for Forex
    estimated_monetary_risk: float
    position_notional_value: float
    estimated_commission: float
    estimated_slippage: float
    explanation: str


class PositionSizer:
    """Calculates permitted position sizing based on risk budget and stop-loss distance."""

    def __init__(self, default_risk_pct: float | None = None, max_risk_pct: float = 0.02):
        self.default_risk_pct = default_risk_pct or settings.risk_per_trade_pct
        self.max_risk_pct = max_risk_pct

    def calculate_forex_pip_value_usd(self, pair: str, current_price: float, unit_count: float = 1.0) -> float:
        """Calculate the USD value of 1 pip for a given unit count.

        Forex Convention for Account Currency USD:
        - Quote is USD (EUR/USD, GBP/USD, AUD/USD, NZD/USD):
          Pip Value = Pip Size * Units (e.g. 0.0001 * 100,000 = $10.00)
        - Base is USD (USD/JPY, USD/CAD, USD/CHF):
          Pip Value in Quote = Pip Size * Units
          Converted to USD = (Pip Size / Current Exchange Rate) * Units
        """
        norm = normalize_forex_pair(pair)
        pip_size = get_pip_size(norm)
        parts = norm.split("/")
        base, quote = parts[0], parts[1]

        if quote == "USD":
            # Direct quote
            return pip_size * unit_count
        elif base == "USD":
            # Indirect quote
            if current_price <= 0:
                return 0.0
            return (pip_size / current_price) * unit_count
        else:
            # Cross currency (fallback to direct conversion if available)
            return (pip_size / max(current_price, 0.0001)) * unit_count

    def calculate_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        target_price: float | None = None,
        account_equity: float = 100000.0,
        risk_pct: float | None = None,
        instrument_type: str | None = None,
        available_cash: float | None = None,
    ) -> PositionSizingResult:
        sym = symbol.strip().upper()
        side_norm = side.strip().upper()

        # 1. Determine instrument type
        if instrument_type is None:
            is_fx = "/" in sym or is_supported_pair(sym) or sym.endswith("=X")
            inst_type = "FOREX" if is_fx else "STOCK"
        else:
            inst_type = instrument_type.upper()

        # 2. Risk percentage calibration (clamp to max 2.0% for safety)
        chosen_risk_pct = risk_pct if risk_pct is not None else self.default_risk_pct
        effective_risk_pct = min(max(chosen_risk_pct, 0.001), self.max_risk_pct)
        risk_budget = round(account_equity * effective_risk_pct, 2)

        # 3. Basic input validations
        if entry_price <= 0:
            return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                  entry_price, stop_loss_price, "Entry price must be strictly positive.")

        if stop_loss_price <= 0:
            return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                  entry_price, stop_loss_price, "Stop-loss price must be strictly positive.")

        if side_norm not in ("BUY", "SELL"):
            return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                  entry_price, stop_loss_price, f"Invalid order side '{side}'. Must be BUY or SELL.")

        # 4. Stop-loss logic check
        if side_norm == "BUY" and stop_loss_price >= entry_price:
            return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                  entry_price, stop_loss_price,
                                  f"For BUY orders, stop-loss ({stop_loss_price}) must be strictly below entry price ({entry_price}).")

        if side_norm == "SELL" and stop_loss_price <= entry_price:
            return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                  entry_price, stop_loss_price,
                                  f"For SELL orders, stop-loss ({stop_loss_price}) must be strictly above entry price ({entry_price}).")

        stop_dist = abs(entry_price - stop_loss_price)
        stop_dist_pct = stop_dist / entry_price

        # 5. Spread and minimum distance check
        if inst_type == "FOREX":
            norm_pair = normalize_forex_pair(sym)
            pip_size = get_pip_size(norm_pair)
            stop_dist_pips = round(stop_dist / pip_size, 1)
            typical_spread = _TYPICAL_SPREAD_PIPS.get(norm_pair, 1.5)
            if stop_dist_pips < typical_spread * 1.5:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"Stop distance ({stop_dist_pips:.1f} pips) is too narrow relative to typical spread ({typical_spread} pips).")
        else:
            stop_dist_pips = None
            min_dist_pct = 0.002  # At least 0.2% distance for stocks
            if stop_dist_pct < min_dist_pct:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"Stop distance ({stop_dist_pct:.2%}) is too narrow (min {min_dist_pct:.1%}).")

        # 6. Target price & Risk/Reward check
        rr_ratio: float | None = None
        if target_price and target_price > 0:
            target_dist = abs(target_price - entry_price)
            rr_ratio = round(target_dist / max(stop_dist, 0.0001), 2)
            if side_norm == "BUY" and target_price <= entry_price:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"For BUY orders, profit target ({target_price}) must be above entry price ({entry_price}).")
            if side_norm == "SELL" and target_price >= entry_price:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"For SELL orders, profit target ({target_price}) must be below entry price ({entry_price}).")

        # 7. Unit-level Loss & Position Size Derivation
        slippage_rate = settings.slippage_bps / 10_000
        commission_rate = settings.transaction_cost_bps / 10_000

        if inst_type == "FOREX":
            norm_pair = normalize_forex_pair(sym)
            pip_val_1_unit = self.calculate_forex_pip_value_usd(norm_pair, entry_price, unit_count=1.0)
            pip_size = get_pip_size(norm_pair)
            pips_at_stop = stop_dist / pip_size
            loss_per_unit = (pips_at_stop * pip_val_1_unit) + (entry_price * commission_rate) + (entry_price * slippage_rate)

            if loss_per_unit <= 0:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price, "Calculated loss per unit was non-positive.")

            raw_units = risk_budget / loss_per_unit
            # Forex micro-lot rounding (1,000 units minimum)
            max_units = math.floor(raw_units / 1000.0) * 1000.0
            if max_units < 1000.0:
                max_units = 1000.0  # Allow 1 micro-lot minimum for simulation
        else:
            # Stock: shares sizing
            loss_per_share = stop_dist + (entry_price * commission_rate) + (entry_price * slippage_rate)
            if loss_per_share <= 0:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price, "Calculated loss per share was non-positive.")

            raw_shares = risk_budget / loss_per_share
            max_units = float(math.floor(raw_shares))
            if max_units < 1.0:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"Risk budget (${risk_budget:,.2f}) insufficient for 1 share at ${entry_price:,.2f} with stop at ${stop_loss_price:,.2f}.")

        # 8. Notional and Cash constraints check
        notional_value = round(max_units * entry_price, 2)
        est_commission = round(notional_value * commission_rate, 2)
        est_slippage = round(notional_value * slippage_rate, 2)
        est_total_risk = round(
            (stop_dist * max_units if inst_type == "STOCK" else stop_dist_pips * self.calculate_forex_pip_value_usd(norm_pair, entry_price, max_units))
            + est_commission + est_slippage,
            2
        )

        # Cash check if cash provided
        if available_cash is not None and inst_type == "STOCK" and notional_value > available_cash:
            # Adjust max_units to available cash
            max_units = float(math.floor(available_cash / entry_price))
            if max_units < 1.0:
                return self._rejected(sym, inst_type, side_norm, account_equity, effective_risk_pct, risk_budget,
                                      entry_price, stop_loss_price,
                                      f"Insufficient available cash (${available_cash:,.2f}) to purchase even 1 share of {sym} at ${entry_price:,.2f}.")
            notional_value = round(max_units * entry_price, 2)
            est_commission = round(notional_value * commission_rate, 2)
            est_slippage = round(notional_value * slippage_rate, 2)
            est_total_risk = round((stop_dist * max_units) + est_commission + est_slippage, 2)

        unit_label = "micro-lots" if inst_type == "FOREX" else "shares"
        display_units = f"{max_units / 1000:.1f}k units" if inst_type == "FOREX" else f"{int(max_units)} shares"

        explanation = (
            f"Risk Budget: ${risk_budget:,.2f} ({effective_risk_pct:.1%} of ${account_equity:,.2f}). "
            f"Stop distance: ${stop_dist:,.2f} ({stop_dist_pct:.2%}"
            f"{f', {stop_dist_pips:.1f} pips' if stop_dist_pips else ''}). "
            f"Permitted size: {display_units} (Notional: ${notional_value:,.2f}). "
            f"Estimated monetary risk at stop: ${est_total_risk:,.2f}."
        )

        return PositionSizingResult(
            symbol=sym,
            instrument_type=inst_type,
            side=side_norm,
            is_valid=True,
            rejection_reason=None,
            account_equity=account_equity,
            risk_pct=round(effective_risk_pct, 4),
            risk_budget_usd=risk_budget,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            stop_distance=round(stop_dist, 4),
            stop_distance_pct=round(stop_dist_pct, 4),
            stop_distance_pips=stop_dist_pips,
            risk_reward_ratio=rr_ratio,
            estimated_loss_per_unit=round(loss_per_share if inst_type == "STOCK" else loss_per_unit, 4),
            max_units=max_units,
            estimated_monetary_risk=est_total_risk,
            position_notional_value=notional_value,
            estimated_commission=est_commission,
            estimated_slippage=est_slippage,
            explanation=explanation,
        )

    def _rejected(
        self,
        symbol: str,
        instrument_type: str,
        side: str,
        equity: float,
        risk_pct: float,
        risk_budget: float,
        entry: float,
        stop: float,
        reason: str,
    ) -> PositionSizingResult:
        return PositionSizingResult(
            symbol=symbol,
            instrument_type=instrument_type,
            side=side,
            is_valid=False,
            rejection_reason=reason,
            account_equity=equity,
            risk_pct=risk_pct,
            risk_budget_usd=risk_budget,
            entry_price=entry,
            stop_loss_price=stop,
            target_price=None,
            stop_distance=abs(entry - stop),
            stop_distance_pct=abs(entry - stop) / max(entry, 0.0001),
            stop_distance_pips=None,
            risk_reward_ratio=None,
            estimated_loss_per_unit=0.0,
            max_units=0.0,
            estimated_monetary_risk=0.0,
            position_notional_value=0.0,
            estimated_commission=0.0,
            estimated_slippage=0.0,
            explanation=f"REJECTED: {reason}",
        )


_sizer_instance: PositionSizer | None = None


def get_position_sizer() -> PositionSizer:
    global _sizer_instance
    if _sizer_instance is None:
        _sizer_instance = PositionSizer()
    return _sizer_instance

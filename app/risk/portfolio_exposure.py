"""Leverage, Margin & Cross-Currency Portfolio Exposure Engine.

Evaluates gross leverage, instrument concentration, margin requirements,
and shared cross-pair currency concentration (e.g. cumulative USD exposure across
EUR/USD, GBP/USD, and USD/JPY).

INVARIANTS:
1. Rejects simulated entries that would breach leverage, margin, or currency ceilings.
2. Explains the exact limit breached.
3. Does not close existing positions; strictly gates new entries.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from app.config import settings
from app.forex.forex_data import is_supported_pair, normalize_forex_pair
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioExposureResult(BaseModel):
    is_approved: bool
    violations: list[str]
    current_portfolio_value: float
    account_equity: float
    proposed_notional: float
    projected_gross_exposure: float
    gross_leverage: float
    max_leverage_allowed: float
    margin_required_projected: float
    free_margin_projected: float
    instrument_exposure_pct: float
    max_instrument_pct: float
    currency_exposures_pct: dict[str, float]
    max_currency_pct: float
    explanation: str


class PortfolioExposureEngine:
    """Manages aggregate portfolio exposure, margin, leverage, and currency concentration."""

    def __init__(
        self,
        max_leverage_stock: float | None = None,
        max_leverage_forex: float | None = None,
        max_instrument_pct: float = 0.20,
        max_currency_pct: float | None = None,
    ):
        self.max_leverage_stock = max_leverage_stock or settings.max_leverage_stock
        self.max_leverage_forex = max_leverage_forex or settings.max_leverage_forex
        self.max_instrument_pct = max_instrument_pct
        self.max_currency_pct = max_currency_pct or settings.max_currency_concentration_pct

    def decompose_forex_currency_exposure(
        self,
        pair: str,
        side: str,
        notional_usd: float,
    ) -> dict[str, float]:
        """Decompose a Forex position into base and quote currency net USD exposures.

        Example:
        - Long EUR/USD ($10,000 notional): Long EUR +$10k, Short USD -$10k.
        - Long USD/JPY ($10,000 notional): Long USD +$10k, Short JPY -$10k.
        """
        norm = normalize_forex_pair(pair)
        parts = norm.split("/")
        base, quote = parts[0], parts[1]
        side_norm = side.upper()

        delta = {}
        if side_norm == "BUY":
            delta[base] = delta.get(base, 0.0) + notional_usd
            delta[quote] = delta.get(quote, 0.0) - notional_usd
        elif side_norm == "SELL":
            delta[base] = delta.get(base, 0.0) - notional_usd
            delta[quote] = delta.get(quote, 0.0) + notional_usd
        return delta

    def evaluate_exposure(
        self,
        proposed_symbol: str,
        proposed_side: str,
        proposed_notional_usd: float,
        account_equity: float,
        existing_positions: dict[str, dict[str, Any]] | None = None,
        instrument_type: str | None = None,
    ) -> PortfolioExposureResult:
        sym = proposed_symbol.strip().upper()
        positions = existing_positions or {}
        equity = max(account_equity, 100.0)

        # 1. Determine instrument type
        if instrument_type is None:
            is_fx = "/" in sym or is_supported_pair(sym) or sym.endswith("=X")
            inst_type = "FOREX" if is_fx else "STOCK"
        else:
            inst_type = instrument_type.upper()

        margin_rate = 0.05 if inst_type == "FOREX" else 0.20  # 20:1 for FX, 5:1 for Stocks
        max_leverage = self.max_leverage_forex if inst_type == "FOREX" else self.max_leverage_stock

        # 2. Aggregate existing positions
        existing_notional_total = 0.0
        existing_margin_total = 0.0
        existing_instrument_notional = 0.0
        currency_net_usd: dict[str, float] = {}

        for p_sym, p_data in positions.items():
            qty = p_data.get("quantity", 0.0)
            price = p_data.get("current_price", p_data.get("avg_entry_price", 0.0))
            notional = abs(qty * price)
            existing_notional_total += notional

            # Margin requirement
            is_p_fx = "/" in p_sym or is_supported_pair(p_sym)
            m_rate = 0.05 if is_p_fx else 0.20
            existing_margin_total += notional * m_rate

            if p_sym.upper() == sym:
                existing_instrument_notional += notional

            # Currency breakdown
            if is_p_fx:
                side_p = "BUY" if qty >= 0 else "SELL"
                decomp = self.decompose_forex_currency_exposure(p_sym, side_p, notional)
                for curr, amt in decomp.items():
                    currency_net_usd[curr] = currency_net_usd.get(curr, 0.0) + amt

        # 3. Add proposed trade to projections
        projected_notional_total = existing_notional_total + proposed_notional_usd
        projected_margin_total = existing_margin_total + (proposed_notional_usd * margin_rate)
        free_margin_projected = equity - projected_margin_total
        projected_inst_notional = existing_instrument_notional + proposed_notional_usd

        # Projected currency exposure
        if inst_type == "FOREX":
            decomp = self.decompose_forex_currency_exposure(sym, proposed_side, proposed_notional_usd)
            for curr, amt in decomp.items():
                currency_net_usd[curr] = currency_net_usd.get(curr, 0.0) + amt

        # Calculate percentages
        gross_leverage = round(projected_notional_total / equity, 2)
        inst_exposure_pct = round(projected_inst_notional / equity, 4)
        currency_exposures_pct = {
            curr: round(abs(amt) / equity, 4) for curr, amt in currency_net_usd.items()
        }

        # 4. Check limit violations
        violations: list[str] = []

        # Leverage check
        if gross_leverage > max_leverage:
            violations.append(
                f"Gross leverage ({gross_leverage:.1f}x) exceeds allowed ceiling of {max_leverage:.1f}x."
            )

        # Margin check
        if free_margin_projected < 0:
            violations.append(
                f"Margin required (${projected_margin_total:,.2f}) exceeds account equity (${equity:,.2f}). Free margin: -${abs(free_margin_projected):,.2f}."
            )

        # Instrument exposure check
        if inst_exposure_pct > self.max_instrument_pct:
            violations.append(
                f"Instrument exposure to {sym} ({inst_exposure_pct:.1%}) exceeds single-instrument cap of {self.max_instrument_pct:.1%}."
            )

        # Cross-currency concentration check
        for curr, pct in currency_exposures_pct.items():
            if pct > self.max_currency_pct:
                violations.append(
                    f"Concentrated currency exposure to {curr} ({pct:.1%}) exceeds maximum currency ceiling of {self.max_currency_pct:.1%}."
                )

        is_approved = len(violations) == 0

        explanation = (
            f"Gross Leverage: {gross_leverage:.1f}x (limit: {max_leverage:.1f}x). "
            f"Projected Margin: ${projected_margin_total:,.2f} (Free Margin: ${free_margin_projected:,.2f}). "
            f"Instrument Exposure: {inst_exposure_pct:.1%}. "
        )
        if currency_exposures_pct:
            top_curr = sorted(currency_exposures_pct.items(), key=lambda x: x[1], reverse=True)[:3]
            top_curr_str = ", ".join([f"{c}: {p:.1%}" for c, p in top_curr])
            explanation += f"Top Currency Exposures: [{top_curr_str}]. "

        if not is_approved:
            explanation += f"REJECTED DUE TO: {'; '.join(violations)}"

        return PortfolioExposureResult(
            is_approved=is_approved,
            violations=violations,
            current_portfolio_value=round(existing_notional_total, 2),
            account_equity=round(equity, 2),
            proposed_notional=round(proposed_notional_usd, 2),
            projected_gross_exposure=round(projected_notional_total, 2),
            gross_leverage=gross_leverage,
            max_leverage_allowed=max_leverage,
            margin_required_projected=round(projected_margin_total, 2),
            free_margin_projected=round(free_margin_projected, 2),
            instrument_exposure_pct=inst_exposure_pct,
            max_instrument_pct=self.max_instrument_pct,
            currency_exposures_pct=currency_exposures_pct,
            max_currency_pct=self.max_currency_pct,
            explanation=explanation,
        )


_exposure_instance: PortfolioExposureEngine | None = None


def get_portfolio_exposure_engine() -> PortfolioExposureEngine:
    global _exposure_instance
    if _exposure_instance is None:
        _exposure_instance = PortfolioExposureEngine()
    return _exposure_instance

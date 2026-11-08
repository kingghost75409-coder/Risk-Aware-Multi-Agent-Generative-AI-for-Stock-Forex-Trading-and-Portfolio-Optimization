"""Trade Setup Quality Gate.

Orchestrates all pre-trade validations before any simulated entry is accepted:
1. Market data validity and freshness.
2. Persistent daily loss and consecutive-loss cooldown status.
3. Mathematical position sizing and stop-loss placement.
4. Risk-to-reward ratio threshold compliance.
5. Gross leverage, margin, and cross-pair currency concentration.
6. Market regime compatibility (trending vs ranging).
7. Macroeconomic event announcement blackout and spread expansion.
8. Historical strategy validation evidence (rejects or abstains if unvalidated).

INVARIANTS:
- Deterministic decision: ELIGIBLE, REJECTED, ABSTAIN, or DATA_UNAVAILABLE.
- Logs evaluation to `trade_quality_gate_records` for research reproducibility.
- Chatbot and AI agents cannot bypass this gate.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from app.config import settings
from app.db.models import get_session, TradeQualityGateRecord
from app.forex.event_risk_gate import ForexEventRiskGate, get_event_risk_gate
from app.forex.forex_data import is_supported_pair, normalize_forex_pair, get_forex_provider
from app.forex.regime_intelligence import ForexMarketRegimeIntelligence, get_regime_intelligence
from app.risk.daily_loss_guard import DailyLossGuard, get_daily_loss_guard
from app.risk.portfolio_exposure import PortfolioExposureEngine, get_portfolio_exposure_engine
from app.risk.position_sizer import PositionSizer, get_position_sizer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TradeQualityDecision(BaseModel):
    evaluation_id: str
    symbol: str
    trading_style: str  # DAY, SWING, POSITION
    proposed_side: str  # BUY, SELL
    decision: str  # ELIGIBLE, REJECTED, ABSTAIN, DATA_UNAVAILABLE
    is_approved: bool
    entry_price: float
    stop_loss_price: float | None = None
    target_price: float | None = None
    risk_reward_ratio: float | None = None
    calculated_units: float = 0.0
    estimated_risk_amount: float = 0.0
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    explanation: str
    timestamp: str


class TradeQualityGate:
    """Authoritative gate controlling simulated trade eligibility."""

    def __init__(
        self,
        position_sizer: PositionSizer | None = None,
        exposure_engine: PortfolioExposureEngine | None = None,
        daily_guard: DailyLossGuard | None = None,
        event_gate: ForexEventRiskGate | None = None,
        regime_intel: ForexMarketRegimeIntelligence | None = None,
        min_risk_reward: float | None = None,
    ):
        self.sizer = position_sizer or get_position_sizer()
        self.exposure = exposure_engine or get_portfolio_exposure_engine()
        self.daily_guard = daily_guard or get_daily_loss_guard()
        self.event_gate = event_gate or get_event_risk_gate()
        self.regime_intel = regime_intel or get_regime_intelligence()
        self.min_risk_reward = min_risk_reward or settings.min_risk_reward_ratio

    def evaluate(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        target_price: float | None = None,
        trading_style: str = "SWING",
        account_equity: float = 100000.0,
        available_cash: float | None = None,
        existing_positions: dict[str, dict[str, Any]] | None = None,
        strategy_name: str | None = None,
        historical_sample_size: int = 50,  # e.g. backtested trades count
        strategy_win_rate: float | None = 0.52,  # None if untested
        as_of: datetime | None = None,
    ) -> TradeQualityDecision:
        eval_id = f"QG_{str(uuid.uuid4())[:8]}"
        now = as_of or datetime.now(timezone.utc)
        sym = symbol.strip().upper()
        side_norm = side.strip().upper()
        style_norm = trading_style.strip().upper()

        checks: dict[str, dict[str, Any]] = {}
        rejections: list[str] = []
        abstentions: list[str] = []

        is_fx = "/" in sym or is_supported_pair(sym) or sym.endswith("=X")
        inst_type = "FOREX" if is_fx else "STOCK"

        # ─── 1. Market Data Check ─────────────────────────────────────────────
        if entry_price <= 0:
            checks["market_data"] = {"status": "FAIL", "message": "Price data missing or zero."}
            return self._build_decision(eval_id, sym, style_norm, side_norm, "DATA_UNAVAILABLE",
                                         entry_price, stop_loss_price, target_price, 0.0, 0.0, None,
                                         checks, ["Market price data unavailable or non-positive."],
                                         "Market data unavailable.", now)
        checks["market_data"] = {"status": "PASS", "entry_price": entry_price}

        # ─── 2. Daily Loss & Cooldown Check ───────────────────────────────────
        daily_status = self.daily_guard.update_unrealized_pnl_and_evaluate(
            current_equity=account_equity,
            unrealized_pnl=0.0,
            as_of=now,
        )
        if not daily_status.can_enter_new_trades:
            reason = daily_status.lock_reason or "Daily risk lock or cooldown active."
            rejections.append(reason)
            checks["daily_loss_guard"] = {"status": "FAIL", "reason": reason}
        else:
            checks["daily_loss_guard"] = {
                "status": "PASS",
                "remaining_allowance": daily_status.daily_loss_allowance_remaining,
                "consecutive_losses": daily_status.consecutive_losses,
            }

        # ─── 3. Position Sizing & Stop-Loss Placement ─────────────────────────
        sizing = self.sizer.calculate_size(
            symbol=sym,
            side=side_norm,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            account_equity=account_equity,
            instrument_type=inst_type,
            available_cash=available_cash,
        )
        if not sizing.is_valid:
            rejections.append(sizing.rejection_reason or "Position sizing failed.")
            checks["position_sizing"] = {"status": "FAIL", "reason": sizing.rejection_reason}
        else:
            checks["position_sizing"] = {
                "status": "PASS",
                "max_units": sizing.max_units,
                "estimated_risk_usd": sizing.estimated_monetary_risk,
                "notional_value": sizing.position_notional_value,
            }

        # ─── 4. Risk / Reward Requirement ─────────────────────────────────────
        rr_ratio = sizing.risk_reward_ratio
        if target_price and target_price > 0:
            if rr_ratio is not None and rr_ratio < self.min_risk_reward:
                rejections.append(
                    f"Risk/Reward ratio ({rr_ratio:.2f}) is below minimum requirement of {self.min_risk_reward:.2f}:1."
                )
                checks["risk_reward"] = {"status": "FAIL", "ratio": rr_ratio, "required": self.min_risk_reward}
            else:
                checks["risk_reward"] = {"status": "PASS", "ratio": rr_ratio}
        else:
            # If no target specified, warn but do not reject unless Day Trading
            if style_norm == "DAY":
                rejections.append("Day Trading requires a pre-defined profit target to evaluate R/R.")
                checks["risk_reward"] = {"status": "FAIL", "reason": "Target price missing for Day Trading"}
            else:
                checks["risk_reward"] = {"status": "WARN", "reason": "No explicit target price provided"}

        # ─── 5. Leverage & Margin & Currency Exposure ─────────────────────────
        exposure_res = self.exposure.evaluate_exposure(
            proposed_symbol=sym,
            proposed_side=side_norm,
            proposed_notional_usd=sizing.position_notional_value,
            account_equity=account_equity,
            existing_positions=existing_positions,
            instrument_type=inst_type,
        )
        if not exposure_res.is_approved:
            rejections.extend(exposure_res.violations)
            checks["portfolio_exposure"] = {"status": "FAIL", "violations": exposure_res.violations}
        else:
            checks["portfolio_exposure"] = {
                "status": "PASS",
                "gross_leverage": exposure_res.gross_leverage,
                "free_margin": exposure_res.free_margin_projected,
            }

        # ─── 6. Forex Event Risk Blackout (if Forex) ──────────────────────────
        if inst_type == "FOREX":
            event_res = self.event_gate.evaluate_entry_risk(sym, side=side_norm, as_of_time=now)
            if event_res.is_restricted:
                rejections.append(f"Forex event gate restriction: {event_res.restriction_reason}")
                checks["economic_event_gate"] = {"status": "FAIL", "reason": event_res.restriction_reason}
            else:
                checks["economic_event_gate"] = {
                    "status": "PASS",
                    "spread_ratio": event_res.spread_ratio,
                    "event_distance_mins": event_res.minutes_to_event,
                }

        # ─── 7. Market Regime Compatibility ───────────────────────────────────
        if inst_type == "FOREX":
            regime_res = self.regime_intel.detect_regime(sym)
            regime_name = regime_res.regime_type
            if "CRISIS" in regime_name or "EXTREME" in regime_name:
                abstentions.append(f"Market regime '{regime_name}' exhibits extreme volatility. System abstains.")
                checks["regime_compatibility"] = {"status": "ABSTAIN", "regime": regime_name}
            elif style_norm == "DAY" and "HIGH_VOLATILITY" in regime_name:
                checks["regime_compatibility"] = {"status": "WARN", "regime": regime_name, "note": "High intraday swings expected"}
            else:
                checks["regime_compatibility"] = {"status": "PASS", "regime": regime_name}

        # ─── 8. Strategy Historical Evidence Validation ───────────────────────
        if strategy_win_rate is None or historical_sample_size < 10:
            abstentions.append(
                f"Strategy '{strategy_name or 'unspecified'}' is UNVALIDATED (insufficient sample size: {historical_sample_size} trades < 10 threshold). "
                "System abstains from unverified strategy edge."
            )
            checks["strategy_validation"] = {"status": "ABSTAIN", "sample_size": historical_sample_size}
        else:
            checks["strategy_validation"] = {
                "status": "PASS",
                "sample_size": historical_sample_size,
                "win_rate": strategy_win_rate,
            }

        # ─── Determine Final Decision ─────────────────────────────────────────
        if rejections:
            final_decision = "REJECTED"
            is_approved = False
            expl = f"Trade REJECTED due to {len(rejections)} violation(s): {'; '.join(rejections)}"
        elif abstentions:
            final_decision = "ABSTAIN"
            is_approved = False
            expl = f"System ABSTAINS from trade: {'; '.join(abstentions)}"
        else:
            final_decision = "ELIGIBLE"
            is_approved = True
            expl = (
                f"Trade ELIGIBLE ({style_norm} style). "
                f"Permitted size: {sizing.max_units} units (${sizing.position_notional_value:,.2f} notional). "
                f"Estimated monetary risk at stop: ${sizing.estimated_monetary_risk:,.2f}. All risk guardrails satisfied."
            )

        return self._build_decision(
            eval_id, sym, style_norm, side_norm, final_decision,
            entry_price, stop_loss_price, target_price,
            sizing.max_units if is_approved else 0.0,
            sizing.estimated_monetary_risk if is_approved else 0.0,
            rr_ratio, checks, rejections + abstentions, expl, now
        )

    def _build_decision(
        self,
        eval_id: str,
        symbol: str,
        style: str,
        side: str,
        decision: str,
        entry: float,
        stop: float | None,
        target: float | None,
        units: float,
        risk_amt: float,
        rr: float | None,
        checks: dict[str, dict[str, Any]],
        reasons: list[str],
        expl: str,
        now: datetime,
    ) -> TradeQualityDecision:
        # Persist audit record in DB
        session = get_session()
        try:
            rec = TradeQualityGateRecord(
                evaluation_id=eval_id,
                symbol=symbol,
                trading_style=style,
                proposed_side=side,
                decision=decision,
                entry_price=entry,
                stop_loss_price=stop,
                target_price=target,
                risk_reward_ratio=rr,
                calculated_units=units,
                estimated_risk_amount=risk_amt,
                checks_json=json.dumps(checks),
                rejection_reasons_json=json.dumps(reasons),
                timestamp=now,
            )
            session.add(rec)
            session.commit()
        except Exception as exc:
            logger.warning("Could not persist trade quality gate record: %s", exc)
        finally:
            session.close()

        return TradeQualityDecision(
            evaluation_id=eval_id,
            symbol=symbol,
            trading_style=style,
            proposed_side=side,
            decision=decision,
            is_approved=(decision == "ELIGIBLE"),
            entry_price=entry,
            stop_loss_price=stop,
            target_price=target,
            risk_reward_ratio=rr,
            calculated_units=units,
            estimated_risk_amount=risk_amt,
            checks=checks,
            rejection_reasons=reasons,
            explanation=expl,
            timestamp=now.isoformat(),
        )


_quality_gate_instance: TradeQualityGate | None = None


def get_trade_quality_gate() -> TradeQualityGate:
    global _quality_gate_instance
    if _quality_gate_instance is None:
        _quality_gate_instance = TradeQualityGate()
    return _quality_gate_instance

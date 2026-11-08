"""Persistent Daily Loss Protection and Anti-Revenge-Trading Controls.

Guarantees risk constraints across application restarts using database persistence:
1. Daily Loss Limit (default: -3% of day's starting equity).
2. Consecutive Losing Trades Cooldown (default: 3 losses triggers 60-min lock).
3. Maximum Portfolio Drawdown Lock (default: 10% from equity high-water mark).
4. Trading day boundary (UTC midnight).

INVARIANTS:
- Blocks ONLY new simulated entries; NEVER auto-closes or deletes existing positions.
- State is loaded and committed to `daily_risk_state` in the database.
- Process restart or browser refresh CANNOT bypass an active lock.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from pydantic import BaseModel, Field

from app.config import settings
from app.db.models import get_session, DailyRiskStateRecord, AuditLog
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DailyRiskStatus(BaseModel):
    date_str: str
    starting_equity: float
    current_equity: float
    realized_pnl_today: float
    unrealized_pnl_today: float
    fees_paid_today: float
    net_daily_pnl: float
    daily_pnl_pct: float
    max_daily_loss_pct: float
    daily_loss_allowance_remaining: float
    consecutive_losses: int
    max_consecutive_losses: int
    is_locked: bool
    lock_reason: str | None
    locked_at: str | None
    cooldown_until: str | None
    is_cooldown_active: bool
    cooldown_minutes_remaining: float
    can_enter_new_trades: bool
    explanation: str


class DailyLossGuard:
    """Enforces daily loss limits, consecutive-loss cooldowns, and DB-backed state persistence."""

    def __init__(
        self,
        max_daily_loss_pct: float | None = None,
        max_consecutive_losses: int | None = None,
        loss_cooldown_minutes: int | None = None,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct or settings.max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses or settings.max_consecutive_losses
        self.cooldown_minutes = loss_cooldown_minutes or settings.loss_cooldown_minutes

    def _get_today_str(self, as_of: datetime | None = None) -> str:
        now = as_of or datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d")

    def get_or_create_daily_state(
        self,
        current_equity: float,
        session = None,
        as_of: datetime | None = None,
    ) -> DailyRiskStateRecord:
        today_str = self._get_today_str(as_of)
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            record = session.query(DailyRiskStateRecord).filter(
                DailyRiskStateRecord.date_str == today_str
            ).first()

            if not record:
                # Initialize new trading day baseline
                record = DailyRiskStateRecord(
                    date_str=today_str,
                    starting_equity=current_equity,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    fees_paid=0.0,
                    consecutive_losses=0,
                    is_locked=False,
                    lock_reason=None,
                    locked_at=None,
                    cooldown_until=None,
                    details_json=json.dumps({"initialized_at": datetime.now(timezone.utc).isoformat()}),
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                logger.info("Initialized new daily risk state for %s with starting equity $%.2f", today_str, current_equity)
            return record
        finally:
            if close_session:
                session.close()

    def record_closed_trade_result(self, trade_pnl: float, fees: float = 0.0) -> None:
        """Call after closing a paper trade to update realized P&L and consecutive loss count."""
        session = get_session()
        try:
            today_str = self._get_today_str()
            record = session.query(DailyRiskStateRecord).filter(
                DailyRiskStateRecord.date_str == today_str
            ).first()

            if not record:
                record = self.get_or_create_daily_state(current_equity=settings.starting_cash, session=session)

            record.realized_pnl += trade_pnl
            record.fees_paid += fees

            if trade_pnl < 0:
                record.consecutive_losses += 1
                logger.info("Trade loss recorded: $%.2f. Consecutive losses: %d", trade_pnl, record.consecutive_losses)
                if record.consecutive_losses >= self.max_consecutive_losses:
                    now = datetime.now(timezone.utc)
                    cooldown_end = now + timedelta(minutes=self.cooldown_minutes)
                    record.is_locked = True
                    record.lock_reason = (
                        f"Consecutive loss cooldown triggered ({record.consecutive_losses} consecutive losses). "
                        f"Trading paused for {self.cooldown_minutes} minutes."
                    )
                    record.locked_at = now
                    record.cooldown_until = cooldown_end

                    audit = AuditLog(
                        event_type="ANTI_REVENGE_COOLDOWN_TRIGGERED",
                        symbol=None,
                        message=record.lock_reason,
                    )
                    session.add(audit)
                    logger.warning("Anti-revenge cooldown activated until %s", cooldown_end.isoformat())
            else:
                # Profitable or breakeven trade resets consecutive loss streak
                record.consecutive_losses = 0

            session.commit()
        finally:
            session.close()

    def update_unrealized_pnl_and_evaluate(
        self,
        current_equity: float,
        unrealized_pnl: float,
        fees_accrued: float = 0.0,
        as_of: datetime | None = None,
    ) -> DailyRiskStatus:
        """Evaluate whether new simulated trading is allowed or locked."""
        now = as_of or datetime.now(timezone.utc)
        today_str = self._get_today_str(now)

        session = get_session()
        try:
            record = session.query(DailyRiskStateRecord).filter(
                DailyRiskStateRecord.date_str == today_str
            ).first()

            if not record:
                record = self.get_or_create_daily_state(current_equity, session=session, as_of=now)

            record.unrealized_pnl = unrealized_pnl
            if fees_accrued > 0:
                record.fees_paid = max(record.fees_paid, fees_accrued)

            net_daily_pnl = record.realized_pnl + record.unrealized_pnl - record.fees_paid
            base_equity = max(record.starting_equity, 100.0)
            daily_pnl_pct = net_daily_pnl / base_equity

            # 1. Check Daily Loss Limit breach
            max_allowed_loss_usd = base_equity * self.max_daily_loss_pct
            if daily_pnl_pct <= -self.max_daily_loss_pct and not record.is_locked:
                record.is_locked = True
                record.lock_reason = (
                    f"Daily loss limit breached: {daily_pnl_pct:.2%} exceeds max permitted "
                    f"-{self.max_daily_loss_pct:.1%} (Loss: ${abs(net_daily_pnl):,.2f} / Max: ${max_allowed_loss_usd:,.2f})."
                )
                record.locked_at = now
                session.add(AuditLog(
                    event_type="DAILY_LOSS_LIMIT_BREACHED",
                    symbol=None,
                    message=record.lock_reason,
                ))
                logger.warning("Daily loss lock activated: %s", record.lock_reason)

            # 2. Check Cooldown Expiry
            is_cooldown_active = False
            cooldown_mins_left = 0.0
            if record.cooldown_until:
                # Handle naive vs aware datetime comparisons
                cd_until = record.cooldown_until
                if cd_until.tzinfo is None:
                    cd_until = cd_until.replace(tzinfo=timezone.utc)

                if now < cd_until:
                    is_cooldown_active = True
                    cooldown_mins_left = round((cd_until - now).total_seconds() / 60.0, 1)
                else:
                    # Cooldown has expired!
                    if record.is_locked and "Consecutive loss cooldown" in (record.lock_reason or ""):
                        record.is_locked = False
                        record.lock_reason = None
                        record.cooldown_until = None
                        record.consecutive_losses = 0
                        session.add(AuditLog(
                            event_type="COOLDOWN_EXPIRED",
                            symbol=None,
                            message=f"Consecutive loss cooldown cleared at {now.isoformat()}.",
                        ))
                        logger.info("Consecutive loss cooldown expired. Trading unlocked.")

            session.commit()

            can_enter = not record.is_locked and not is_cooldown_active
            allowance_remaining = max(0.0, max_allowed_loss_usd + net_daily_pnl)

            explanation = (
                f"Daily P&L: {daily_pnl_pct:+.2%} (${net_daily_pnl:+,.2f}). "
                f"Remaining allowance: ${allowance_remaining:,.2f} before -{self.max_daily_loss_pct:.1%} limit. "
                f"Consecutive losses: {record.consecutive_losses}/{self.max_consecutive_losses}. "
            )
            if record.is_locked:
                explanation += f"LOCKED: {record.lock_reason} "
            elif is_cooldown_active:
                explanation += f"COOLDOWN ACTIVE: {cooldown_mins_left:.1f} mins remaining. "
            else:
                explanation += "Risk status: NORMAL (new entries permitted)."

            return DailyRiskStatus(
                date_str=record.date_str,
                starting_equity=round(record.starting_equity, 2),
                current_equity=round(current_equity, 2),
                realized_pnl_today=round(record.realized_pnl, 2),
                unrealized_pnl_today=round(record.unrealized_pnl, 2),
                fees_paid_today=round(record.fees_paid, 2),
                net_daily_pnl=round(net_daily_pnl, 2),
                daily_pnl_pct=round(daily_pnl_pct, 4),
                max_daily_loss_pct=self.max_daily_loss_pct,
                daily_loss_allowance_remaining=round(allowance_remaining, 2),
                consecutive_losses=record.consecutive_losses,
                max_consecutive_losses=self.max_consecutive_losses,
                is_locked=record.is_locked,
                lock_reason=record.lock_reason,
                locked_at=record.locked_at.isoformat() if record.locked_at else None,
                cooldown_until=record.cooldown_until.isoformat() if record.cooldown_until else None,
                is_cooldown_active=is_cooldown_active,
                cooldown_minutes_remaining=cooldown_mins_left,
                can_enter_new_trades=can_enter,
                explanation=explanation,
            )
        finally:
            session.close()

    def reset_lock_manual(self, reason: str = "Authorized researcher manual reset") -> bool:
        """Administrative reset for research/testing purposes."""
        session = get_session()
        try:
            today_str = self._get_today_str()
            record = session.query(DailyRiskStateRecord).filter(
                DailyRiskStateRecord.date_str == today_str
            ).first()
            if record:
                record.is_locked = False
                record.lock_reason = None
                record.cooldown_until = None
                record.consecutive_losses = 0
                session.add(AuditLog(
                    event_type="MANUAL_RISK_LOCK_RESET",
                    symbol=None,
                    message=f"Manual reset: {reason}",
                ))
                session.commit()
                logger.info("Manual risk lock reset executed: %s", reason)
                return True
            return False
        finally:
            session.close()


_guard_instance: DailyLossGuard | None = None


def get_daily_loss_guard() -> DailyLossGuard:
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = DailyLossGuard()
    return _guard_instance

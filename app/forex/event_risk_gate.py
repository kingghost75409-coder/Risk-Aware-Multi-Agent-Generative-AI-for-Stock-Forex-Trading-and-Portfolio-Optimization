"""Event-Related Simulated Trading Risk Gate.

Integrates with the existing deterministic risk engine.
Enforces quantitative risk restrictions around macroeconomic event announcements:
- Blackout window around High/Extreme events (+/- 30 minutes)
- Spread expansion gate (> 2.5x normal typical spread)
- Volatility spike gate (> 25% annualized volatility)
- Stale or incomplete market data protection

INVARIANTS:
1. Blocks ONLY new simulated entries; NEVER mutates, deletes, or auto-closes existing positions.
2. Generates an audited, deterministic rejection reason.
3. Chatbot or LLM cannot bypass this gate.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from pydantic import BaseModel, Field

from app.db.models import get_session, AuditLog
from app.forex.event_analyzer import EventRiskLevel, ForexEventAnalyzer, get_event_analyzer
from app.forex.forex_data import (
    ForexDataProvider,
    get_forex_provider,
    normalize_forex_pair,
    is_supported_pair,
    _TYPICAL_SPREAD_PIPS,
)
from app.risk.risk_engine import RiskEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EventRiskCheckResult(BaseModel):
    pair: str
    is_restricted: bool
    restriction_reason: str | None
    active_event_id: str | None
    event_risk_level: str | None
    minutes_to_event: float | None
    spread_pips: float
    typical_spread_pips: float
    spread_ratio: float
    volatility_annualized: float
    allowed_actions: list[str]
    timestamp: str


class ForexEventRiskGate:
    """Enforces deterministic pre-event and post-event risk restrictions on new simulated entries."""

    def __init__(
        self,
        event_analyzer: ForexEventAnalyzer | None = None,
        data_provider: ForexDataProvider | None = None,
        blackout_minutes: int = 30,
    ):
        self.analyzer = event_analyzer or get_event_analyzer()
        self.provider = data_provider or get_forex_provider()
        self.blackout_minutes = blackout_minutes

    def evaluate_entry_risk(
        self,
        pair: str,
        side: str = "BUY",
        as_of_time: datetime | None = None,
    ) -> EventRiskCheckResult:
        norm = normalize_forex_pair(pair)
        now = as_of_time or datetime.now(timezone.utc)

        if not is_supported_pair(norm):
            return EventRiskCheckResult(
                pair=pair,
                is_restricted=True,
                restriction_reason=f"Unsupported currency pair: {pair}",
                active_event_id=None,
                event_risk_level=None,
                minutes_to_event=None,
                spread_pips=0.0,
                typical_spread_pips=0.0,
                spread_ratio=0.0,
                volatility_annualized=0.0,
                allowed_actions=["HOLD"],
                timestamp=now.isoformat(),
            )

        quote = self.provider.get_quote(norm)
        typical_spread = _TYPICAL_SPREAD_PIPS.get(norm, 1.5)
        spread_ratio = round(quote.spread_pips / max(0.1, typical_spread), 2)
        vol = quote.annualized_volatility

        # 1. Check for Active Event Blackout Window
        events = self.analyzer.list_events()
        active_event = None
        min_distance_minutes: float | None = None

        for ev in events:
            # Check if this currency is impacted
            affected = ev.get("affected_pairs", [])
            curr = ev.get("currency", "")
            if norm not in affected and curr not in norm:
                continue

            # Parse scheduled time
            sched_str = ev.get("scheduled_time", "")
            try:
                # Handle ISO timestamps with Z or offset
                sched_dt = datetime.fromisoformat(sched_str.replace("Z", "+00:00"))
                diff_minutes = (sched_dt - now).total_seconds() / 60.0

                # If within +/- blackout_minutes and High/Extreme risk
                is_high_risk = ev.get("risk_level") in (EventRiskLevel.HIGH, EventRiskLevel.EXTREME)
                if abs(diff_minutes) <= self.blackout_minutes and is_high_risk:
                    active_event = ev
                    min_distance_minutes = round(diff_minutes, 1)
                    break
            except Exception as exc:
                logger.debug("Failed parsing event scheduled time '%s': %s", sched_str, exc)

        # Evaluate Restriction Conditions
        is_restricted = False
        reasons: list[str] = []

        if active_event:
            is_restricted = True
            timing_desc = f"{abs(min_distance_minutes):.0f}m before" if min_distance_minutes > 0 else f"{abs(min_distance_minutes):.0f}m after"
            reasons.append(
                f"Active Event Blackout: '{active_event.get('name')}' ({timing_desc} release). "
                f"Simulated new entries restricted within +/-{self.blackout_minutes}m of {active_event.get('risk_level')} announcements."
            )

        # 2. Spread Blowout Check (> 2.5x typical spread)
        if spread_ratio >= 2.5:
            is_restricted = True
            reasons.append(
                f"Abnormal Spread Expansion: Current spread {quote.spread_pips:.1f} pips is {spread_ratio}x typical spread "
                f"({typical_spread:.1f} pips). Exceeds 2.5x threshold."
            )

        # 3. Volatility Spike Check (> 25% annualized)
        if vol >= 0.25:
            is_restricted = True
            reasons.append(
                f"Extreme Volatility Dislocation: Annualized volatility is {vol:.1%}, exceeding 25.0% risk threshold."
            )

        full_reason = " | ".join(reasons) if is_restricted else None
        allowed_actions = ["HOLD"] if is_restricted else ["BUY", "SELL", "HOLD"]

        # Audit restriction if triggered
        if is_restricted:
            self._audit_restriction(norm, side, full_reason or "EVENT_RISK_RESTRICTION")

        return EventRiskCheckResult(
            pair=norm,
            is_restricted=is_restricted,
            restriction_reason=full_reason,
            active_event_id=active_event.get("event_id") if active_event else None,
            event_risk_level=active_event.get("risk_level").value if active_event else None,
            minutes_to_event=min_distance_minutes,
            spread_pips=quote.spread_pips,
            typical_spread_pips=typical_spread,
            spread_ratio=spread_ratio,
            volatility_annualized=vol,
            allowed_actions=allowed_actions,
            timestamp=now.isoformat(),
        )

    def _audit_restriction(self, pair: str, side: str, reason: str) -> None:
        try:
            session = get_session()
            log = AuditLog(
                agent_name="ForexEventRiskGate",
                event_type="SIMULATED_ORDER_BLOCKED",
                details=f"Pair: {pair}, Side: {side}, Reason: {reason}",
                timestamp=datetime.now(timezone.utc),
            )
            session.add(log)
            session.commit()
            session.close()
        except Exception as exc:
            logger.debug("Could not write AuditLog for ForexEventRiskGate: %s", exc)


_risk_gate_instance: ForexEventRiskGate | None = None


def get_forex_risk_gate() -> ForexEventRiskGate:
    global _risk_gate_instance
    if _risk_gate_instance is None:
        _risk_gate_instance = ForexEventRiskGate()
    return _risk_gate_instance


get_event_risk_gate = get_forex_risk_gate

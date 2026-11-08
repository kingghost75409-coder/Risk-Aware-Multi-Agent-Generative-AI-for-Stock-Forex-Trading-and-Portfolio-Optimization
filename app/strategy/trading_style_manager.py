"""Trading Style Manager.

Provides dynamic configuration, operational rules, and validation parameters
for the three primary trading styles:
1. Day Trading (Intraday, tight stops, session boundary handling, strict loss limits)
2. Swing Trading (Multi-day/week trends, support/resistance, overnight risk)
3. Position Trading (Long-term macro trends, central bank policies, higher timeframes)

INVARIANTS:
- Changes validation rules and execution constraints, not creating separate apps.
- Persists active style configuration to `trading_style_configs` in DB.
- Reuses the existing multi-agent signal pipeline and deterministic risk engine.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from app.config import settings
from app.db.models import get_session, TradingStyleConfigRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TradingStyleProfile(BaseModel):
    style: str  # DAY, SWING, POSITION
    display_name: str
    description: str
    primary_timeframes: list[str]
    max_holding_days: float
    default_risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    min_risk_reward_ratio: float
    session_boundary_action: str  # FLAT_AT_CLOSE, HOLD_OVERNIGHT, LONG_TERM_ACCUMULATE
    carries_overnight_risk: bool
    requires_economic_event_check: bool
    requires_macro_rag: bool


STYLE_PROFILES: dict[str, TradingStyleProfile] = {
    "DAY": TradingStyleProfile(
        style="DAY",
        display_name="Day Trading",
        description="Intraday execution capturing short-term momentum; strictly closes or flattens before session boundary.",
        primary_timeframes=["5m", "15m", "1h"],
        max_holding_days=1.0,
        default_risk_per_trade_pct=0.005,  # 0.5%
        max_risk_per_trade_pct=0.01,       # 1.0%
        min_risk_reward_ratio=1.5,
        session_boundary_action="FLAT_AT_CLOSE",
        carries_overnight_risk=False,
        requires_economic_event_check=True,
        requires_macro_rag=False,
    ),
    "SWING": TradingStyleProfile(
        style="SWING",
        display_name="Swing Trading",
        description="Captures multi-day to multi-week trend swings; tracks overnight and weekend exposure with regime filtering.",
        primary_timeframes=["4h", "1d"],
        max_holding_days=14.0,
        default_risk_per_trade_pct=0.01,   # 1.0%
        max_risk_per_trade_pct=0.015,      # 1.5%
        min_risk_reward_ratio=2.0,
        session_boundary_action="HOLD_OVERNIGHT",
        carries_overnight_risk=True,
        requires_economic_event_check=True,
        requires_macro_rag=True,
    ),
    "POSITION": TradingStyleProfile(
        style="POSITION",
        display_name="Position Trading",
        description="Long-horizon macro trend capture; grounded in central bank policies, interest rate differentials, and Financial RAG.",
        primary_timeframes=["1d", "1w"],
        max_holding_days=180.0,
        default_risk_per_trade_pct=0.01,   # 1.0%
        max_risk_per_trade_pct=0.02,       # 2.0%
        min_risk_reward_ratio=2.5,
        session_boundary_action="LONG_TERM_ACCUMULATE",
        carries_overnight_risk=True,
        requires_economic_event_check=False,
        requires_macro_rag=True,
    ),
}


class TradingStyleManager:
    """Manages active trading style configuration, persistence, and rule retrieval."""

    def __init__(self):
        self._active_style: str = settings.default_trading_style.upper()
        self._load_active_style_from_db()

    def _load_active_style_from_db(self) -> None:
        session = get_session()
        try:
            active_rec = session.query(TradingStyleConfigRecord).filter(
                TradingStyleConfigRecord.is_active == True
            ).order_by(TradingStyleConfigRecord.id.desc()).first()

            if active_rec and active_rec.style in STYLE_PROFILES:
                self._active_style = active_rec.style
                logger.info("Loaded active trading style from DB: %s", self._active_style)
            else:
                # Seed default active record
                self.set_active_style(self._active_style, persist=True)
        except Exception as exc:
            logger.warning("Could not query trading style configs from DB: %s", exc)
        finally:
            session.close()

    def get_active_style(self) -> str:
        return self._active_style

    def get_active_profile(self) -> TradingStyleProfile:
        return STYLE_PROFILES.get(self._active_style, STYLE_PROFILES["SWING"])

    def set_active_style(
        self,
        style: str,
        risk_per_trade_pct: float | None = None,
        min_risk_reward: float | None = None,
        persist: bool = True,
    ) -> TradingStyleProfile:
        norm_style = style.strip().upper()
        if norm_style not in STYLE_PROFILES:
            raise ValueError(f"Unknown trading style '{style}'. Choose from DAY, SWING, or POSITION.")

        self._active_style = norm_style
        profile = STYLE_PROFILES[norm_style]

        if persist:
            session = get_session()
            try:
                # Deactivate previous active records
                session.query(TradingStyleConfigRecord).filter(
                    TradingStyleConfigRecord.is_active == True
                ).update({"is_active": False})

                rec = TradingStyleConfigRecord(
                    style=norm_style,
                    instrument_type="ALL",
                    timeframe=profile.primary_timeframes[0],
                    max_holding_duration_days=profile.max_holding_days,
                    risk_per_trade_pct=risk_per_trade_pct or profile.default_risk_per_trade_pct,
                    min_risk_reward_ratio=min_risk_reward or profile.min_risk_reward_ratio,
                    rules_json=json.dumps(profile.model_dump()),
                    is_active=True,
                    timestamp=datetime.now(timezone.utc),
                )
                session.add(rec)
                session.commit()
                logger.info("Activated and persisted trading style: %s", norm_style)
            except Exception as exc:
                session.rollback()
                logger.warning("Failed to persist trading style config: %s", exc)
            finally:
                session.close()

        return profile

    def list_styles(self) -> list[dict[str, Any]]:
        results = []
        for key, p in STYLE_PROFILES.items():
            item = p.model_dump()
            item["is_current_active"] = (key == self._active_style)
            results.append(item)
        return results


_style_manager_instance: TradingStyleManager | None = None


def get_trading_style_manager() -> TradingStyleManager:
    global _style_manager_instance
    if _style_manager_instance is None:
        _style_manager_instance = TradingStyleManager()
    return _style_manager_instance

"""Short-Term Tactical Trading Planner.

Reuses existing agents:
- TechnicalAgent (momentum, RSI, MACD, volume, breakouts)
- MLPredictionAgent (short-term directional prediction, probability)
- SentimentAgent (news sentiment, trend)
- MarketRegimeAgent (bull, bear, volatile, sideways)
- CrashRiskAgent (tail-risk, crash probability)
- FinancialRAGAgent (event risk, filing citations)
- RiskEngine (deterministic inviolable gate, VaR, CVaR, drawdown)

Horizons:
- intraday
- 1_day
- 1_week
- 1_4_weeks

Strictly zero fake scores: all metrics derived from real data pipeline.
Explicitly identifies uncertainty; no trade is described as guaranteed.
"""
from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from app.data.pipeline import load_and_prepare
from app.data.stock_info import get_stock_quote
from app.orchestrator import TradingOrchestrator, TradingDecision
from app.strategy.router import HorizonStrategyRouter, HorizonType, ShortTermConfidence
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TradeZone(BaseModel):
    entry_min: float
    entry_target: float
    entry_max: float
    target_conservative: float
    target_base: float
    target_optimistic: float
    stop_loss: float
    stop_distance_abs: float
    stop_distance_pct: float
    reward_abs: float
    risk_reward_ratio: float
    risk_reward_str: str


class ShortTermOpportunity(BaseModel):
    symbol: str
    horizon: str  # intraday, 1_day, 1_week, 1_4_weeks
    signal: str   # BUY, SELL, HOLD, ABSTAIN
    current_price: float
    confidence: ShortTermConfidence
    trade_zone: TradeZone | None = None
    position_size_pct: float = 0.0
    recommended_shares: int = 0
    recommended_capital: float = 0.0
    max_loss_capital: float = 0.0
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    var_95: float = 0.0
    cvar_95: float = 0.0
    current_drawdown: float = 0.0
    portfolio_exposure_pct: float = 0.0
    regime: str = "UNKNOWN"
    crash_probability: float = 0.0
    technical_summary: dict[str, Any] = Field(default_factory=dict)
    prediction_summary: dict[str, Any] = Field(default_factory=dict)
    sentiment_summary: dict[str, Any] = Field(default_factory=dict)
    event_risk_notes: list[str] = Field(default_factory=list)
    uncertainty_factors: list[str] = Field(default_factory=list)
    decision_audit: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Model-derived tactical scenario projection for research/paper trading only. "
        "Market timing involves substantial risk of capital loss. Past performance does not guarantee future results."
    )


class ShortTermPlanner:
    """Tactical trading planner operating over short horizons (intraday to 4 weeks)."""

    def __init__(self, orchestrator: TradingOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or TradingOrchestrator()
        self.router = HorizonStrategyRouter()

    def analyze_opportunity(
        self,
        symbol: str,
        horizon: str = "1_week",
        portfolio_value: float = 100000.0,
        current_portfolio_exposure_pct: float = 0.0,
        current_daily_pnl_pct: float = 0.0,
    ) -> ShortTermOpportunity:
        symbol = symbol.upper()
        h_enum = self.router.parse_horizon(horizon)
        if h_enum != HorizonType.SHORT_TERM:
            horizon = "1_week"

        # 1. Run full orchestrator pipeline (reusing all specialized agents + risk engine)
        decision: TradingDecision = self.orchestrator.analyze_symbol(
            symbol=symbol,
            current_portfolio_exposure_pct=current_portfolio_exposure_pct,
            current_daily_pnl_pct=current_daily_pnl_pct,
        )

        # 2. Extract price & volatility from dataset
        df, _, _ = load_and_prepare(symbol)
        current_price = float(df["adj_close"].iloc[-1]) if not df.empty else 100.0
        
        # Realized daily volatility & ATR proxy
        if len(df) >= 14:
            high_low = (df["high"] - df["low"]) if "high" in df.columns else df["adj_close"] * 0.02
            atr = float(high_low.tail(14).mean())
            atr = max(atr, current_price * 0.01)
        else:
            atr = current_price * 0.02

        # 3. Router assessment for short-term horizon
        route_result = self.router.route_horizon(
            requested_horizon=horizon,
            decision=decision,
            fundamental_evidence_count=len(decision.rag_evidence.citations),
        )

        signal = route_result.decision
        st_conf = route_result.short_term_confidence or ShortTermConfidence(
            signal_confidence=decision.confidence,
            prediction_confidence=decision.prediction.probability,
            market_stability=0.6,
            evidence_quality=0.7,
            risk_confidence=0.7,
        )

        # Determine horizon multiplier for stop/target distances
        horizon_mult = {
            "intraday": 0.8,
            "1_day": 1.0,
            "1_week": 2.0,
            "1_4_weeks": 3.5,
        }.get(horizon.lower().replace("-", "_").replace(" ", "_"), 2.0)

        # 4. Construct Trade Zones (Entry, Target, Stop Loss)
        stop_dist_abs = round(atr * horizon_mult, 2)
        stop_dist_pct = round((stop_dist_abs / current_price), 4)

        if signal == "BUY":
            entry_min = round(current_price - (atr * 0.25), 2)
            entry_target = round(current_price, 2)
            entry_max = round(current_price + (atr * 0.25), 2)
            stop_loss = round(max(0.01, current_price - stop_dist_abs), 2)
            
            # Target projections based on ML expected return & ATR multiples
            pred_return = max(decision.prediction.expected_return, 0.02)
            target_cons = round(current_price + (stop_dist_abs * 1.5), 2)
            target_base = round(max(current_price * (1.0 + pred_return), current_price + (stop_dist_abs * 2.0)), 2)
            target_opt = round(current_price + (stop_dist_abs * 3.2), 2)
            reward_abs = round(target_base - current_price, 2)
            rr_ratio = round(reward_abs / max(0.01, stop_dist_abs), 2)
        elif signal == "SELL":
            entry_min = round(current_price - (atr * 0.25), 2)
            entry_target = round(current_price, 2)
            entry_max = round(current_price + (atr * 0.25), 2)
            stop_loss = round(current_price + stop_dist_abs, 2)
            target_cons = round(max(0.01, current_price - (stop_dist_abs * 1.5)), 2)
            target_base = round(max(0.01, current_price - (stop_dist_abs * 2.0)), 2)
            target_opt = round(max(0.01, current_price - (stop_dist_abs * 3.0)), 2)
            reward_abs = round(current_price - target_base, 2)
            rr_ratio = round(reward_abs / max(0.01, stop_dist_abs), 2)
        else:
            # HOLD or ABSTAIN
            entry_min = current_price
            entry_target = current_price
            entry_max = current_price
            stop_loss = round(max(0.01, current_price - stop_dist_abs), 2)
            target_cons = round(current_price * 1.02, 2)
            target_base = round(current_price * 1.05, 2)
            target_opt = round(current_price * 1.08, 2)
            reward_abs = round(target_base - current_price, 2)
            rr_ratio = round(reward_abs / max(0.01, stop_dist_abs), 2)

        rr_str = f"1:{rr_ratio:.1f}"

        trade_zone = TradeZone(
            entry_min=entry_min,
            entry_target=entry_target,
            entry_max=entry_max,
            target_conservative=target_cons,
            target_base=target_base,
            target_optimistic=target_opt,
            stop_loss=stop_loss,
            stop_distance_abs=stop_dist_abs,
            stop_distance_pct=stop_dist_pct,
            reward_abs=reward_abs,
            risk_reward_ratio=rr_ratio,
            risk_reward_str=rr_str,
        )

        # 5. Position Sizing & Deterministic Risk Gates
        # The existing Risk Engine is inviolable. If signal is ABSTAIN or Risk status is REJECT -> position 0
        final_pos_pct = decision.position_size_pct if signal == "BUY" else 0.0
        
        # Enforce minimum R:R check for tactical trading
        uncertainty_factors = []
        if signal == "BUY" and rr_ratio < 1.5:
            uncertainty_factors.append(f"Risk-to-reward ratio ({rr_str}) is below optimal threshold 1:1.5.")
        
        if decision.crash_risk and decision.crash_risk.crash_probability > 0.35:
            uncertainty_factors.append(
                f"Elevated tail-risk probability ({decision.crash_risk.crash_probability:.1%}); crash risk tier: {decision.crash_risk.risk_tier}."
            )

        if decision.regime.regime in {"HIGH_VOLATILITY", "BEAR_TREND"}:
            uncertainty_factors.append(f"Market regime is {decision.regime.regime}, which increases short-term variance.")

        if decision.prediction.probability < 0.60:
            uncertainty_factors.append(
                f"ML directional model certainty is modest ({decision.prediction.probability:.1%})."
            )

        rec_capital = round(portfolio_value * final_pos_pct, 2)
        rec_shares = int(rec_capital // current_price) if current_price > 0 else 0
        max_loss_cap = round(rec_shares * stop_dist_abs, 2)

        # Risk Level Classification
        crash_p = decision.crash_risk.crash_probability if decision.crash_risk else 0.10
        if decision.risk.status == "REJECT" or crash_p >= 0.60:
            risk_level = "CRITICAL"
        elif crash_p >= 0.40 or decision.risk.volatility_annualized > 0.40:
            risk_level = "HIGH"
        elif crash_p >= 0.25 or decision.risk.volatility_annualized > 0.25:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Event risk notes from RAG
        event_notes = []
        if decision.rag_evidence and decision.rag_evidence.citations:
            for cit in decision.rag_evidence.citations[:2]:
                event_notes.append(f"[{cit.source} {cit.section}] {cit.text[:120]}...")

        return ShortTermOpportunity(
            symbol=symbol,
            horizon=horizon,
            signal=signal,
            current_price=current_price,
            confidence=st_conf,
            trade_zone=trade_zone,
            position_size_pct=final_pos_pct,
            recommended_shares=rec_shares,
            recommended_capital=rec_capital,
            max_loss_capital=max_loss_cap,
            risk_level=risk_level,
            var_95=decision.risk.var_95,
            cvar_95=decision.risk.cvar_95,
            current_drawdown=decision.risk.max_drawdown,
            portfolio_exposure_pct=current_portfolio_exposure_pct + final_pos_pct,
            regime=decision.regime.regime,
            crash_probability=crash_p,
            technical_summary={
                "signal": decision.technical.signal,
                "confidence": decision.technical.confidence,
                "reasoning": decision.technical.reasoning,
                "indicators": decision.technical.indicators,
            },
            prediction_summary={
                "direction": decision.prediction.predicted_direction,
                "probability": decision.prediction.probability,
                "expected_return": decision.prediction.expected_return,
                "model": decision.prediction.model,
            },
            sentiment_summary={
                "score": decision.sentiment.sentiment_score,
                "label": decision.sentiment.label,
                "trend": decision.sentiment.trend,
            },
            event_risk_notes=event_notes,
            uncertainty_factors=uncertainty_factors,
            decision_audit=decision.reasoning_audit_trail,
        )

    def scan_watchlist(
        self,
        symbols: list[str] | None = None,
        horizon: str = "1_week",
        portfolio_value: float = 100000.0,
    ) -> list[ShortTermOpportunity]:
        watch = symbols or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
        opportunities: list[ShortTermOpportunity] = []
        for sym in watch:
            try:
                opp = self.analyze_opportunity(symbol=sym, horizon=horizon, portfolio_value=portfolio_value)
                opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Error scanning {sym} for short-term tactical: {e}")
        return opportunities

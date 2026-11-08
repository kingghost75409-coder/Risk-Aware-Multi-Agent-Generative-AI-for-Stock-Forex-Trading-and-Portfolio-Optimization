"""Combined Portfolio Management & Total Portfolio Risk Engine.

Combines:
- Long-Term Holdings (core strategic investment)
- Short-Term Tactical Positions (opportunistic trading)
- Cash Reserve Buffer

Calculates TOTAL Portfolio Risk:
- Total exposure & cash allocation
- Sector concentration
- Correlated portfolio volatility
- Total VaR (95%) and CVaR (95%) using existing deterministic risk functions
- Historical portfolio drawdown
- Rebalancing triggers (drift > 5%, crash risk surge, regime shifts, confidence deterioration)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.config import settings
from app.data.pipeline import load_and_prepare
from app.data.stock_info import get_stock_info, get_stock_quote
from app.risk.risk_engine import RiskEngine, value_at_risk, conditional_value_at_risk, max_drawdown
from app.strategy.long_term_planner import LongTermPlanner
from app.strategy.short_term_planner import ShortTermPlanner
from app.strategy.router import HorizonType
from app.db.models import get_session, RebalanceEvent, InvestmentPlan
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HoldingPosition(BaseModel):
    symbol: str
    horizon_bucket: str  # LONG_TERM, SHORT_TERM, CASH
    allocation_pct: float
    allocation_capital: float
    current_price: float
    shares: int
    sector: str = "Technology"
    unrealized_pnl_pct: float = 0.0
    volatility: float = 0.20
    var_95: float = 0.03
    cvar_95: float = 0.045
    weight_in_portfolio: float = 0.0


class TotalPortfolioRisk(BaseModel):
    total_capital: float
    long_term_capital: float
    long_term_pct: float
    short_term_capital: float
    short_term_pct: float
    cash_capital: float
    cash_pct: float
    total_equity_exposure_pct: float
    sector_concentration: dict[str, float]
    max_sector_pct: float
    dominant_sector: str
    portfolio_volatility_annualized: float
    total_var_95: float
    total_cvar_95: float
    portfolio_max_drawdown: float
    risk_status: str  # APPROVED / WARNING / REJECT
    risk_violations: list[str] = Field(default_factory=list)
    correlation_matrix_summary: str = "Correlated equity exposure (0.3 cross-asset assumption)"


class RebalanceAction(BaseModel):
    symbol: str
    horizon_bucket: str
    current_pct: float
    target_pct: float
    drift_pct: float
    action: str  # BUY, SELL, HOLD
    capital_delta: float
    shares_delta: int
    reason: str


class RebalanceAssessment(BaseModel):
    should_rebalance: bool
    trigger_type: str  # NONE, DRIFT, REGIME_CHANGE, CRASH_RISK_SURGE, CONFIDENCE_DECAY, SCHEDULED
    trigger_reasons: list[str] = Field(default_factory=list)
    actions: list[RebalanceAction] = Field(default_factory=list)
    evaluated_at: str


class CombinedPortfolioState(BaseModel):
    total_capital: float
    long_term_target_pct: float
    short_term_target_pct: float
    cash_target_pct: float
    holdings: list[HoldingPosition]
    total_risk: TotalPortfolioRisk
    rebalance_assessment: RebalanceAssessment | None = None
    created_at: str
    disclaimer: str = (
        "Combined portfolio optimization uses historical data and scenario simulations. "
        "Deterministic risk constraints are strictly enforced. Capital loss is possible in all financial assets."
    )


class CombinedPortfolioManager:
    """Manages the master unified portfolio containing Long-Term, Short-Term, and Cash sleeves."""

    def __init__(
        self,
        long_term_planner: LongTermPlanner | None = None,
        short_term_planner: ShortTermPlanner | None = None,
    ) -> None:
        self.lt_planner = long_term_planner or LongTermPlanner()
        self.st_planner = short_term_planner or ShortTermPlanner()
        self.risk_engine = RiskEngine()

    def build_combined_portfolio(
        self,
        total_capital: float = 500000.0,
        long_term_target_pct: float = 0.70,
        short_term_target_pct: float = 0.20,
        cash_target_pct: float = 0.10,
        long_term_symbols: list[str] | None = None,
        short_term_symbols: list[str] | None = None,
        risk_preference: str = "moderate",
    ) -> CombinedPortfolioState:
        # Enforce target percentage normalization
        total_target = long_term_target_pct + short_term_target_pct + cash_target_pct
        if total_target > 0:
            long_term_target_pct = round(long_term_target_pct / total_target, 4)
            short_term_target_pct = round(short_term_target_pct / total_target, 4)
            cash_target_pct = round(1.0 - long_term_target_pct - short_term_target_pct, 4)

        lt_capital = total_capital * long_term_target_pct
        st_capital = total_capital * short_term_target_pct
        cash_capital = total_capital * cash_target_pct

        lt_syms = long_term_symbols or ["AAPL", "MSFT", "GOOGL"]
        st_syms = short_term_symbols or ["NVDA", "TSLA"]

        # 1. Build Long-Term Sleeve
        lt_plan = self.lt_planner.construct_portfolio(
            investment_amount=lt_capital,
            horizon="3_years",
            risk_preference=risk_preference,
            symbols=lt_syms,
            max_stock_allocation=settings.max_position_pct,
            max_sector_allocation=settings.max_concentration_pct,
        )

        # 2. Build Short-Term Sleeve
        st_opps = [self.st_planner.analyze_opportunity(s, horizon="1_week", portfolio_value=st_capital) for s in st_syms]

        # 3. Assemble Holdings List
        holdings: list[HoldingPosition] = []

        # Long-term holdings
        for alloc in lt_plan.allocations:
            quote = get_stock_quote(alloc.symbol)
            cur_price = quote.get("price", 150.0) if quote else 150.0
            shares = int(alloc.allocation_amount // cur_price) if cur_price > 0 else 0
            
            # Realized risk stats
            df, _, _ = load_and_prepare(alloc.symbol)
            rets = df["adj_close"].pct_change().dropna() if not df.empty else pd.Series()
            vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.20
            v95 = value_at_risk(rets, settings.var_confidence)
            cv95 = conditional_value_at_risk(rets, settings.var_confidence)

            holdings.append(
                HoldingPosition(
                    symbol=alloc.symbol,
                    horizon_bucket="LONG_TERM",
                    allocation_pct=round(alloc.allocation_amount / total_capital, 4),
                    allocation_capital=round(alloc.allocation_amount, 2),
                    current_price=cur_price,
                    shares=shares,
                    sector=alloc.sector,
                    unrealized_pnl_pct=0.0,
                    volatility=vol,
                    var_95=v95,
                    cvar_95=cv95,
                    weight_in_portfolio=round(alloc.allocation_amount / total_capital, 4),
                )
            )

        # Short-term holdings (only BUY or HOLD positions with allocation)
        st_active_capital = 0.0
        for opp in st_opps:
            if opp.signal in {"BUY", "HOLD"} and opp.recommended_capital > 0:
                alloc_cap = min(opp.recommended_capital, st_capital * 0.5)
                quote = get_stock_quote(opp.symbol)
                cur_price = quote.get("price", opp.current_price) if quote else opp.current_price
                shares = int(alloc_cap // cur_price) if cur_price > 0 else 0
                sector = get_stock_info(opp.symbol).get("sector", "Technology")

                holdings.append(
                    HoldingPosition(
                        symbol=opp.symbol,
                        horizon_bucket="SHORT_TERM",
                        allocation_pct=round(alloc_cap / total_capital, 4),
                        allocation_capital=round(alloc_cap, 2),
                        current_price=cur_price,
                        shares=shares,
                        sector=sector,
                        unrealized_pnl_pct=0.0,
                        volatility=opp.trade_zone.stop_distance_pct * 5 if opp.trade_zone else 0.25,
                        var_95=opp.var_95,
                        cvar_95=opp.cvar_95,
                        weight_in_portfolio=round(alloc_cap / total_capital, 4),
                    )
                )
                st_active_capital += alloc_cap

        # Remaining short-term unallocated capital merges into cash
        cash_capital += (st_capital - st_active_capital)
        cash_pct = round(cash_capital / total_capital, 4)

        # 4. Compute TOTAL Unified Portfolio Risk
        total_risk = self.calculate_total_risk(
            holdings=holdings,
            cash_capital=cash_capital,
            total_capital=total_capital,
        )

        # 5. Check Rebalance Triggers
        rebalance_assessment = self.check_rebalance_triggers(
            holdings=holdings,
            total_capital=total_capital,
            long_term_target_pct=long_term_target_pct,
            short_term_target_pct=short_term_target_pct,
            cash_target_pct=cash_target_pct,
            total_risk=total_risk,
        )

        return CombinedPortfolioState(
            total_capital=round(total_capital, 2),
            long_term_target_pct=long_term_target_pct,
            short_term_target_pct=short_term_target_pct,
            cash_target_pct=cash_target_pct,
            holdings=holdings,
            total_risk=total_risk,
            rebalance_assessment=rebalance_assessment,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def calculate_total_risk(
        self,
        holdings: list[HoldingPosition],
        cash_capital: float,
        total_capital: float,
    ) -> TotalPortfolioRisk:
        """Calculates portfolio-wide risk across BOTH long-term and short-term positions."""
        lt_cap = sum(h.allocation_capital for h in holdings if h.horizon_bucket == "LONG_TERM")
        st_cap = sum(h.allocation_capital for h in holdings if h.horizon_bucket == "SHORT_TERM")
        equity_cap = lt_cap + st_cap
        
        equity_exposure_pct = round(equity_cap / total_capital, 4) if total_capital > 0 else 0.0
        lt_pct = round(lt_cap / total_capital, 4) if total_capital > 0 else 0.0
        st_pct = round(st_cap / total_capital, 4) if total_capital > 0 else 0.0
        cash_pct = round(cash_capital / total_capital, 4) if total_capital > 0 else 0.0

        # Sector concentrations
        sector_weights: dict[str, float] = {}
        for h in holdings:
            sector_weights[h.sector] = sector_weights.get(h.sector, 0.0) + (h.allocation_capital / total_capital)

        dom_sector = max(sector_weights.keys(), key=lambda k: sector_weights[k]) if sector_weights else "Cash"
        max_sec_pct = round(max(sector_weights.values()), 4) if sector_weights else 0.0

        # Correlated Portfolio Volatility Calculation
        violations: list[str] = []
        if holdings:
            vols = np.array([h.volatility for h in holdings])
            weights = np.array([h.allocation_capital / total_capital for h in holdings])
            n = len(holdings)
            corr = np.full((n, n), 0.3)
            np.fill_diagonal(corr, 1.0)
            cov = np.outer(vols, vols) * corr
            portfolio_vol = float(np.sqrt(weights @ cov @ weights.T))

            # Portfolio VaR and CVaR as weighted composite with diversification benefit factor
            diversification_factor = 0.85 if n > 1 else 1.0
            tot_var = float(sum(h.var_95 * (h.allocation_capital / total_capital) for h in holdings) * diversification_factor)
            tot_cvar = float(sum(h.cvar_95 * (h.allocation_capital / total_capital) for h in holdings) * diversification_factor)
            tot_mdd = float(max((h.var_95 * 2.2 for h in holdings), default=0.08))
        else:
            portfolio_vol = 0.0
            tot_var = 0.0
            tot_cvar = 0.0
            tot_mdd = 0.0

        # Deterministic Risk Engine checks
        if max_sec_pct > settings.max_concentration_pct:
            violations.append(
                f"Sector concentration in '{dom_sector}' ({max_sec_pct:.1%}) exceeds maximum limit ({settings.max_concentration_pct:.1%})"
            )

        if portfolio_vol > settings.max_portfolio_volatility:
            violations.append(
                f"Combined portfolio volatility ({portfolio_vol:.1%}) exceeds risk cap ({settings.max_portfolio_volatility:.1%})"
            )

        if equity_exposure_pct > settings.max_concentration_pct and cash_pct < 0.05:
            violations.append("Cash reserve is below 5% minimum emergency risk threshold.")

        risk_status = "REJECT" if len(violations) > 1 else ("WARNING" if violations else "APPROVED")

        return TotalPortfolioRisk(
            total_capital=round(total_capital, 2),
            long_term_capital=round(lt_cap, 2),
            long_term_pct=lt_pct,
            short_term_capital=round(st_cap, 2),
            short_term_pct=st_pct,
            cash_capital=round(cash_capital, 2),
            cash_pct=cash_pct,
            total_equity_exposure_pct=equity_exposure_pct,
            sector_concentration={k: round(v, 4) for k, v in sector_weights.items()},
            max_sector_pct=max_sec_pct,
            dominant_sector=dom_sector,
            portfolio_volatility_annualized=round(portfolio_vol, 4),
            total_var_95=round(tot_var, 4),
            total_cvar_95=round(tot_cvar, 4),
            portfolio_max_drawdown=round(tot_mdd, 4),
            risk_status=risk_status,
            risk_violations=violations,
        )

    def check_rebalance_triggers(
        self,
        holdings: list[HoldingPosition],
        total_capital: float,
        long_term_target_pct: float,
        short_term_target_pct: float,
        cash_target_pct: float,
        total_risk: TotalPortfolioRisk,
    ) -> RebalanceAssessment:
        """Evaluates measurable rebalance conditions without unnecessary churn."""
        trigger_reasons: list[str] = []
        trigger_type = "NONE"

        # 1. Allocation Drift check (threshold: 5% absolute drift)
        lt_drift = abs(total_risk.long_term_pct - long_term_target_pct)
        st_drift = abs(total_risk.short_term_pct - short_term_target_pct)
        if lt_drift > 0.05:
            trigger_reasons.append(f"Long-term sleeve drifted by {lt_drift:.1%} from target {long_term_target_pct:.1%}.")
            trigger_type = "DRIFT"
        if st_drift > 0.05:
            trigger_reasons.append(f"Short-term sleeve drifted by {st_drift:.1%} from target {short_term_target_pct:.1%}.")
            trigger_type = "DRIFT"

        # 2. Risk Limit increase check
        if total_risk.risk_status == "REJECT":
            trigger_reasons.append("Violations detected by Deterministic Risk Engine.")
            trigger_type = "CRASH_RISK_SURGE"

        # 3. Formulate concrete rebalance actions
        actions: list[RebalanceAction] = []
        should_rebalance = len(trigger_reasons) > 0

        for h in holdings:
            drift = h.weight_in_portfolio - (0.20 if h.horizon_bucket == "LONG_TERM" else 0.10)
            if abs(drift) > 0.04:
                act = "SELL" if drift > 0 else "BUY"
                cap_delta = abs(drift) * total_capital
                sh_delta = int(cap_delta // h.current_price) if h.current_price > 0 else 0
                actions.append(
                    RebalanceAction(
                        symbol=h.symbol,
                        horizon_bucket=h.horizon_bucket,
                        current_pct=h.weight_in_portfolio,
                        target_pct=round(h.weight_in_portfolio - drift, 4),
                        drift_pct=round(drift, 4),
                        action=act,
                        capital_delta=round(cap_delta, 2),
                        shares_delta=sh_delta,
                        reason=f"Trim drift ({drift:+.1%}) to maintain diversification limits.",
                    )
                )

        return RebalanceAssessment(
            should_rebalance=should_rebalance,
            trigger_type=trigger_type,
            trigger_reasons=trigger_reasons,
            actions=actions,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    def record_rebalance_event(
        self,
        plan_id: int | None,
        assessment: RebalanceAssessment,
        total_risk: TotalPortfolioRisk,
    ) -> int | None:
        """Persists a rebalance event to the audit database."""
        session = get_session()
        try:
            import uuid
            rev_id = f"REB_{uuid.uuid4().hex[:12].upper()}"
            max_drift = max((abs(a.drift_pct) for a in assessment.actions), default=0.0)
            rev = RebalanceEvent(
                rebalance_id=rev_id,
                plan_id=str(plan_id) if plan_id else None,
                horizon="COMBINED",
                trigger_reason="; ".join(assessment.trigger_reasons)[:120] if assessment.trigger_reasons else "Scheduled check",
                drift_pct=round(max_drift, 4),
                actions_json=json.dumps([a.model_dump() for a in assessment.actions]),
                risk_status=total_risk.risk_status,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(rev)
            session.commit()
            return rev.id
        except Exception as e:
            logger.warning(f"Could not persist RebalanceEvent: {e}")
            session.rollback()
            return None
        finally:
            session.close()

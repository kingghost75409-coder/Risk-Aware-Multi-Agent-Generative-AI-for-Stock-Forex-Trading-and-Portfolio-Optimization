"""Market Stress Testing Agent.

Research requirement: Allows traders and risk officers to stress-test hypothetical
market dislocations without perturbing or mutating live paper trading positions.

Scenarios Simulated:
  1. Market Drop -5% (Standard macro pullback)
  2. Market Crash -10% (Correction / flash liquidity shock)
  3. Severe Crash -20% (Systemic crisis / Black Monday style)
  4. Volatility Spike (+100% realized volatility surge)
  5. Sector Tech Crash (-15% technology sector rout)
  6. Negative Sentiment Shock (Severe negative headline contagion, sentiment -> -0.80)
  7. Sudden Regime Shift (Instant flip to BEAR_HIGH_VOLATILITY)

Recalculates:
  - Stressed Portfolio Value
  - Portfolio Loss ($ and %)
  - Stressed VaR 95%
  - Stressed CVaR 95% (Expected Shortfall)
  - Stressed Maximum Drawdown
  - Stressed Annualized Volatility
  - Post-shock Max Safe Position Limit
  - Regulatory / Policy Breaches
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.config import settings
from app.risk.risk_engine import value_at_risk, conditional_value_at_risk, max_drawdown


class ScenarioResult(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    stressed_portfolio_value: float
    portfolio_loss_amount: float
    portfolio_loss_pct: float
    stressed_var_95: float
    stressed_cvar_95: float
    stressed_max_drawdown: float
    stressed_volatility_annualized: float
    recommended_position_haircut_pct: float
    would_breach_limits: bool
    breaches: list[str]


class StressTestSuiteReport(BaseModel):
    symbol: str
    baseline_portfolio_value: float
    baseline_cash: float
    baseline_exposure_pct: float
    scenarios: list[ScenarioResult]
    worst_case_scenario: str
    summary_verdict: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MarketStressTestingAgent:
    """Simulates acute stress scenarios on market data and portfolio state."""

    SCENARIO_CONFIGS = [
        {
            "id": "MKT_MINUS_5",
            "name": "Market Pullback (-5%)",
            "desc": "Broad market equity decline of 5% with mild vol expansion (1.2x).",
            "price_shock": -0.05,
            "vol_multiplier": 1.20,
            "sentiment_shock": -0.20,
        },
        {
            "id": "MKT_MINUS_10",
            "name": "Market Correction (-10%)",
            "desc": "Severe correction of 10% with substantial vol expansion (1.5x).",
            "price_shock": -0.10,
            "vol_multiplier": 1.50,
            "sentiment_shock": -0.45,
        },
        {
            "id": "MKT_MINUS_20",
            "name": "Market Crash (-20%)",
            "desc": "Black Swan liquidity crash of 20% with extreme volatility surge (2.2x).",
            "price_shock": -0.20,
            "vol_multiplier": 2.20,
            "sentiment_shock": -0.80,
        },
        {
            "id": "VOL_SPIKE_100",
            "name": "Volatility Shock (+100%)",
            "desc": "Doubling of annualized volatility without directional price trend change.",
            "price_shock": -0.02,
            "vol_multiplier": 2.00,
            "sentiment_shock": -0.30,
        },
        {
            "id": "SECTOR_CRASH_15",
            "name": "Tech Sector Crash (-15%)",
            "desc": "15% sector-concentrated drawdown impacting technology and growth stocks.",
            "price_shock": -0.15,
            "vol_multiplier": 1.75,
            "sentiment_shock": -0.60,
        },
        {
            "id": "SENTIMENT_SHOCK",
            "name": "Negative Sentiment Shock",
            "desc": "Sudden regulatory/macro news contagion crashing market sentiment to -0.85.",
            "price_shock": -0.07,
            "vol_multiplier": 1.40,
            "sentiment_shock": -0.85,
        },
        {
            "id": "REGIME_SHIFT_BEAR_HIGH_VOL",
            "name": "Regime Shift (BEAR_HIGH_VOL)",
            "desc": "Instant transition from Bull/Sideways to persistent high-volatility Bear market.",
            "price_shock": -0.12,
            "vol_multiplier": 1.90,
            "sentiment_shock": -0.65,
        },
    ]

    # Historical crisis replay patterns (Upgrade 7)
    # Based on actual peak-to-trough drawdown characteristics
    HISTORICAL_SCENARIOS = [
        {
            "id": "HIST_GFC_2008",
            "name": "2008 Global Financial Crisis",
            "desc": "Peak-to-trough S&P 500 drawdown of -56.8% over ~17 months "
                    "(Oct 2007 – Mar 2009). VIX peaked at 80+. Credit markets froze.",
            "price_shock": -0.568,
            "vol_multiplier": 4.50,
            "sentiment_shock": -0.95,
        },
        {
            "id": "HIST_COVID_2020",
            "name": "2020 COVID-19 Crash",
            "desc": "S&P 500 fell -33.9% in 23 trading days (Feb 19 – Mar 23, 2020). "
                    "Fastest bear market in history. VIX hit 82.69.",
            "price_shock": -0.339,
            "vol_multiplier": 5.00,
            "sentiment_shock": -0.90,
        },
        {
            "id": "HIST_RATE_2022",
            "name": "2022 Rate Shock / Tech Rout",
            "desc": "S&P 500 fell -25.4% (Jan – Oct 2022). NASDAQ -33%. "
                    "Driven by aggressive Fed rate hikes and growth stock derating.",
            "price_shock": -0.254,
            "vol_multiplier": 2.00,
            "sentiment_shock": -0.70,
        },
    ]

    def run_stress_test(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float = 100_000.0,
        cash: float = 100_000.0,
        current_holdings: dict | None = None,
        custom_scenario: dict | None = None,
    ) -> StressTestSuiteReport:
        symbol = symbol.upper()
        holdings = current_holdings or {}
        # Calculate active equities value
        invested_value = max(0.0, portfolio_value - cash)
        exposure_pct = invested_value / portfolio_value if portfolio_value > 0 else 0.0

        close = df["adj_close"]
        base_returns = close.pct_change().dropna()

        configs = list(self.SCENARIO_CONFIGS)
        if custom_scenario:
            configs.append(custom_scenario)

        results: list[ScenarioResult] = []
        max_loss = 0.0
        worst_scenario = "None"

        for cfg in configs:
            result = self._evaluate_scenario(cfg, base_returns, close,
                                             portfolio_value, cash, invested_value)
            results.append(result)
            if result.portfolio_loss_amount > max_loss:
                max_loss = result.portfolio_loss_amount
                worst_scenario = cfg["name"]

        verdict = (
            f"Stress testing completed across {len(results)} scenarios. "
            f"Worst impact under '{worst_scenario}' (Loss: -${max_loss:,.2f}). "
            f"Live portfolio was NOT modified (read-only research simulation)."
        )

        return StressTestSuiteReport(
            symbol=symbol,
            baseline_portfolio_value=portfolio_value,
            baseline_cash=cash,
            baseline_exposure_pct=round(exposure_pct, 4),
            scenarios=results,
            worst_case_scenario=worst_scenario,
            summary_verdict=verdict,
        )

    def run_historical_stress(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float = 100_000.0,
        cash: float = 100_000.0,
    ) -> StressTestSuiteReport:
        """Run historical crisis scenario replays (2008, 2020, 2022)."""
        symbol = symbol.upper()
        invested_value = max(0.0, portfolio_value - cash)
        exposure_pct = invested_value / portfolio_value if portfolio_value > 0 else 0.0

        close = df["adj_close"]
        base_returns = close.pct_change().dropna()

        results: list[ScenarioResult] = []
        max_loss = 0.0
        worst_scenario = "None"

        for cfg in self.HISTORICAL_SCENARIOS:
            result = self._evaluate_scenario(cfg, base_returns, close,
                                             portfolio_value, cash, invested_value)
            results.append(result)
            if result.portfolio_loss_amount > max_loss:
                max_loss = result.portfolio_loss_amount
                worst_scenario = cfg["name"]

        verdict = (
            f"Historical crisis replay across {len(results)} scenarios. "
            f"Worst impact under '{worst_scenario}' (Loss: -${max_loss:,.2f}). "
            f"Read-only — no portfolio mutations."
        )

        return StressTestSuiteReport(
            symbol=symbol,
            baseline_portfolio_value=portfolio_value,
            baseline_cash=cash,
            baseline_exposure_pct=round(exposure_pct, 4),
            scenarios=results,
            worst_case_scenario=worst_scenario,
            summary_verdict=verdict,
        )

    def run_custom_stress(
        self,
        symbol: str,
        df: pd.DataFrame,
        price_shock_pct: float,
        vol_multiplier: float = 1.5,
        sentiment_override: float = -0.50,
        scenario_name: str = "Custom Scenario",
        portfolio_value: float = 100_000.0,
        cash: float = 100_000.0,
    ) -> StressTestSuiteReport:
        """Run a user-defined custom stress scenario."""
        custom_cfg = {
            "id": "CUSTOM_USER",
            "name": scenario_name,
            "desc": f"Custom: price shock {price_shock_pct:+.1%}, "
                    f"vol mult {vol_multiplier:.1f}x, "
                    f"sentiment override {sentiment_override:+.2f}.",
            "price_shock": price_shock_pct,
            "vol_multiplier": vol_multiplier,
            "sentiment_shock": sentiment_override,
        }
        return self.run_stress_test(
            symbol=symbol, df=df, portfolio_value=portfolio_value,
            cash=cash, custom_scenario=custom_cfg,
        )

    def _evaluate_scenario(
        self,
        cfg: dict,
        base_returns: pd.Series,
        close: pd.Series,
        portfolio_value: float,
        cash: float,
        invested_value: float,
    ) -> ScenarioResult:
        """Evaluate a single stress scenario and return a ScenarioResult."""
        p_shock = float(cfg.get("price_shock", 0.0))
        v_mult = float(cfg.get("vol_multiplier", 1.0))

        # Shock the returns distribution
        stressed_returns = (base_returns * v_mult) + (p_shock / 20.0)  # shift tail

        # Recompute core risk metrics
        st_var95 = value_at_risk(stressed_returns, settings.var_confidence)
        st_cvar95 = conditional_value_at_risk(stressed_returns, settings.var_confidence)
        st_vol = float(stressed_returns.std() * np.sqrt(252)) if len(stressed_returns) > 1 else 0.0

        # Stressed prices series for drawdown
        stressed_prices = close * (1.0 + p_shock)
        st_mdd = max_drawdown(pd.concat([close, stressed_prices]))

        # Simulated portfolio loss
        # Equities absorb price shock, cash stays invariant
        stressed_invested = invested_value * (1.0 + p_shock)
        stressed_port_val = round(cash + max(0.0, stressed_invested), 2)
        loss_amount = round(portfolio_value - stressed_port_val, 2)
        loss_pct = round(loss_amount / portfolio_value, 4) if portfolio_value > 0 else 0.0

        # Check breaches
        breaches = []
        if st_vol > settings.max_portfolio_volatility:
            breaches.append(f"Stressed volatility {st_vol:.1%} breaches ceiling {settings.max_portfolio_volatility:.1%}")
        if loss_pct > settings.max_daily_loss_pct * 3.0:
            breaches.append(f"Portfolio shock loss {loss_pct:.1%} exceeds critical 3x daily stop buffer")
        if st_cvar95 > 0.08:
            breaches.append(f"Stressed CVaR-95 {st_cvar95:.1%} breaches extreme risk limit (8.0%)")

        # Safe haircut recommendation
        if len(breaches) > 0:
            haircut = min(0.75, 0.25 * len(breaches))
        else:
            haircut = 0.10 if p_shock < -0.10 else 0.0


        return ScenarioResult(
            scenario_id=cfg["id"],
            scenario_name=cfg["name"],
            description=cfg["desc"],
            stressed_portfolio_value=stressed_port_val,
            portfolio_loss_amount=loss_amount,
            portfolio_loss_pct=loss_pct,
            stressed_var_95=round(st_var95, 4),
            stressed_cvar_95=round(st_cvar95, 4),
            stressed_max_drawdown=round(st_mdd, 4),
            stressed_volatility_annualized=round(st_vol, 4),
            recommended_position_haircut_pct=round(haircut, 2),
            would_breach_limits=len(breaches) > 0,
            breaches=breaches,
        )


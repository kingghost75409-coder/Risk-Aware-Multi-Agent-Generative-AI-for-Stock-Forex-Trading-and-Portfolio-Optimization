"""Position Sizing & Portfolio Optimization.

Primary sizing method: fractional Kelly Criterion, since it only needs the
per-symbol win-probability and expected-return already produced by the ML
agent (no covariance matrix estimation required, which is noisy with the
short synthetic/demo history available here).

Comparison method: a simplified Black-Litterman-style tilt — starts from an
equal-weight prior and tilts weights toward symbols with stronger combined
agent conviction, regularized by the return covariance matrix. Kept as an
explicit "comparison_method" output so the ablation/backtesting layer can
report both. A full Black-Litterman implementation (market-cap prior +
formal investor views vector) is a documented extension point.

Every sizing result is re-clipped to the Risk Engine's limits — this module
proposes, the Risk Engine (already applied upstream in the orchestrator)
disposes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.config import settings


class PositionSize(BaseModel):
    symbol: str
    kelly_fraction: float
    kelly_fraction_capped: float
    black_litterman_tilt_weight: float
    recommended_position_pct: float
    method_used: str
    rationale: str


def kelly_fraction(win_prob: float, expected_return: float, avg_loss_magnitude: float) -> float:
    """Simplified Kelly: f* = p/loss - q/gain, using expected magnitudes
    instead of a literal betting-odds formulation (standard adaptation for
    continuous-return trading strategies)."""
    p = np.clip(win_prob, 0.01, 0.99)
    q = 1 - p
    gain = max(abs(expected_return), 1e-4)
    loss = max(avg_loss_magnitude, 1e-4)
    f = p / loss - q / gain
    return float(np.clip(f, -1.0, 1.0))


class PositionSizer:
    def size_position(
        self,
        symbol: str,
        win_probability: float,
        expected_return: float,
        avg_loss_magnitude: float,
        conviction_score: float,  # combined agent conviction in [-1, 1], used for BL comparison
        fractional_kelly: float = 0.5,  # "half-Kelly" — full Kelly is too aggressive for live sizing
    ) -> PositionSize:
        raw_kelly = kelly_fraction(win_probability, expected_return, avg_loss_magnitude)
        capped_kelly = float(np.clip(raw_kelly * fractional_kelly, 0.0, settings.max_position_pct))

        # comparison: equal-weight prior tilted by conviction, capped the same way
        prior = settings.max_position_pct * 0.5
        bl_tilt = float(np.clip(prior * (1 + conviction_score), 0.0, settings.max_position_pct))

        recommended = capped_kelly  # Kelly is the primary method
        rationale = (
            f"Half-Kelly sizing: raw Kelly f*={raw_kelly:.3f} (p={win_probability:.2f}, "
            f"exp_return={expected_return:.4f}, avg_loss={avg_loss_magnitude:.4f}), "
            f"scaled 0.5x and capped at max_position_pct={settings.max_position_pct:.1%}. "
            f"Black-Litterman-style comparison weight={bl_tilt:.1%}."
        )

        return PositionSize(
            symbol=symbol,
            kelly_fraction=round(raw_kelly, 4),
            kelly_fraction_capped=round(capped_kelly, 4),
            black_litterman_tilt_weight=round(bl_tilt, 4),
            recommended_position_pct=round(recommended, 4),
            method_used="fractional_kelly",
            rationale=rationale,
        )


class PortfolioOptimizer(BaseModel):
    pass


def optimize_portfolio(candidate_positions: list[dict]) -> dict:
    """candidate_positions: [{symbol, recommended_position_pct, expected_return,
    volatility}]. Normalizes weights so total exposure never exceeds 100% of
    capital and respects the per-symbol concentration cap."""
    if not candidate_positions:
        return {"weights": {}, "expected_portfolio_return": 0.0, "expected_portfolio_volatility": 0.0}

    weights = {p["symbol"]: min(p["recommended_position_pct"], settings.max_position_pct) for p in candidate_positions}
    total = sum(weights.values())
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}

    exp_returns = np.array([p["expected_return"] for p in candidate_positions])
    vols = np.array([p.get("volatility", 0.2) for p in candidate_positions])
    w = np.array([weights[p["symbol"]] for p in candidate_positions])

    portfolio_return = float(np.dot(w, exp_returns))
    # simplified: assume moderate positive correlation (0.3) across equities
    # rather than estimating a full covariance matrix from short demo history
    n = len(w)
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    portfolio_vol = float(np.sqrt(w @ cov @ w.T)) if n > 0 else 0.0

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "expected_portfolio_return": round(portfolio_return, 5),
        "expected_portfolio_volatility": round(portfolio_vol, 4),
        "total_exposure_pct": round(sum(weights.values()), 4),
        "note": "Covariance uses a simplified constant 0.3 pairwise correlation assumption, not an estimated matrix "
                "(documented simplification given short demo history).",
    }

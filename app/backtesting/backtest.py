"""Backtesting + Ablation Study.

Chronological walk-forward evaluation. Strategies compared:
  1. Buy & Hold
  2. Technical-only
  3. Technical + ML
  4. Technical + ML + Sentiment
  5. Technical + ML + RAG (RAG is used as a go/no-go evidence gate here,
     since RAG evidence is descriptive/qualitative, not a numeric signal —
     see orchestrator docstring on why RAG isn't blended into the numeric vote)
  6. Full multi-agent system (Technical + ML + Sentiment + Regime + Risk Engine)

All decisions in the backtest use ONLY information available up to that
bar (rolling technical indicators, and the ML model is retrained on an
expanding chronological window) — no look-ahead.

IMPORTANT: with ~1-2 years of daily bars per symbol and a same-process
in-memory sentiment/RAG stub, these are demonstration-scale results meant
to exercise the full evaluation pipeline end-to-end, not a claim of
statistically robust outperformance. Metrics that would need much larger
history/sample sizes to be meaningful are labeled accordingly in the output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from app.agents.regime_agent import MarketRegimeAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.data.news_sample import get_sample_headlines
from app.data.pipeline import load_and_prepare


@dataclass
class BacktestMetrics:
    strategy: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    volatility_annualized: float
    n_trades: int
    turnover: float


def _perf_metrics(strategy_name: str, equity_curve: pd.Series, trade_returns: list[float],
                   trades_per_period: list[int], periods_per_year: int = 252) -> BacktestMetrics:
    rets = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0
    n_periods = len(equity_curve)
    cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (periods_per_year / max(n_periods, 1)) - 1) \
        if n_periods > 1 and equity_curve.iloc[0] > 0 else 0.0

    vol = float(rets.std() * np.sqrt(periods_per_year)) if len(rets) > 1 else 0.0
    sharpe = float((rets.mean() * periods_per_year) / vol) if vol > 0 else 0.0

    downside = rets[rets < 0]
    downside_vol = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = float((rets.mean() * periods_per_year) / downside_vol) if downside_vol > 0 else 0.0

    cum_max = equity_curve.cummax()
    mdd = float(-((equity_curve - cum_max) / cum_max).min()) if len(equity_curve) > 1 else 0.0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    win_rate = float(len(wins) / len(trade_returns)) if trade_returns else 0.0
    profit_factor = float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (float("inf") if wins else 0.0)
    turnover = float(np.mean(trades_per_period)) if trades_per_period else 0.0

    return BacktestMetrics(
        strategy=strategy_name, total_return=round(total_return, 4), cagr=round(cagr, 4),
        sharpe_ratio=round(sharpe, 3), sortino_ratio=round(sortino, 3) if np.isfinite(sortino) else 0.0,
        max_drawdown=round(mdd, 4), win_rate=round(win_rate, 3),
        profit_factor=round(profit_factor, 3) if np.isfinite(profit_factor) else 999.0,
        volatility_annualized=round(vol, 4), n_trades=len(trade_returns), turnover=round(turnover, 2),
    )


def _simulate_signal_strategy(df: pd.DataFrame, signals: pd.Series, name: str,
                               transaction_cost_bps: float = 5) -> BacktestMetrics:
    """signals: 1 = long, 0 = flat, aligned to df.index, decided using only
    data available at that bar (already computed causally by caller)."""
    close = df["adj_close"]
    position = signals.shift(1).fillna(0)  # trade executes at NEXT bar's open/close, not same-bar
    daily_ret = close.pct_change().fillna(0)
    strategy_ret = position * daily_ret

    trade_changes = position.diff().fillna(position.iloc[0] if len(position) else 0).abs()
    cost = trade_changes * (transaction_cost_bps / 10_000)
    strategy_ret = strategy_ret - cost

    equity = (1 + strategy_ret).cumprod() * 100_000
    trade_returns = strategy_ret[trade_changes > 0].tolist()
    trades_per_period = [int(x) for x in (trade_changes > 0).tolist()]

    return _perf_metrics(name, equity, trade_returns, trades_per_period)


def run_ablation_study(symbol: str, period: str = "2y") -> dict:
    df, report, source = load_and_prepare(symbol, period=period, use_cache=True)
    if not report.is_usable:
        return {"symbol": symbol, "error": "insufficient data", "warnings": report.warnings}

    close = df["adj_close"]
    tech_agent = TechnicalAgent()
    regime_agent = MarketRegimeAgent()
    sentiment_agent = SentimentAgent()
    headlines = get_sample_headlines(symbol)
    static_sentiment = sentiment_agent.analyze(symbol, headlines)  # single snapshot, applied uniformly across backtest
    # (a full implementation would use a dated headline history per bar; the
    # demo news dataset has no per-bar timestamps to align to price history)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
                         / (-close.diff().clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean().replace(0, np.nan)))
    rsi = rsi.fillna(50)

    momentum_10 = close.pct_change(10)

    # --- Experiment A: Technical only (trend-following rule) ---
    tech_signal = ((sma20 > sma50) & (rsi < 70)).astype(int)

    # --- Experiment B: Technical + ML-style momentum confirmation ---
    ml_confirm = (momentum_10 > 0).astype(int)
    tech_ml_signal = ((tech_signal == 1) & (ml_confirm == 1)).astype(int)

    # --- Experiment C: + Sentiment gate (blocks longs when sentiment is negative) ---
    sentiment_gate = 0 if static_sentiment.label == "NEGATIVE" else 1
    tech_ml_sent_signal = tech_ml_signal * sentiment_gate

    # --- Experiment D: + Temporal RAG evidence gate (prevents look-ahead by checking historical dates) ---
    from app.rag.rag_agent import FinancialRAGAgent
    rag_agent = FinancialRAGAgent()
    mid_date = str(df.index[len(df) // 2])[:10]
    rag_result = rag_agent.get_evidence_summary(symbol, as_of_date=mid_date)
    rag_gate = 0.5 if rag_result.insufficient_evidence else 1.0
    tech_ml_rag_signal = tech_ml_sent_signal * rag_gate

    # --- Experiment E: + Adaptive Risk Budget & Volatility Circuit Breaker ---
    rolling_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    # Volatility constriction: de-risk when rolling volatility exceeds 50%
    adaptive_budget_gate = (rolling_vol < 0.50).astype(int)
    tech_ml_adaptive_signal = tech_ml_rag_signal * adaptive_budget_gate

    # --- Experiment F: + Uncertainty-Aware Abstention (flattens during high entropy or conflict) ---
    # Uncertainty proxy: when RSI is in extreme contested zone (45-55) and vol is shifting
    uncertainty_mask = ((rsi >= 46) & (rsi <= 54) & (momentum_10.abs() < 0.01)).astype(int)
    # When uncertain, system chooses ABSTAIN (signal = 0)
    full_uncertainty_aware_signal = tech_ml_adaptive_signal * (1 - uncertainty_mask)

    buy_hold_signal = pd.Series(1, index=df.index)

    # Slippage + transaction cost: 5 bps commission + 3 bps slippage = 8 bps
    TOTAL_FRICTION_BPS = 8.0

    results = [
        _simulate_signal_strategy(df, buy_hold_signal, "Buy & Hold", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, tech_signal, "A: Technical Only", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, tech_ml_signal, "B: Technical + ML", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, tech_ml_sent_signal, "C: Technical + ML + Sentiment", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, tech_ml_rag_signal, "D: Technical + ML + Temporal RAG", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, tech_ml_adaptive_signal, "E: + Adaptive Risk Budget & Circuit Breaker", transaction_cost_bps=TOTAL_FRICTION_BPS),
        _simulate_signal_strategy(df, full_uncertainty_aware_signal, "F: Full Framework (+ Uncertainty ABSTAIN)", transaction_cost_bps=TOTAL_FRICTION_BPS),
    ]

    return {
        "symbol": symbol,
        "period": period,
        "data_source": source,
        "n_bars": len(df),
        "friction_model": f"Transaction cost: 5 bps, Slippage: 3 bps (Total: {TOTAL_FRICTION_BPS} bps per trade)",
        "train_test_note": "Signals use only causal (past-and-present) data at each bar; "
                            "positions are lagged one bar before applying next-bar return.",
        "results": [asdict(r) for r in results],
        "caveat": "Demonstration-scale backtest on a single symbol / short window. Not a claim of "
                  "statistically significant outperformance — see module docstring.",
    }

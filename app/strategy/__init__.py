"""Multi-Horizon Investment Strategy and Portfolio Planning Module.

Additive feature package integrating:
- HorizonStrategyRouter (routing, horizon-aware confidence, abstention)
- LongTermPlanner (fundamental scoring, 10-K RAG citations, SIP simulation, portfolio allocation)
- ShortTermPlanner (tactical trade zones, stop-loss, risk/reward, ML + momentum, uncertainty)
- CombinedPortfolioManager (unified long/short/cash sleeves, total portfolio risk, rebalancing)
- MultiHorizonBacktester (Long-Term vs Short-Term vs Buy & Hold evaluation)
"""
from app.strategy.router import (
    HorizonType,
    HorizonStrategyRouter,
    HorizonRoutingResult,
    LongTermConfidence,
    ShortTermConfidence,
    parse_horizon,
)
from app.strategy.long_term_planner import (
    LongTermPlanner,
    LongTermStockAnalysis,
    FundamentalScores,
    LongTermPlan,
    AssetAllocation,
    SIPSimulationResult,
)
from app.strategy.short_term_planner import (
    ShortTermPlanner,
    ShortTermOpportunity,
    TradeZone,
)
from app.strategy.combined_portfolio import (
    CombinedPortfolioManager,
    CombinedPortfolioState,
    TotalPortfolioRisk,
    HoldingPosition,
    RebalanceAssessment,
    RebalanceAction,
)
from app.strategy.horizon_backtest import (
    MultiHorizonBacktester,
    HorizonBacktestMetrics,
)

__all__ = [
    "HorizonType",
    "HorizonStrategyRouter",
    "HorizonRoutingResult",
    "LongTermConfidence",
    "ShortTermConfidence",
    "parse_horizon",
    "LongTermPlanner",
    "LongTermStockAnalysis",
    "FundamentalScores",
    "LongTermPlan",
    "AssetAllocation",
    "SIPSimulationResult",
    "ShortTermPlanner",
    "ShortTermOpportunity",
    "TradeZone",
    "CombinedPortfolioManager",
    "CombinedPortfolioState",
    "TotalPortfolioRisk",
    "HoldingPosition",
    "RebalanceAssessment",
    "RebalanceAction",
    "MultiHorizonBacktester",
    "HorizonBacktestMetrics",
]

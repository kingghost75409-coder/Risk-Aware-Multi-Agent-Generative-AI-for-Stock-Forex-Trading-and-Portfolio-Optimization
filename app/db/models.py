"""SQLAlchemy models + engine/session setup.

Uses SQLite by default (see app/config.py DATABASE_URL) — swap to a
Postgres URL for production; no other code changes needed since we use
the SQLAlchemy ORM throughout rather than raw SQLite-specific SQL.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, String, Text, Boolean,
                         create_engine, Index)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(32), unique=True, index=True)
    symbol = Column(String(16), index=True)
    side = Column(String(8))
    quantity = Column(Float)
    price = Column(Float)
    executed_price = Column(Float)
    transaction_cost = Column(Float)
    reason = Column(Text)
    risk_status = Column(String(16))
    timestamp = Column(DateTime, default=_now, index=True)


class AgentDecision(Base):
    __tablename__ = "agent_decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    final_decision = Column(String(16))  # BUY / SELL / HOLD / ABSTAIN
    confidence = Column(Float)
    position_size_pct = Column(Float)
    technical_signal = Column(String(8))
    ml_direction = Column(String(8))
    ml_probability = Column(Float)
    sentiment_label = Column(String(16))
    regime = Column(String(48))
    risk_status = Column(String(16))
    abstention_reason = Column(Text, nullable=True)
    selected_agents_json = Column(Text, nullable=True)
    signal_confidence = Column(Float, nullable=True)
    prediction_confidence = Column(Float, nullable=True)
    evidence_quality = Column(Float, nullable=True)
    market_stability = Column(Float, nullable=True)
    risk_confidence = Column(Float, nullable=True)
    critic_analysis_json = Column(Text, nullable=True)
    counterfactual_json = Column(Text, nullable=True)
    risk_budget_json = Column(Text, nullable=True)
    rag_citations_json = Column(Text, nullable=True)
    audit_trail_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class StressTestResult(Base):
    __tablename__ = "stress_test_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    scenario_name = Column(String(64))
    portfolio_impact_pct = Column(Float)
    stressed_var_95 = Column(Float)
    stressed_cvar_95 = Column(Float)
    stressed_max_drawdown = Column(Float)
    stressed_volatility = Column(Float)
    would_breach = Column(Boolean)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), unique=True, index=True)
    entity_type = Column(String(32), index=True)  # Company, Sector, Product, RiskFactor, etc.
    name = Column(String(128))
    attributes_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now)


class KnowledgeRelationship(Base):
    __tablename__ = "knowledge_relationships"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(64), index=True)
    target_id = Column(String(64), index=True)
    relation_type = Column(String(48), index=True)  # COMPETITOR_OF, SUPPLIER_TO, BELONGS_TO_SECTOR, etc.
    weight = Column(Float, default=1.0)
    metadata_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now)


class AdversarialEvalLog(Base):
    __tablename__ = "adversarial_eval_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_type = Column(String(64), index=True)
    symbol = Column(String(16), index=True)
    query = Column(Text)
    retrieval_quality_score = Column(Float)
    conflict_detected = Column(Boolean)
    unsupported_claims_count = Column(Integer)
    hallucination_prevented = Column(Boolean)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class RiskMetric(Base):
    __tablename__ = "risk_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    max_drawdown = Column(Float)
    volatility_annualized = Column(Float)
    status = Column(String(16))
    timestamp = Column(DateTime, default=_now, index=True)


class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), unique=True, index=True)
    added_at = Column(DateTime, default=_now)
    active = Column(Boolean, default=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), index=True)
    symbol = Column(String(16), index=True, nullable=True)
    message = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class InvestmentPlan(Base):
    __tablename__ = "investment_plans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(32), unique=True, index=True)
    plan_type = Column(String(32), default="LONG_TERM", index=True)  # LONG_TERM, SHORT_TERM, COMBINED
    horizon = Column(String(16), index=True)  # 6m, 1y, 3y, 5y, intraday, 1d, 1w, 1-4w
    capital = Column(Float)
    risk_profile = Column(String(32))  # Conservative, Moderate, Aggressive
    allocations_json = Column(Text)  # JSON dict of symbols to weights and share counts
    expected_return_range = Column(String(64))
    portfolio_volatility = Column(Float)
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    max_drawdown = Column(Float)
    status = Column(String(16), default="ACTIVE")
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class InvestmentGoal(Base):
    __tablename__ = "investment_goals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    goal_id = Column(String(32), unique=True, index=True)
    name = Column(String(64))
    target_amount = Column(Float)
    initial_investment = Column(Float)
    periodic_amount = Column(Float)
    frequency = Column(String(16), default="MONTHLY")  # MONTHLY, QUARTERLY, CUSTOM
    horizon_years = Column(Float)
    projection_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class RebalanceEvent(Base):
    __tablename__ = "rebalance_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    rebalance_id = Column(String(32), unique=True, index=True)
    plan_id = Column(String(32), nullable=True, index=True)
    horizon = Column(String(16))
    trigger_reason = Column(String(128))
    drift_pct = Column(Float, nullable=True)
    actions_json = Column(Text)  # JSON list of rebalancing trades
    risk_status = Column(String(16))
    timestamp = Column(DateTime, default=_now, index=True)


class HorizonAssessmentRecord(Base):
    __tablename__ = "horizon_assessments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    recommended_horizon = Column(String(32), index=True)
    long_term_score = Column(Float)
    short_term_score = Column(Float)
    confidence = Column(Float)
    details_json = Column(Text, nullable=True)
    user_override_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class DigitalTwinScenarioRecord(Base):
    __tablename__ = "digital_twin_scenarios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String(32), unique=True, index=True)
    scenario_name = Column(String(64))
    baseline_val = Column(Float)
    simulated_val = Column(Float)
    turnover_pct = Column(Float)
    transaction_cost = Column(Float)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class EvidenceConflictRecord(Base):
    __tablename__ = "evidence_conflicts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), index=True)
    category = Column(String(48), index=True)
    severity = Column(String(16), index=True)
    conflict_score = Column(Float)
    affected_horizon = Column(String(24))
    recommended_action = Column(String(24))
    claims_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class PerformanceAttributionRecord(Base):
    __tablename__ = "performance_attributions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    period = Column(String(32), index=True)
    total_return = Column(Float)
    benchmark_return = Column(Float)
    excess_return = Column(Float)
    allocation_effect = Column(Float)
    selection_effect = Column(Float)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class SystemReliabilityLog(Base):
    __tablename__ = "system_reliability_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    overall_status = Column(String(16), index=True)
    block_trading = Column(Boolean)
    position_multiplier = Column(Float)
    alerts_count = Column(Integer)
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class ForexEventRecord(Base):
    __tablename__ = "forex_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(48), unique=True, index=True)
    name = Column(String(128), index=True)
    category = Column(String(32), index=True)
    currency = Column(String(16), index=True)
    scheduled_time = Column(String(32), index=True)
    previous_value = Column(Float, nullable=True)
    consensus_expectation = Column(Float, nullable=True)
    unit = Column(String(16))
    risk_level = Column(String(16))
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class ForexPreEventAssessmentRecord(Base):
    __tablename__ = "forex_pre_event_assessments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(48), index=True)
    event_name = Column(String(128))
    currency = Column(String(16), index=True)
    affected_pairs_json = Column(Text)
    scheduled_time = Column(String(32), index=True)
    consensus_expectation = Column(Float, nullable=True)
    previous_value = Column(Float, nullable=True)
    expected_risk_level = Column(String(16))
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class ForexPostEventAnalysisRecord(Base):
    __tablename__ = "forex_post_event_analyses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(48), index=True)
    event_name = Column(String(128))
    currency = Column(String(16), index=True)
    primary_pair = Column(String(16), index=True)
    actual_value = Column(Float)
    consensus_expectation = Column(Float, nullable=True)
    raw_surprise = Column(Float)
    reaction_pattern = Column(String(24))
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class ForexRegimeLogRecord(Base):
    __tablename__ = "forex_regime_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(16), index=True)
    previous_regime = Column(String(48))
    new_regime = Column(String(48), index=True)
    reason = Column(Text)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class ForexEventMemoryRecord(Base):
    __tablename__ = "forex_event_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(48), unique=True, index=True)
    name = Column(String(128))
    category = Column(String(32), index=True)
    currency = Column(String(16), index=True)
    primary_pair = Column(String(16), index=True)
    surprise = Column(Float)
    observed_pattern = Column(String(24))
    details_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


class TradingStyleConfigRecord(Base):
    __tablename__ = "trading_style_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    style = Column(String(24), index=True)  # DAY, SWING, POSITION
    instrument_type = Column(String(16), default="ALL")  # STOCK, FOREX, ALL
    timeframe = Column(String(16))  # 5m, 15m, 1h, 4h, 1d, 1w
    max_holding_duration_days = Column(Float)
    risk_per_trade_pct = Column(Float, default=0.01)
    min_risk_reward_ratio = Column(Float, default=1.5)
    rules_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=_now, index=True)


class DailyRiskStateRecord(Base):
    __tablename__ = "daily_risk_state"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date_str = Column(String(16), unique=True, index=True)  # YYYY-MM-DD
    starting_equity = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    fees_paid = Column(Float, default=0.0)
    consecutive_losses = Column(Integer, default=0)
    is_locked = Column(Boolean, default=False)
    lock_reason = Column(String(128), nullable=True)
    locked_at = Column(DateTime, nullable=True)
    cooldown_until = Column(DateTime, nullable=True)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, index=True)


class TradeQualityGateRecord(Base):
    __tablename__ = "trade_quality_gate_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(String(48), unique=True, index=True)
    symbol = Column(String(16), index=True)
    trading_style = Column(String(24), index=True)
    proposed_side = Column(String(8))
    decision = Column(String(24), index=True)  # ELIGIBLE, REJECTED, ABSTAIN, DATA_UNAVAILABLE
    entry_price = Column(Float)
    stop_loss_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    calculated_units = Column(Float)
    estimated_risk_amount = Column(Float)
    checks_json = Column(Text)
    rejection_reasons_json = Column(Text)
    timestamp = Column(DateTime, default=_now, index=True)


Index("ix_trades_symbol_timestamp", Trade.symbol, Trade.timestamp)
Index("ix_decisions_symbol_timestamp", AgentDecision.symbol, AgentDecision.timestamp)
Index("ix_stress_tests_symbol_timestamp", StressTestResult.symbol, StressTestResult.timestamp)
Index("ix_investment_plans_type_time", InvestmentPlan.plan_type, InvestmentPlan.timestamp)
Index("ix_rebalance_events_time", RebalanceEvent.timestamp)
Index("ix_horizon_symbol_time", HorizonAssessmentRecord.symbol, HorizonAssessmentRecord.timestamp)
Index("ix_conflicts_symbol_time", EvidenceConflictRecord.symbol, EvidenceConflictRecord.timestamp)
Index("ix_reliability_status_time", SystemReliabilityLog.overall_status, SystemReliabilityLog.timestamp)
Index("ix_forex_pre_event_id_time", ForexPreEventAssessmentRecord.event_id, ForexPreEventAssessmentRecord.timestamp)
Index("ix_forex_post_event_id_time", ForexPostEventAnalysisRecord.event_id, ForexPostEventAnalysisRecord.timestamp)
Index("ix_forex_regime_pair_time", ForexRegimeLogRecord.pair, ForexRegimeLogRecord.timestamp)
Index("ix_forex_memory_cat_curr", ForexEventMemoryRecord.category, ForexEventMemoryRecord.currency)
Index("ix_style_configs_style", TradingStyleConfigRecord.style, TradingStyleConfigRecord.is_active)
Index("ix_quality_gate_symbol_time", TradeQualityGateRecord.symbol, TradeQualityGateRecord.timestamp)

_engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}
                         if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from sqlalchemy import text, inspect
    Base.metadata.create_all(_engine)
    if settings.database_url.startswith("sqlite"):
        try:
            insp = inspect(_engine)
            cols = [c["name"] for c in insp.get_columns("agent_decisions")]
            desired_cols = {
                "abstention_reason": "TEXT",
                "selected_agents_json": "TEXT",
                "signal_confidence": "FLOAT",
                "prediction_confidence": "FLOAT",
                "evidence_quality": "FLOAT",
                "market_stability": "FLOAT",
                "risk_confidence": "FLOAT",
                "critic_analysis_json": "TEXT",
                "counterfactual_json": "TEXT",
                "risk_budget_json": "TEXT",
                "rag_citations_json": "TEXT",
            }
            with _engine.connect() as conn:
                for col_name, col_type in desired_cols.items():
                    if col_name not in cols:
                        try:
                            conn.execute(text(f"ALTER TABLE agent_decisions ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                        except Exception:
                            pass
        except Exception:
            pass


def get_session():
    return SessionLocal()

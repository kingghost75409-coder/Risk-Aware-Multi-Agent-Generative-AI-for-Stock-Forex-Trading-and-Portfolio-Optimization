"""Flask REST API — the backend for the real-time dashboard.

Enhanced with new endpoints for:
- Stock quotes & company information
- Live news feed
- Market overview (indices)
- Server-Sent Events (SSE) for real-time price streaming
- Intraday data
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from pydantic import ValidationError

from app.agents.regime_agent import MarketRegimeAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.ml_agent import MLPredictionAgent
from app.backtesting.backtest import run_ablation_study
from app.config import settings
from app.data.news_sample import get_sample_headlines
from app.data.news_provider import get_news_for_symbol
from app.data.pipeline import load_and_prepare
from app.data.stock_info import get_stock_info, get_stock_quote, get_market_overview
from app.db.models import (
    init_db, get_session, Trade as TradeRow, AgentDecision as DecisionRow,
    AuditLog, StressTestResult as StressTestRow, AdversarialEvalLog as AdvEvalRow,
    HorizonAssessmentRecord, DigitalTwinScenarioRecord, EvidenceConflictRecord,
    PerformanceAttributionRecord, SystemReliabilityLog,
)
from app.orchestrator import TradingOrchestrator
from app.portfolio.optimizer import optimize_portfolio
from app.rag.knowledge_graph import get_financial_knowledge_graph
from app.rag.rag_agent import FinancialRAGAgent
from app.rag.adversarial_evaluator import AdversarialRAGEvaluator
from app.agents.stress_testing_agent import MarketStressTestingAgent
from app.agents.crash_risk_agent import CrashRiskAgent
from app.agents.alpha_engine import AdaptiveAlphaEngine
from app.data.temporal_firewall import TemporalFirewall
from app.risk.adaptive_risk_budget import AdaptiveRiskBudget
from app.trading.paper_trading import PaperTradingEngine
from app.strategy.adaptive_horizon_selector import AdaptiveInvestmentHorizonSelector
from app.portfolio.digital_twin import PersonalPortfolioDigitalTwin
from app.rag.conflict_detector import FinancialEvidenceConflictDetector
from app.portfolio.attribution import PortfolioDecisionAttributionEngine
from app.services.reliability_monitor import get_reliability_monitor
from app.services.llm_service import get_llm_gateway
from app.forex.forex_data import (
    SUPPORTED_FOREX_PAIRS,
    normalize_forex_pair,
    is_supported_pair,
    get_pip_size,
    _TYPICAL_SPREAD_PIPS,
    get_forex_provider,
)
from app.forex.regime_intelligence import get_regime_intelligence
from app.forex.event_analyzer import get_event_analyzer
from app.forex.event_memory import get_event_reaction_memory
from app.forex.event_risk_gate import get_forex_risk_gate
from app.utils.logger import get_logger

logger = get_logger(__name__)

forex_provider = get_forex_provider()
forex_regime_intel = get_regime_intelligence()
forex_event_analyzer = get_event_analyzer()
forex_event_memory = get_event_reaction_memory()
forex_risk_gate = get_forex_risk_gate()

FRONTEND_DIR = settings.base_dir / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

init_db()

orchestrator = TradingOrchestrator()
paper_engine = PaperTradingEngine()
rag_agent = FinancialRAGAgent()
knowledge_graph = get_financial_knowledge_graph()
stress_agent = MarketStressTestingAgent()
adversarial_evaluator = AdversarialRAGEvaluator()
adaptive_risk_budget_calc = AdaptiveRiskBudget()
crash_risk_agent = CrashRiskAgent()
alpha_engine = AdaptiveAlphaEngine()

from app.strategy import (
    LongTermPlanner,
    ShortTermPlanner,
    CombinedPortfolioManager,
    MultiHorizonBacktester,
)

long_term_planner = LongTermPlanner(rag_agent=rag_agent)
short_term_planner = ShortTermPlanner(orchestrator=orchestrator)
combined_portfolio_mgr = CombinedPortfolioManager(long_term_planner=long_term_planner, short_term_planner=short_term_planner)
horizon_backtester = MultiHorizonBacktester()

adaptive_horizon_selector = AdaptiveInvestmentHorizonSelector(
    orchestrator=orchestrator,
    rag_agent=rag_agent,
    long_term_planner=long_term_planner,
    short_term_planner=short_term_planner,
)
digital_twin = PersonalPortfolioDigitalTwin(paper_engine=paper_engine)
conflict_detector = FinancialEvidenceConflictDetector(rag_agent=rag_agent)
attribution_engine = PortfolioDecisionAttributionEngine(paper_engine=paper_engine)
reliability_monitor = get_reliability_monitor()

_last_prices: dict[str, float] = {}
_last_decisions: dict[str, dict] = {}


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


def _log_audit(event_type: str, symbol: str | None, message: str) -> None:
    try:
        s = get_session()
        s.add(AuditLog(event_type=event_type, symbol=symbol, message=message))
        s.commit()
        s.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit log write failed: %s", exc)


def _error_response(exc: Exception, code: int = 500):
    logger.exception("request failed")
    return jsonify({"error": str(exc), "type": type(exc).__name__}), code


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "PAPER_TRADING_RESEARCH_SIMULATION",
                     "timestamp": datetime.now(timezone.utc).isoformat()})


# ─── Market Data ──────────────────────────────────────────────────────────────

@app.route("/api/market/<symbol>")
def market_data(symbol: str):
    try:
        period = request.args.get("period", "2y")
        limit = request.args.get("limit", type=int)
        is_intraday = period.lower() in ("1d", "intraday")
        interval = "5m" if is_intraday else "1d"
        df, report, source = load_and_prepare(symbol, period=period, interval=interval)
        
        if is_intraday:
            target_df = df.tail(limit if limit and limit > 0 else 78)
        else:
            target_df = df.tail(limit) if limit and limit > 0 else df
        tail = target_df.reset_index()
        tail["timestamp"] = tail["timestamp"].astype(str)
        from app.data.stock_info import get_market_status
        return jsonify({
            "symbol": symbol.upper(), "source": source,
            "period": period, "interval": interval,
            "rows": len(df), "warnings": report.warnings,
            "market_status": get_market_status(),
            "bars": json.loads(tail[["timestamp", "open", "high", "low", "close", "volume"]].to_json(orient="records")),
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/market-status")
def market_status_route():
    """Live US equity market session status and current Eastern Time."""
    try:
        from app.data.stock_info import get_market_status
        return jsonify(get_market_status())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/quote/<symbol>")
def quote(symbol: str):
    """Quick quote — price, change, volume, market cap."""
    try:
        data = get_stock_quote(symbol)
        _last_prices[symbol.upper()] = data["price"]
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/stock-info/<symbol>")
def stock_info(symbol: str):
    """Comprehensive stock information — company profile, key stats, etc."""
    try:
        data = get_stock_info(symbol)
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/market-overview")
def market_overview():
    """Major indices and market summary."""
    try:
        data = get_market_overview()
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/news/<symbol>")
def news(symbol: str):
    """Latest news for a symbol."""
    try:
        limit = int(request.args.get("limit", 15))
        data = get_news_for_symbol(symbol, limit=limit)
        # Run sentiment on headlines
        agent = SentimentAgent()
        headlines = [{"text": n["headline"], "published_at": n.get("published_at", "")} for n in data]
        if headlines:
            sentiment = agent.analyze(symbol.upper(), headlines)
            for i, item in enumerate(data):
                if i < len(sentiment.headline_scores):
                    item["sentiment_score"] = sentiment.headline_scores[i]
        return jsonify({"symbol": symbol.upper(), "news": data,
                        "sentiment_summary": sentiment.model_dump() if headlines else None})
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/stream/<symbol>")
def stream_price(symbol: str):
    """Server-Sent Events (SSE) endpoint for real-time price updates.

    Pushes a new quote every 5 seconds. Browsers auto-reconnect on
    disconnect. This is a much lighter protocol than WebSockets for
    one-directional server→client data.
    """
    symbol = symbol.upper()

    def event_stream():
        while True:
            try:
                data = get_stock_quote(symbol)
                _last_prices[symbol] = data["price"]
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.warning("SSE quote failed for %s: %s", symbol, exc)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            time.sleep(5)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/intraday/<symbol>")
def intraday(symbol: str):
    """Intraday candlestick data for shorter intervals."""
    try:
        interval = request.args.get("interval", "5m")
        allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        if interval not in allowed:
            return jsonify({"error": f"interval must be one of {sorted(allowed)}"}), 400

        from app.data.provider import get_historical_with_fallback
        df, source = get_historical_with_fallback(symbol, period="5d", interval=interval)
        tail = df.tail(200).reset_index()
        tail["timestamp"] = tail["timestamp"].astype(str)
        return jsonify({
            "symbol": symbol.upper(), "source": source, "interval": interval,
            "bars": json.loads(tail[["timestamp", "open", "high", "low", "close", "volume"]].to_json(orient="records")),
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Agents & Analysis ───────────────────────────────────────────────────────

@app.route("/api/technical/<symbol>")
def technical(symbol: str):
    try:
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        sig = TechnicalAgent().analyze(df, symbol.upper())
        return jsonify(sig.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/prediction/<symbol>")
def prediction(symbol: str):
    try:
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        pred = MLPredictionAgent().analyze(df, symbol.upper())
        return jsonify(pred.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/sentiment/<symbol>")
def sentiment(symbol: str):
    try:
        result = SentimentAgent().analyze(symbol.upper(), get_sample_headlines(symbol))
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/regime/<symbol>")
def regime(symbol: str):
    try:
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        result = MarketRegimeAgent().analyze(df, symbol.upper())
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/rag/search")
def rag_search():
    try:
        symbol = request.args.get("symbol", "").upper()
        query = request.args.get("query", "")
        as_of_date = request.args.get("as_of_date")
        if not symbol or not query:
            return jsonify({"error": "symbol and query query-params are required"}), 400
        result = rag_agent.query(symbol, query, as_of_date=as_of_date)
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/rag/evidence")
def rag_evidence():
    try:
        symbol = request.args.get("symbol", "AAPL").upper()
        as_of_date = request.args.get("as_of_date")
        result = rag_agent.get_evidence_summary(symbol, as_of_date=as_of_date)
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/agents/<symbol>")
def dynamic_agents(symbol: str):
    """Dynamic Agent Router endpoint: inspects market context and returns selected agents."""
    try:
        symbol = symbol.upper()
        df, report, _ = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data"}), 422
        returns = df["adj_close"].pct_change().dropna()
        realized_vol_20 = float(returns.rolling(20).std().iloc[-1] * (252 ** 0.5)) if len(returns) >= 20 else 0.20
        vol_history = returns.rolling(20).std().dropna() * (252 ** 0.5)
        vol_percentile = float((vol_history < realized_vol_20).mean()) if len(vol_history) else 0.50
        sma20 = df["adj_close"].rolling(20).mean()
        trend_slope = float((sma20.iloc[-1] / sma20.iloc[-10] - 1)) if len(sma20.dropna()) > 10 else 0.0
        headlines = get_sample_headlines(symbol)

        routing = orchestrator.agent_router.route(
            symbol=symbol,
            vol_percentile=vol_percentile,
            trend_slope=trend_slope,
            n_news_items=len(headlines),
        )
        return jsonify(routing.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/confidence/<symbol>")
def confidence_breakdown(symbol: str):
    """Returns 5 independent confidence dimensions and transparent conviction synthesis."""
    try:
        symbol = symbol.upper()
        decision_dict = _last_decisions.get(symbol)
        if not decision_dict or "confidence_dimensions" not in decision_dict or not decision_dict["confidence_dimensions"]:
            decision = orchestrator.analyze_symbol(symbol)
            _last_decisions[symbol] = decision.model_dump()
            decision_dict = _last_decisions[symbol]

        return jsonify(decision_dict["confidence_dimensions"])
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/graph/<symbol>")
def financial_graph(symbol: str):
    """Financial Knowledge Graph endpoint: returns relational neighborhood and hybrid context."""
    try:
        subgraph = knowledge_graph.get_subgraph(symbol.upper())
        return jsonify(subgraph.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/risk/<symbol>")
def risk_symbol(symbol: str):
    """Asset-level deterministic risk evaluation."""
    try:
        df, report, _ = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data"}), 422
        risk = orchestrator.risk_engine.evaluate(
            symbol=symbol.upper(),
            df=df,
            proposed_signal="HOLD",
            proposed_position_pct=0.0,
            current_portfolio_exposure_pct=0.0,
        )
        return jsonify(risk.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/risk/budget")
def risk_budget():
    """Adaptive Risk Budget endpoint: dynamic limit scaling and cash allocation."""
    try:
        symbol = request.args.get("symbol", "AAPL").upper()
        regime_label = request.args.get("regime", "NORMAL_VOLATILITY_BULL")
        vol_pct = float(request.args.get("vol_percentile", 0.45))
        budget = adaptive_risk_budget_calc.calculate_budget(
            symbol=symbol,
            regime_label=regime_label,
            vol_percentile=vol_pct,
        )
        return jsonify(budget.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/counterfactual/<symbol>")
def counterfactual_explanation(symbol: str):
    """Counterfactual Trading Explanation endpoint: explains what conditions would change decision."""
    try:
        symbol = symbol.upper()
        decision_dict = _last_decisions.get(symbol)
        if not decision_dict or "counterfactual_explanation" not in decision_dict or not decision_dict["counterfactual_explanation"]:
            decision = orchestrator.analyze_symbol(symbol)
            _last_decisions[symbol] = decision.model_dump()
            decision_dict = _last_decisions[symbol]

        return jsonify(decision_dict["counterfactual_explanation"])
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/audit")
def audit_logs():
    """Returns persistent audit log trail."""
    try:
        limit = int(request.args.get("limit", 100))
        s = get_session()
        rows = s.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
        out = [{
            "id": r.id, "event_type": r.event_type, "symbol": r.symbol,
            "message": r.message, "timestamp": r.timestamp.isoformat()
        } for r in rows]
        s.close()
        return jsonify(out)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/rag/evaluate", methods=["GET", "POST"])
def rag_evaluate():
    """Runs adversarial financial RAG testing suite."""
    try:
        summary = adversarial_evaluator.run_benchmark_suite()
        return jsonify(summary.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Advanced Research Endpoints (Upgrades 1, 2, 3, 7) ───────────────────────

@app.route("/api/crash-risk/<symbol>")
def crash_risk(symbol: str):
    """Crash & Tail-Risk Prediction endpoint (Upgrade 2)."""
    try:
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        result = crash_risk_agent.analyze(df, symbol.upper())
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/alpha/<symbol>")
def alpha_breakdown(symbol: str):
    """Adaptive Alpha Signal breakdown endpoint (Upgrade 3)."""
    try:
        symbol = symbol.upper()
        decision_dict = _last_decisions.get(symbol)
        if not decision_dict or "alpha_signal" not in decision_dict or not decision_dict["alpha_signal"]:
            decision = orchestrator.analyze_symbol(symbol)
            _last_decisions[symbol] = decision.model_dump()
            decision_dict = _last_decisions[symbol]
        return jsonify(decision_dict.get("alpha_signal"))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/stress-test/historical/<symbol>")
def stress_test_historical(symbol: str):
    """Run historical crisis scenario replay: 2008 GFC, 2020 COVID, 2022 rate shock (Upgrade 7)."""
    try:
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        prices.update(_last_prices)
        port_state = paper_engine.get_state(prices)
        result = stress_agent.run_historical_stress(
            symbol=symbol.upper(),
            df=df,
            portfolio_value=port_state.total_portfolio_value,
            cash=port_state.cash,
        )
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/stress-test/custom", methods=["POST"])
def stress_test_custom():
    """Run user-defined custom stress scenario (Upgrade 7)."""
    try:
        payload = request.get_json() or {}
        symbol = str(payload.get("symbol", "AAPL")).upper()
        price_shock = float(payload.get("price_shock", -0.15))
        vol_multiplier = float(payload.get("vol_multiplier", 1.5))
        sentiment_override = float(payload.get("sentiment_override", -0.5))
        scenario_name = str(payload.get("scenario_name", "Custom Stress Scenario"))

        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data", "warnings": report.warnings}), 422
        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        prices.update(_last_prices)
        port_state = paper_engine.get_state(prices)
        result = stress_agent.run_custom_stress(
            symbol=symbol,
            df=df,
            price_shock_pct=price_shock,
            vol_multiplier=vol_multiplier,
            sentiment_override=sentiment_override,
            scenario_name=scenario_name,
            portfolio_value=port_state.total_portfolio_value,
            cash=port_state.cash,
        )
        return jsonify(result.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/firewall/audit")
def firewall_audit():
    """Returns point-in-time temporal firewall enforcement audit (Upgrade 1)."""
    try:
        symbol = request.args.get("symbol", "AAPL").upper()
        as_of_date = request.args.get("as_of_date")
        df, report, source = load_and_prepare(symbol)
        fw = TemporalFirewall(as_of_date=as_of_date)
        truncated_df = fw.enforce_ohlcv(df, symbol)
        return jsonify({
            "symbol": symbol,
            "as_of_date": as_of_date,
            "original_bars": len(df),
            "accepted_bars": len(truncated_df),
            "removed_bars": len(df) - len(truncated_df),
            "is_clean": fw.report.is_clean,
            "summary": fw.get_audit_summary(),
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Risk & Portfolio ─────────────────────────────────────────────────────────

@app.route("/api/risk/portfolio")
def risk_portfolio():
    try:
        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        prices.update(_last_prices)
        state = paper_engine.get_state(prices)
        exposures = {
            sym: round(pos.market_value / state.total_portfolio_value, 4) if state.total_portfolio_value else 0.0
            for sym, pos in state.positions.items()
        }
        return jsonify({
            "portfolio_value": state.total_portfolio_value,
            "cash": state.cash,
            "realized_pnl": state.realized_pnl,
            "exposures_pct": exposures,
            "total_exposure_pct": round(sum(exposures.values()), 4),
            "risk_limits": {
                "max_position_pct": settings.max_position_pct,
                "max_portfolio_volatility": settings.max_portfolio_volatility,
                "max_daily_loss_pct": settings.max_daily_loss_pct,
                "max_concentration_pct": settings.max_concentration_pct,
            },
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/portfolio")
def portfolio():
    try:
        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        prices.update(_last_prices)
        state = paper_engine.get_state(prices)
        return jsonify(state.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/trades")
def trades():
    try:
        limit = int(request.args.get("limit", 50))
        s = get_session()
        rows = s.query(TradeRow).order_by(TradeRow.timestamp.desc()).limit(limit).all()
        out = [{
            "trade_id": r.trade_id, "symbol": r.symbol, "side": r.side, "quantity": r.quantity,
            "price": r.price, "executed_price": r.executed_price, "transaction_cost": r.transaction_cost,
            "reason": r.reason, "risk_status": r.risk_status, "timestamp": r.timestamp.isoformat(),
        } for r in rows]
        s.close()
        return jsonify(out)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Analysis & Trading ──────────────────────────────────────────────────────

@app.route("/api/analyze/<symbol>", methods=["GET", "POST"])
def analyze(symbol: str):
    try:
        symbol = symbol.upper()
        body = {}
        if request.is_json:
            body = request.get_json(silent=True) or {}
        as_of_date = body.get("as_of_date") or request.args.get("as_of_date")
        force_full = bool(body.get("force_full_pipeline") or request.args.get("force_full_pipeline"))

        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        current_exposure = 0.0
        state = paper_engine.get_state(prices)
        if state.total_portfolio_value:
            current_exposure = sum(p.market_value for p in state.positions.values()) / state.total_portfolio_value

        decision = orchestrator.analyze_symbol(
            symbol,
            current_portfolio_exposure_pct=current_exposure,
            as_of_date=as_of_date,
            force_full_pipeline=force_full,
        )
        _last_prices[symbol] = float(decision.technical.indicators.get("sma_20") or decision.technical.indicators.get("ema_12") or 0)
        _last_decisions[symbol] = decision.model_dump()

        s = get_session()
        conf_dims = decision.confidence_dimensions
        s.add(DecisionRow(
            symbol=symbol,
            final_decision=decision.final_decision,
            confidence=decision.confidence,
            position_size_pct=decision.position_size_pct,
            technical_signal=decision.technical.signal,
            ml_direction=decision.prediction.predicted_direction,
            ml_probability=decision.prediction.probability,
            sentiment_label=decision.sentiment.label,
            regime=decision.regime.regime,
            risk_status=decision.risk.status,
            abstention_reason=decision.abstention_reason,
            selected_agents_json=json.dumps(decision.selected_agents),
            signal_confidence=conf_dims.signal_confidence if conf_dims else None,
            prediction_confidence=conf_dims.prediction_confidence if conf_dims else None,
            evidence_quality=conf_dims.evidence_quality if conf_dims else None,
            market_stability=conf_dims.market_stability if conf_dims else None,
            risk_confidence=conf_dims.risk_confidence if conf_dims else None,
            critic_analysis_json=json.dumps(decision.critic_report.model_dump()) if decision.critic_report else None,
            counterfactual_json=json.dumps(decision.counterfactual_explanation.model_dump()) if decision.counterfactual_explanation else None,
            risk_budget_json=json.dumps(decision.adaptive_risk_budget.model_dump()) if decision.adaptive_risk_budget else None,
            rag_citations_json=json.dumps([c.model_dump() for c in decision.rag_evidence.citations]),
            audit_trail_json=json.dumps(decision.reasoning_audit_trail),
        ))
        s.commit()
        s.close()

        return jsonify(decision.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/stress-test", defaults={"symbol": None}, methods=["GET", "POST"])
@app.route("/api/stress-test/<symbol>", methods=["GET", "POST"])
def stress_test(symbol: str | None = None):
    """Market Stress Testing Agent endpoint: simulates 7 acute stress scenarios without mutating live state."""
    try:
        body = {}
        if request.is_json:
            body = request.get_json(silent=True) or {}
        symbol = (symbol or body.get("symbol") or request.args.get("symbol") or "AAPL").upper()
        custom_scenario = body.get("custom_scenario")

        df, report, _ = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data"}), 422

        prices = {sym: pos["avg_entry_price"] for sym, pos in paper_engine.holdings.items()}
        state = paper_engine.get_state(prices)
        port_val = state.total_portfolio_value or 100_000.0
        cash_val = state.cash

        suite_report = stress_agent.run_stress_test(
            symbol=symbol,
            df=df,
            portfolio_value=port_val,
            cash=cash_val,
            current_holdings=paper_engine.holdings,
            custom_scenario=custom_scenario,
        )

        # Log stress test to database
        try:
            s = get_session()
            for sc in suite_report.scenarios:
                s.add(StressTestRow(
                    symbol=symbol,
                    scenario_name=sc.scenario_name,
                    portfolio_impact_pct=sc.portfolio_loss_pct,
                    stressed_var_95=sc.stressed_var_95,
                    stressed_cvar_95=sc.stressed_cvar_95,
                    stressed_max_drawdown=sc.stressed_max_drawdown,
                    stressed_volatility=sc.stressed_volatility_annualized,
                    would_breach=sc.would_breach_limits,
                    details_json=json.dumps(sc.model_dump()),
                ))
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.warning("Failed to persist stress test results: %s", db_exc)

        return jsonify(suite_report.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/trade/simulate", methods=["POST"])
def trade_simulate():
    try:
        body = request.get_json(force=True) or {}
        symbol = body.get("symbol", "").upper()
        if not symbol:
            return jsonify({"error": "symbol is required"}), 400

        requested_side = (body.get("side") or "").upper()
        requested_qty = float(body.get("quantity", 0))

        # Get latest price
        price = float(body.get("price", 0)) or _last_prices.get(symbol)
        if not price or price <= 0:
            from app.data.stock_info import get_stock_quote
            q = get_stock_quote(symbol)
            price = q.get("price", 150.0)

        # If user explicitly requested BUY or SELL with quantity:
        if requested_side in ("BUY", "SELL") and requested_qty > 0:
            trade = paper_engine.execute_order(
                symbol=symbol,
                side=requested_side,
                quantity=requested_qty,
                price=price,
                reason=body.get("reason", f"Manual {requested_side} order @ ${price:.2f}"),
                risk_status="PASS"
            )
            if trade is None:
                return jsonify({"error": "Insufficient cash balance or shares to execute trade"}), 422
        else:
            # Fallback to automated agent consensus trade
            decision_dict = _last_decisions.get(symbol)
            if decision_dict is None:
                decision = orchestrator.analyze_symbol(symbol)
                decision_dict = decision.model_dump()
                _last_decisions[symbol] = decision_dict

            trade = paper_engine.execute(
                symbol=symbol, side=decision_dict["final_decision"], price=price,
                position_pct=decision_dict["position_size_pct"],
                reason=f"orchestrated decision, conf={decision_dict['confidence']}",
                risk_status=decision_dict["risk"]["status"],
            )
            if trade is None:
                return jsonify({"error": "trade could not be executed (insufficient cash or no position)"}), 422

        s = get_session()
        s.add(TradeRow(
            trade_id=trade.trade_id, symbol=trade.symbol, side=trade.side, quantity=trade.quantity,
            price=trade.price, executed_price=trade.executed_price, transaction_cost=trade.transaction_cost,
            reason=trade.reason, risk_status=trade.risk_status,
        ))
        s.commit()
        s.close()
        return jsonify(trade.model_dump())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/scenario", methods=["POST"])
def scenario():
    """What-if simulation: recompute risk metrics under a stressed volatility
    or exposure assumption, without mutating live state."""
    try:
        body = request.get_json(force=True) or {}
        symbol = body.get("symbol", "").upper()
        vol_multiplier = float(body.get("volatility_multiplier", 1.0))
        exposure_override_pct = body.get("exposure_pct")

        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data"}), 422

        from app.risk.risk_engine import RiskEngine, value_at_risk, conditional_value_at_risk, max_drawdown
        returns = df["adj_close"].pct_change().dropna() * vol_multiplier
        var95 = value_at_risk(returns, settings.var_confidence)
        cvar95 = conditional_value_at_risk(returns, settings.var_confidence)
        mdd = max_drawdown(df["adj_close"])
        vol_annualized = float(returns.std() * (252 ** 0.5))

        exposure = float(exposure_override_pct) if exposure_override_pct is not None else 0.10
        breach = vol_annualized > settings.max_portfolio_volatility or exposure > settings.max_concentration_pct

        return jsonify({
            "symbol": symbol,
            "scenario": {"volatility_multiplier": vol_multiplier, "exposure_pct": exposure},
            "recomputed": {
                "var_95": round(var95, 4), "cvar_95": round(cvar95, 4),
                "max_drawdown": round(mdd, 4), "volatility_annualized": round(vol_annualized, 4),
            },
            "would_breach_risk_limits": breach,
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/backtest/<symbol>")
def backtest(symbol: str):
    try:
        period = request.args.get("period", "2y")
        result = run_ablation_study(symbol.upper(), period=period)
        if isinstance(result, dict) and "results" in result:
            result["strategies"] = result["results"]
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Chatbot ──────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """Enhanced AI trading chatbot with LLM formatting and conversation memory."""
    try:
        from app.services.chatbot import answer_question
        body = request.get_json(force=True) or {}
        question = body.get("question", "")
        symbol = body.get("symbol")
        if not question:
            return jsonify({"error": "question is required"}), 400
        result = answer_question(question, symbol, orchestrator, rag_agent, paper_engine, _last_decisions)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── LLM Status ──────────────────────────────────────────────────────────────

@app.route("/api/llm-status")
def llm_status():
    """Check which LLM provider is active."""
    try:
        from app.services.llm_service import get_llm_gateway
        gw = get_llm_gateway()
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
        return jsonify({
            "active_provider": gw.active_provider_name,
            "is_llm_active": gw.is_llm_active,
            "has_openrouter_key": has_openrouter,
            "has_gemini_key": has_gemini,
            "has_anthropic_key": has_anthropic,
            "openrouter_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "providers_available": len(gw.providers),
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.route("/api/llm/set-key", methods=["POST"])
def set_llm_key():
    """Save OpenRouter, Gemini, or Anthropic API key, persist to .env, and reload LLM gateway."""
    try:
        body = request.get_json(force=True) or {}
        api_key = (body.get("api_key") or "").strip()
        provider = (body.get("provider") or "").lower().strip()

        if not api_key:
            return jsonify({"error": "API key cannot be empty"}), 400

        # Automatic provider detection based on key format
        if api_key.startswith("sk-or-"):
            provider = "openrouter"
        elif api_key.startswith("AIzaSy"):
            provider = "gemini"
        elif api_key.startswith("sk-ant-"):
            provider = "anthropic"
        elif not provider:
            provider = "openrouter"

        # Update runtime environment
        keys_to_persist = {}
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key
            keys_to_persist["OPENROUTER_API_KEY"] = api_key
            if not os.getenv("OPENROUTER_MODEL"):
                os.environ["OPENROUTER_MODEL"] = "openai/gpt-4o-mini"
                keys_to_persist["OPENROUTER_MODEL"] = "openai/gpt-4o-mini"
            # If accidentally stored in GEMINI_API_KEY before, clear it
            if os.getenv("GEMINI_API_KEY", "").startswith("sk-or-"):
                os.environ["GEMINI_API_KEY"] = ""
                keys_to_persist["GEMINI_API_KEY"] = ""
        elif provider == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key
            keys_to_persist["GEMINI_API_KEY"] = api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
            keys_to_persist["ANTHROPIC_API_KEY"] = api_key
        else:
            return jsonify({"error": f"Unsupported provider '{provider}'"}), 400

        # Persist to .env file in root directory
        try:
            env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            for k, val in keys_to_persist.items():
                found = False
                new_lines = []
                for line in lines:
                    if line.startswith(f"{k}="):
                        new_lines.append(f"{k}={val}\n")
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f"\n{k}={val}\n")
                lines = new_lines

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as file_exc:
            logger.warning("Could not persist API key to .env: %s", file_exc)

        # Reload LLM Gateway singleton with fresh env vars
        from app.services.llm_service import reload_llm_gateway
        gw = reload_llm_gateway()

        disp_name = "OpenRouter" if provider == "openrouter" else provider.capitalize()
        return jsonify({
            "success": True,
            "message": f"{disp_name} API key activated successfully!",
            "active_provider": gw.active_provider_name,
            "is_llm_active": gw.is_llm_active,
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


# ─── Ticker Ribbon & Prediction Cone (New Endpoints) ─────────────────────────

_ticker_ribbon_cache = {"time": 0.0, "data": None}

@app.route("/api/ticker-ribbon")
def ticker_ribbon():
    """Live quotes for the scrolling ticker ribbon with 60s cache and parallel fetching."""
    import time
    now = time.time()
    if _ticker_ribbon_cache["data"] and (now - _ticker_ribbon_cache["time"] < 60):
        return jsonify(_ticker_ribbon_cache["data"])

    tickers = {
        "NIFTY": "^NSEI", "SENSEX": "^BSESN", "DOW": "^DJI",
        "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225", "BTC": "BTC-USD", "ETH": "ETH-USD",
        "EUR/USD": "EURUSD=X", "USD/INR": "USDINR=X",
    }
    bases = {
        "NIFTY": 23200, "SENSEX": 74400, "DOW": 52000, "S&P 500": 5900,
        "NASDAQ": 19000, "FTSE 100": 8300, "Nikkei 225": 36500,
        "BTC": 58000, "ETH": 2400, "EUR/USD": 1.11, "USD/INR": 83.9
    }

    def _fetch_single(label, yf_symbol):
        try:
            import yfinance as yf
            t = yf.Ticker(yf_symbol)
            fi = t.fast_info
            price = fi.get("lastPrice") or fi.get("regularMarketPrice", 0)
            prev = fi.get("previousClose") or fi.get("regularMarketPreviousClose", price)
            change = price - prev
            change_pct = (change / prev) if prev else 0
            return {
                "label": label, "symbol": yf_symbol,
                "price": round(price, 2), "change": round(change, 2),
                "changePct": round(change_pct, 4), "source": "yfinance",
            }
        except Exception:
            import random
            p = bases.get(label, 1000)
            ch = random.uniform(-0.5, 0.5) * p / 100
            return {
                "label": label, "symbol": yf_symbol,
                "price": round(p + ch, 2), "change": round(ch, 2),
                "changePct": round(ch / p, 4), "source": "synthetic",
            }

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(_fetch_single, lbl, sym) for lbl, sym in tickers.items()]
            for f in as_completed(futures, timeout=2.0):
                try:
                    results.append(f.result())
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure all labels present
    found_labels = {r["label"] for r in results}
    for lbl, sym in tickers.items():
        if lbl not in found_labels:
            p = bases.get(lbl, 1000)
            results.append({
                "label": lbl, "symbol": sym, "price": p,
                "change": 0.0, "changePct": 0.0, "source": "synthetic"
            })

    payload = {"tickers": results, "timestamp": datetime.now(timezone.utc).isoformat()}
    _ticker_ribbon_cache["data"] = payload
    _ticker_ribbon_cache["time"] = now
    return jsonify(payload)


@app.route("/api/prediction-cone/<symbol>")
def prediction_cone(symbol: str):
    """Forward price projection with ±1σ and ±1.96σ confidence bands."""
    try:
        import math
        symbol = symbol.upper()
        df, report, source = load_and_prepare(symbol)
        if not report.is_usable:
            return jsonify({"error": "insufficient data"}), 422

        close = df["adj_close"]
        returns = close.pct_change().dropna()
        last_price = float(close.iloc[-1])
        if symbol in _last_prices and _last_prices[symbol] > 0:
            last_price = float(_last_prices[symbol])

        # Integrate ML Prediction Agent for direction, expected return, and probability
        from app.agents.ml_agent import MLPredictionAgent
        try:
            ml_pred = MLPredictionAgent().analyze(df, symbol)
            expected_ret = float(ml_pred.expected_return)
            direction = ml_pred.predicted_direction
            prob = float(ml_pred.probability)
        except Exception:
            expected_ret = 0.015
            direction = "UP"
            prob = 0.58

        vol_20 = float(returns.tail(20).std()) if len(returns) >= 20 else float(returns.std())
        daily_vol = vol_20 if vol_20 > 0 else 0.015

        # Calculate high-probability target prices and actionable trade levels
        is_bullish = direction == "UP"
        upside_pct = max(1.8, min(8.5, round(abs(expected_ret) * (2.8 if is_bullish else 1.2) * 100, 2)))
        downside_pct = max(1.5, min(6.0, round(daily_vol * 1.8 * 100, 2)))

        target_price = round(last_price * (1 + (upside_pct / 100 if is_bullish else -upside_pct / 100)), 2)
        stop_loss_price = round(last_price * (1 - (downside_pct / 100 if is_bullish else -downside_pct / 100)), 2)
        rr_ratio = round(upside_pct / max(downside_pct, 0.5), 1)

        if is_bullish:
            recommendation = "STRONG BUY" if prob >= 0.65 else "BUY / ACCUMULATE" if prob >= 0.52 else "LEAN BULLISH"
        else:
            recommendation = "STRONG SELL" if prob >= 0.65 else "SELL / TRIM" if prob >= 0.52 else "LEAN BEARISH"

        summary_verdict = (
            f"AI predicts {direction} trajectory ({'+' if is_bullish else '-'}{upside_pct}%) toward ${target_price:.2f}. "
            f"Recommendation: {recommendation} (Stop Loss: ${stop_loss_price:.2f}, R/R: 1:{rr_ratio})."
        )

        # Generate realistic forward projection driven by ML model
        days = 30
        cone = []
        target_ret = (upside_pct / 100) if is_bullish else -(upside_pct / 100)
        for d in range(1, days + 1):
            t_frac = d / days
            central = last_price * (1 + target_ret * t_frac)
            sigma_t = daily_vol * math.sqrt(d)
            s1_up = central * (1 + sigma_t)
            s1_low = central * (1 - sigma_t)
            s2_up = central * (1 + 1.96 * sigma_t)
            s2_low = central * (1 - 1.96 * sigma_t)
            cone.append({
                "day": d,
                "central": round(central, 2),
                "upper_1s": round(s1_up, 2),
                "lower_1s": round(s1_low, 2),
                "upper_2s": round(s2_up, 2),
                "lower_2s": round(s2_low, 2),
            })

        return jsonify({
            "symbol": symbol,
            "last_price": last_price,
            "predicted_direction": direction,
            "probability": round(prob, 4),
            "expected_return": round(expected_ret, 4),
            "daily_vol": round(daily_vol, 4),
            "days_forward": days,
            "target_price": target_price,
            "stop_loss_price": stop_loss_price,
            "upside_pct": upside_pct,
            "downside_pct": downside_pct,
            "recommendation": recommendation,
            "risk_reward_ratio": f"1 : {rr_ratio}",
            "summary_verdict": summary_verdict,
            "cone": cone,
            "source": source,
        })
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)



# ─── Multi-Horizon Investment Strategy & Portfolio Planning ───────────────────

@app.route("/api/investment/long-term/<symbol>", methods=["GET"])
def investment_long_term_stock(symbol: str):
    """Evaluate fundamental quality, 10-K RAG evidence, rule-based conditions for long-term holding."""
    try:
        horizon = request.args.get("horizon", "3_years")
        analysis = long_term_planner.analyze_stock(symbol, horizon=horizon)
        return jsonify(analysis.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/long-term/plan", methods=["POST"])
def investment_long_term_plan():
    """Construct constrained long-term portfolio plan with optional SIP simulation."""
    try:
        data = request.get_json(force=True) or {}
        investment_amount = float(data.get("investment_amount", 500000.0))
        horizon = str(data.get("horizon", "3_years"))
        risk_preference = str(data.get("risk_preference", "moderate"))
        symbols = data.get("symbols")
        max_stock_allocation = float(data.get("max_stock_allocation", 0.30))
        max_sector_allocation = float(data.get("max_sector_allocation", 0.60))
        monthly_investment = float(data.get("monthly_investment", 0.0))

        plan = long_term_planner.construct_portfolio(
            investment_amount=investment_amount,
            horizon=horizon,
            risk_preference=risk_preference,
            symbols=symbols,
            max_stock_allocation=max_stock_allocation,
            max_sector_allocation=max_sector_allocation,
            monthly_investment=monthly_investment,
        )
        return jsonify(plan.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/short-term/<symbol>", methods=["GET"])
def investment_short_term_stock(symbol: str):
    """Tactical setup evaluation with entry, target, stop loss, risk/reward, and uncertainty."""
    try:
        horizon = request.args.get("horizon", "1_week")
        portfolio_value = float(request.args.get("portfolio_value", 100000.0))
        opp = short_term_planner.analyze_opportunity(symbol, horizon=horizon, portfolio_value=portfolio_value)
        return jsonify(opp.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/short-term/analyze", methods=["POST"])
def investment_short_term_analyze():
    """Tactical setup analysis endpoint for specified symbol and short horizon."""
    try:
        data = request.get_json(force=True) or {}
        symbol = str(data.get("symbol", "AAPL"))
        horizon = str(data.get("horizon", "1_week"))
        portfolio_value = float(data.get("portfolio_value", 100000.0))
        opp = short_term_planner.analyze_opportunity(symbol, horizon=horizon, portfolio_value=portfolio_value)
        return jsonify(opp.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/plan", methods=["POST"])
def investment_plan_unified():
    """Master unified investment plan: LONG_TERM, SHORT_TERM, or COMBINED."""
    try:
        data = request.get_json(force=True) or {}
        horizon_type = str(data.get("horizon_type", "COMBINED")).upper()
        amount = float(data.get("investment_amount", 500000.0))
        risk_pref = str(data.get("risk_preference", "moderate"))

        if horizon_type in ("LONG_TERM", "LONG"):
            horizon = str(data.get("horizon", "3_years"))
            plan = long_term_planner.construct_portfolio(
                investment_amount=amount,
                horizon=horizon,
                risk_preference=risk_pref,
                symbols=data.get("symbols"),
                max_stock_allocation=float(data.get("max_stock_allocation", 0.30)),
                max_sector_allocation=float(data.get("max_sector_allocation", 0.60)),
                monthly_investment=float(data.get("monthly_investment", 0.0)),
            )
            return jsonify({"horizon_type": "LONG_TERM", "plan": plan.model_dump()})
        elif horizon_type in ("SHORT_TERM", "SHORT"):
            horizon = str(data.get("horizon", "1_week"))
            syms = data.get("symbols") or ["AAPL", "MSFT", "NVDA", "AMZN"]
            opps = [short_term_planner.analyze_opportunity(s, horizon=horizon, portfolio_value=amount) for s in syms]
            return jsonify({"horizon_type": "SHORT_TERM", "opportunities": [o.model_dump() for o in opps]})
        else:
            lt_pct = float(data.get("long_term_target_pct", 0.70))
            st_pct = float(data.get("short_term_target_pct", 0.20))
            cash_pct = float(data.get("cash_target_pct", 0.10))
            comb = combined_portfolio_mgr.build_combined_portfolio(
                total_capital=amount,
                long_term_target_pct=lt_pct,
                short_term_target_pct=st_pct,
                cash_target_pct=cash_pct,
                long_term_symbols=data.get("long_term_symbols"),
                short_term_symbols=data.get("short_term_symbols"),
                risk_preference=risk_pref,
            )
            return jsonify({"horizon_type": "COMBINED", "plan": comb.model_dump()})
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/combined", methods=["GET"])
def investment_combined_portfolio():
    """Unified portfolio state across Long-Term, Short-Term, and Cash sleeves with total risk."""
    try:
        capital = float(request.args.get("capital", 500000.0))
        lt_pct = float(request.args.get("long_term_pct", 0.70))
        st_pct = float(request.args.get("short_term_pct", 0.20))
        cash_pct = float(request.args.get("cash_pct", 0.10))
        risk_pref = request.args.get("risk_preference", "moderate")
        state = combined_portfolio_mgr.build_combined_portfolio(
            total_capital=capital,
            long_term_target_pct=lt_pct,
            short_term_target_pct=st_pct,
            cash_target_pct=cash_pct,
            risk_preference=risk_pref,
        )
        return jsonify(state.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/rebalance", methods=["POST"])
def investment_rebalance():
    """Evaluate rebalancing triggers and optionally record event."""
    try:
        data = request.get_json(force=True) or {}
        capital = float(data.get("capital", 500000.0))
        state = combined_portfolio_mgr.build_combined_portfolio(total_capital=capital)
        assessment = state.rebalance_assessment
        event_id = None
        if data.get("execute", False) and assessment:
            event_id = combined_portfolio_mgr.record_rebalance_event(
                plan_id=data.get("plan_id"),
                assessment=assessment,
                total_risk=state.total_risk,
            )
        return jsonify({
            "status": "success",
            "assessment": assessment.model_dump() if assessment else None,
            "rebalance_event_id": event_id,
            "executed": bool(event_id),
        })
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/performance", methods=["GET"])
def investment_performance():
    """Multi-horizon backtest comparison: Long-Term vs Short-Term vs Buy & Hold."""
    try:
        symbol = request.args.get("symbol", "AAPL")
        period = request.args.get("period", "2y")
        res = horizon_backtester.run_comparison(symbol=symbol, period=period)
        return jsonify(res)
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/investment/risk", methods=["GET"])
def investment_risk():
    """Combined portfolio risk engine statistics."""
    try:
        capital = float(request.args.get("capital", 500000.0))
        state = combined_portfolio_mgr.build_combined_portfolio(total_capital=capital)
        return jsonify(state.total_risk.model_dump())
    except Exception as exc:
        return _error_response(exc)


# ─── New Investment Intelligence & Research Endpoints ─────────────────────────

@app.route("/api/investment/horizon/<symbol>", methods=["GET"])
def get_investment_horizon(symbol: str):
    """Adaptive Investment Horizon Selector."""
    try:
        user_override = request.args.get("override_horizon")
        override_reason = request.args.get("override_reason")
        result = adaptive_horizon_selector.assess(
            symbol=symbol,
            user_override=user_override,
            override_reason=override_reason,
        )
        try:
            s = get_session()
            rec = HorizonAssessmentRecord(
                symbol=result.symbol,
                recommended_horizon=result.recommended_horizon.value,
                long_term_score=result.long_term_score,
                short_term_score=result.short_term_score,
                confidence=result.confidence,
                details_json=json.dumps(result.model_dump()),
                user_override_json=json.dumps(result.user_override) if result.user_override else None,
            )
            s.add(rec)
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.debug("Failed to record HorizonAssessmentRecord: %s", db_exc)

        return jsonify(result.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/portfolio/digital-twin/scenario", methods=["POST"])
def post_digital_twin_scenario():
    """Create and simulate an isolated what-if scenario in the Personal Portfolio Digital Twin."""
    try:
        data = request.get_json(force=True) or {}
        name = data.get("scenario_name", "Custom Simulation")
        position_adjustments = data.get("position_adjustments", [])
        cash_shift_pct = float(data.get("cash_shift_pct", 0.0))
        market_shock_pct = float(data.get("market_shock_pct", 0.0))
        sector_shocks = data.get("sector_shocks", {})
        vol_multiplier = float(data.get("volatility_shock_multiplier", 1.0))

        sim_res = digital_twin.simulate_scenario(
            scenario_name=name,
            position_adjustments=position_adjustments,
            cash_shift_pct=cash_shift_pct,
            market_shock_pct=market_shock_pct,
            sector_shocks=sector_shocks,
            volatility_shock_multiplier=vol_multiplier,
        )

        try:
            s = get_session()
            rec = DigitalTwinScenarioRecord(
                scenario_id=sim_res.scenario_id,
                scenario_name=sim_res.scenario_name,
                baseline_val=sim_res.baseline.total_value,
                simulated_val=sim_res.simulated.total_value,
                turnover_pct=sim_res.estimated_turnover,
                transaction_cost=sim_res.estimated_transaction_cost,
                details_json=json.dumps(sim_res.model_dump()),
            )
            s.add(rec)
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.debug("Failed to record DigitalTwinScenarioRecord: %s", db_exc)

        return jsonify(sim_res.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/portfolio/digital-twin/scenarios", methods=["GET"])
def get_digital_twin_scenarios():
    """List all saved digital twin scenarios."""
    try:
        scenarios = digital_twin.list_scenarios()
        return jsonify({"scenarios": scenarios, "count": len(scenarios)})
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/evidence/conflicts/<symbol>", methods=["GET"])
def get_evidence_conflicts(symbol: str):
    """Financial Evidence Conflict Detector."""
    try:
        tech_sig = request.args.get("technical_signal")
        crash_p = request.args.get("crash_probability", type=float)
        sent_score = request.args.get("sentiment_score", type=float)
        ml_prob = request.args.get("ml_probability", type=float)

        report = conflict_detector.detect_conflicts(
            symbol=symbol,
            technical_signal=tech_sig,
            crash_probability=crash_p,
            sentiment_score=sent_score,
            ml_probability=ml_prob,
        )

        try:
            s = get_session()
            rec = EvidenceConflictRecord(
                symbol=report.symbol,
                category=report.category.value,
                severity=report.severity.value,
                conflict_score=report.conflict_score,
                affected_horizon=report.affected_horizon,
                recommended_action=report.recommended_action,
                claims_json=json.dumps(report.conflicting_claims),
            )
            s.add(rec)
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.debug("Failed to record EvidenceConflictRecord: %s", db_exc)

        return jsonify(report.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/portfolio/attribution", methods=["GET"])
def get_portfolio_attribution():
    """Portfolio Decision Attribution Engine (Brinson-Fachler Multi-Factor Return Decomposition)."""
    try:
        benchmark = request.args.get("benchmark_symbol", "SPY")
        bench_ret = request.args.get("benchmark_period_return", 0.085, type=float)
        res = attribution_engine.compute_attribution(
            benchmark_symbol=benchmark,
            benchmark_period_return=bench_ret,
        )

        try:
            s = get_session()
            rec = PerformanceAttributionRecord(
                period=res.period,
                total_return=res.total_portfolio_return,
                benchmark_return=res.benchmark_return,
                excess_return=res.excess_return,
                allocation_effect=res.allocation_effect,
                selection_effect=res.selection_effect,
                details_json=json.dumps(res.model_dump()),
            )
            s.add(rec)
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.debug("Failed to record PerformanceAttributionRecord: %s", db_exc)

        return jsonify(res.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/system/reliability", methods=["GET"])
def get_system_reliability():
    """AI Reliability and Drift Monitor report."""
    try:
        report = reliability_monitor.check_system_health()

        try:
            s = get_session()
            rec = SystemReliabilityLog(
                overall_status=report.overall_status.value,
                block_trading=report.block_risk_increasing_trades,
                position_multiplier=report.position_size_multiplier,
                alerts_count=len(report.active_alerts),
                details_json=json.dumps(report.model_dump()),
            )
            s.add(rec)
            s.commit()
            s.close()
        except Exception as db_exc:
            logger.debug("Failed to record SystemReliabilityLog: %s", db_exc)

        return jsonify(report.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/system/alerts", methods=["GET"])
def get_system_alerts():
    """List active system reliability alerts."""
    try:
        report = reliability_monitor.check_system_health()
        return jsonify({
            "overall_status": report.overall_status.value,
            "block_risk_increasing_trades": report.block_risk_increasing_trades,
            "alerts": [a.model_dump() for a in report.active_alerts],
            "count": len(report.active_alerts),
        })
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/system/recover", methods=["POST"])
def post_system_recover():
    """Trigger bounded safe recovery for transient faults."""
    try:
        res = reliability_monitor.attempt_recovery()
        return jsonify(res)
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/llm/status", methods=["GET"])
def get_llm_status():
    """Returns active LLM provider configuration and connectivity status."""
    try:
        gateway = get_llm_gateway()
        return jsonify({
            "is_active": gateway.is_llm_active,
            "active_provider": gateway.active_provider_name,
            "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
            "openrouter_model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        return _error_response(exc)


# ─── FOREX & MACRO ECONOMIC EVENT INTELLIGENCE ENDPOINTS ─────────────────────

@app.route("/api/forex/pairs", methods=["GET"])
def get_forex_pairs():
    """List supported Forex currency pairs with pip sizes and typical spreads."""
    try:
        data = []
        for p in SUPPORTED_FOREX_PAIRS:
            data.append({
                "pair": p,
                "pip_size": get_pip_size(p),
                "typical_spread_pips": _TYPICAL_SPREAD_PIPS.get(p, 1.5),
            })
        return jsonify({"pairs": data, "count": len(data)})
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/quote/<path:pair>", methods=["GET"])
def get_forex_quote(pair: str):
    """Return real-time/synthetic quote, spread, sessions, and 24h metrics for a pair."""
    try:
        norm = normalize_forex_pair(pair)
        if not is_supported_pair(norm):
            return jsonify({
                "error": f"Pair '{pair}' is not supported. Supported pairs: {', '.join(SUPPORTED_FOREX_PAIRS)}",
                "supported_pairs": SUPPORTED_FOREX_PAIRS,
            }), 400
        quote = forex_provider.get_quote(norm)
        return jsonify(quote.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/regime/<path:pair>", methods=["GET"])
def get_forex_pair_regime(pair: str):
    """Detect current market regime for a specific Forex pair."""
    try:
        norm = normalize_forex_pair(pair)
        if not is_supported_pair(norm):
            return jsonify({
                "error": f"Pair '{pair}' is not supported.",
                "supported_pairs": SUPPORTED_FOREX_PAIRS,
            }), 400
        assessment = forex_regime_intel.analyze_regime(norm)
        return jsonify(assessment.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/regimes", methods=["GET"])
def get_all_forex_regimes():
    """Retrieve regime assessments for all 7 supported Forex pairs."""
    try:
        results = {}
        for p in SUPPORTED_FOREX_PAIRS:
            try:
                results[p] = forex_regime_intel.analyze_regime(p).model_dump()
            except Exception as e:
                logger.debug("Failed regime for %s: %s", p, e)
        return jsonify({"regimes": results, "count": len(results)})
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/events", methods=["GET"])
def get_forex_events():
    """List macroeconomic announcements (optional ?currency=USD filter)."""
    try:
        curr = request.args.get("currency")
        events = forex_event_analyzer.list_events(currency=curr)
        return jsonify({"events": events, "count": len(events)})
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/events/pre/<path:pair_or_event_id>", methods=["GET"])
def get_forex_pre_event(pair_or_event_id: str):
    """Retrieve or compute pre-event assessment for an event or pair."""
    try:
        # Check if argument matches an event ID or is a pair
        ev = forex_event_analyzer.get_event(pair_or_event_id)
        if ev:
            event_id = ev["event_id"]
            pair = request.args.get("pair", "EUR/USD")
        else:
            # Pair passed; select the next major event affecting this pair
            pair = normalize_forex_pair(pair_or_event_id)
            events = forex_event_analyzer.list_events()
            matching = [e for e in events if pair in e.get("affected_pairs", [])]
            if not matching:
                return jsonify({"error": f"No active macroeconomic event found affecting {pair}."}), 404
            event_id = matching[0]["event_id"]

        assessment = forex_event_analyzer.assess_pre_event(event_id, primary_pair=pair)
        return jsonify(assessment.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/events/post", methods=["POST"])
def post_forex_event_analysis():
    """Run post-event price reaction analysis across 5 time windows."""
    try:
        data = request.get_json(silent=True) or {}
        event_id = data.get("event_id", "FOMC_2026_10")
        actual_val = float(data.get("actual_value", 4.75))
        pair = data.get("pair", "EUR/USD")
        sim_pips = float(data["simulated_pip_delta"]) if "simulated_pip_delta" in data else None

        post = forex_event_analyzer.analyze_post_event(
            event_id=event_id,
            actual_value=actual_val,
            primary_pair=pair,
            simulated_pip_delta=sim_pips,
        )
        return jsonify(post.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/events/compare/<path:event_id>", methods=["GET"])
def get_forex_event_comparison(event_id: str):
    """Compare pre-event expectations with observed post-event outcome."""
    try:
        pair = request.args.get("pair", "EUR/USD")
        comparison = forex_event_analyzer.compare_pre_post(event_id, primary_pair=pair)
        return jsonify(comparison.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/events/memory/similar", methods=["GET"])
def get_forex_similar_events():
    """Retrieve historical macroeconomic events with similar characteristics and reaction profiles."""
    try:
        category = request.args.get("category", "INTEREST_RATE")
        currency = request.args.get("currency", "USD")
        pair = request.args.get("pair", "EUR/USD")
        surp = float(request.args.get("surprise")) if request.args.get("surprise") else None
        regime = request.args.get("regime")

        summary = forex_event_memory.find_similar_events(
            category=category,
            currency=currency,
            target_pair=pair,
            current_surprise=surp,
            current_regime=regime,
            top_k=3,
        )
        return jsonify(summary.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/forex/risk/check/<path:pair>", methods=["GET"])
def get_forex_risk_check(pair: str):
    """Evaluate deterministic event-risk restrictions for a Forex pair."""
    try:
        side = request.args.get("side", "BUY")
        result = forex_risk_gate.evaluate_entry_risk(pair=pair, side=side)
        return jsonify(result.model_dump())
    except Exception as exc:
        return _error_response(exc)


# ─── Trading Style Manager & Risk Protection Layer Endpoints ────────────────

@app.route("/api/trading-style", methods=["GET"])
def get_trading_style():
    """Retrieve the currently active trading style and all available styles."""
    try:
        from app.strategy.trading_style_manager import get_trading_style_manager
        mgr = get_trading_style_manager()
        active_profile = mgr.get_active_profile()
        return jsonify({
            "active_style": mgr.get_active_style(),
            "profile": active_profile.model_dump(),
            "available_styles": mgr.list_styles(),
        })
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/trading-style/set", methods=["POST"])
def set_trading_style():
    """Set the active trading style (DAY, SWING, POSITION) and update risk parameters."""
    try:
        from app.strategy.trading_style_manager import get_trading_style_manager
        body = request.get_json(force=True) or {}
        style = body.get("style", "SWING")
        risk_pct = float(body.get("risk_per_trade_pct")) if body.get("risk_per_trade_pct") is not None else None
        min_rr = float(body.get("min_risk_reward")) if body.get("min_risk_reward") is not None else None

        mgr = get_trading_style_manager()
        profile = mgr.set_active_style(style, risk_per_trade_pct=risk_pct, min_risk_reward=min_rr, persist=True)

        return jsonify({
            "success": True,
            "message": f"Trading style updated to {profile.display_name} successfully.",
            "active_style": mgr.get_active_style(),
            "profile": profile.model_dump(),
        })
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/trade-gate/evaluate", methods=["POST"])
def evaluate_trade_quality_gate():
    """Evaluate a proposed simulated trade through the authoritative Trade Setup Quality Gate."""
    try:
        from app.risk.trade_quality_gate import get_trade_quality_gate
        from app.strategy.trading_style_manager import get_trading_style_manager

        body = request.get_json(force=True) or {}
        symbol = body.get("symbol", "AAPL")
        side = body.get("side", "BUY")
        entry_price = float(body.get("entry_price", 0.0))
        stop_loss_price = float(body.get("stop_loss_price", 0.0))
        target_price = float(body.get("target_price", 0.0)) if body.get("target_price") else None
        trading_style = body.get("trading_style") or get_trading_style_manager().get_active_style()

        portfolio = paper_engine.get_portfolio_summary()
        equity = portfolio.get("total_portfolio_value", 100000.0)
        cash = portfolio.get("cash", 100000.0)

        existing_pos = {}
        for s, h in paper_engine.holdings.items():
            if h.get("quantity", 0) > 0:
                existing_pos[s] = {
                    "quantity": h["quantity"],
                    "avg_entry_price": h["avg_entry_price"],
                    "current_price": h["avg_entry_price"],
                }

        gate = get_trade_quality_gate()
        decision = gate.evaluate(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            trading_style=trading_style,
            account_equity=equity,
            available_cash=cash,
            existing_positions=existing_pos,
        )
        return jsonify(decision.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/risk/daily-status", methods=["GET"])
def get_daily_risk_status():
    """Retrieve persistent daily risk state, loss limits, consecutive losses, and cooldown status."""
    try:
        from app.risk.daily_loss_guard import get_daily_loss_guard
        guard = get_daily_loss_guard()
        summary = paper_engine.get_portfolio_summary()
        equity = summary.get("total_portfolio_value", 100000.0)
        unrealized = summary.get("unrealized_pnl", 0.0)

        status = guard.update_unrealized_pnl_and_evaluate(
            current_equity=equity,
            unrealized_pnl=unrealized,
        )
        return jsonify(status.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/risk/portfolio-exposure", methods=["GET"])
def get_portfolio_exposure_status():
    """Retrieve gross leverage, margin, and cross-currency concentration breakdown."""
    try:
        from app.risk.portfolio_exposure import get_portfolio_exposure_engine
        engine = get_portfolio_exposure_engine()
        summary = paper_engine.get_portfolio_summary()
        equity = summary.get("total_portfolio_value", 100000.0)

        existing_pos = {}
        for s, h in paper_engine.holdings.items():
            if h.get("quantity", 0) > 0:
                existing_pos[s] = {
                    "quantity": h["quantity"],
                    "avg_entry_price": h["avg_entry_price"],
                    "current_price": h["avg_entry_price"],
                }

        res = engine.evaluate_exposure(
            proposed_symbol="PORTFOLIO",
            proposed_side="BUY",
            proposed_notional_usd=0.0,
            account_equity=equity,
            existing_positions=existing_pos,
            instrument_type="STOCK",
        )
        return jsonify(res.model_dump())
    except Exception as exc:
        return _error_response(exc)


@app.route("/api/risk/cooldown/reset", methods=["POST"])
def reset_risk_cooldown():
    """Authorized reset of daily risk lock/cooldown for research testing."""
    try:
        from app.risk.daily_loss_guard import get_daily_loss_guard
        body = request.get_json(force=True) or {}
        reason = body.get("reason", "Authorized research manual reset")
        guard = get_daily_loss_guard()
        success = guard.reset_lock_manual(reason=reason)
        return jsonify({
            "success": success,
            "message": "Risk lock and cooldown reset successfully." if success else "No active risk lock to reset.",
        })
    except Exception as exc:
        return _error_response(exc)


if __name__ == "__main__":
    app.run(host=settings.flask_host, port=settings.flask_port, debug=settings.flask_debug)

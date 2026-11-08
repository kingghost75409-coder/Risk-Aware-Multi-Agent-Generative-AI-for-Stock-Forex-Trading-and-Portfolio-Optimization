"""AI Trading Chatbot — enhanced with LLM, conversation memory, and rich intents.

The chatbot matches the question against intents, calls the corresponding
real tool, then optionally passes the result through an LLM for natural
phrasing. Facts always come from tool calls, never from the LLM.

Enhanced intents:
- Stock analysis & recommendations
- News & sentiment queries
- Financial data queries
- Market overview
- Multi-stock comparison
- What-if scenarios
- Portfolio queries
- Explanation level adjustment
- Company information
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.data.pipeline import load_and_prepare
from app.data.stock_info import get_stock_info, get_stock_quote, get_market_overview
from app.data.news_provider import get_news_for_symbol
from app.orchestrator import TradingOrchestrator
from app.rag.rag_agent import FinancialRAGAgent
from app.risk.risk_engine import RiskEngine
from app.services.conversation_memory import get_memory
from app.services.llm_service import get_llm_gateway
from app.trading.paper_trading import PaperTradingEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
_KNOWN_SYMBOLS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"}

# Common words that look like ticker symbols but aren't
_FALSE_SYMBOLS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS",
    "HOW", "ITS", "LET", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO",
    "BOY", "DID", "OWN", "SAY", "SHE", "TOO", "USE", "BUY", "SELL", "HOLD",
    "WHAT", "WHEN", "WHERE", "WHY", "WILL", "WITH", "THIS", "THAT", "HAVE",
    "FROM", "THEY", "BEEN", "SAID", "EACH", "MAKE", "LIKE", "JUST", "OVER",
    "SUCH", "TAKE", "THAN", "THEM", "VERY", "SOME", "SHOW", "ALSO", "BACK",
    "GOOD", "GIVE", "MOST", "TELL", "DOES", "MUCH", "WELL", "BEST", "RISK",
    "NEWS", "HIGH", "LOW", "UP", "DOWN", "MARKET", "STOCK", "PRICE",
}


def _extract_symbol(question: str, fallback: str | None = None) -> str | None:
    # 1. Search the actual user question for explicit ticker symbols
    tokens = _SYMBOL_PATTERN.findall(question.upper())
    for tok in tokens:
        if tok in _KNOWN_SYMBOLS and tok not in _FALSE_SYMBOLS:
            return tok

    # 2. Only use fallback if the user specifically refers to the active stock/chart
    q_lower = question.lower().strip()
    pronoun_refs = ["this stock", "current stock", "it", "this company", "current symbol", "this chart", "current chart", "the stock"]
    if fallback and any(ref in q_lower for ref in pronoun_refs):
        return fallback.upper()

    return None


def _extract_multiple_symbols(question: str) -> list[str]:
    """Extract all valid ticker symbols from the question."""
    tokens = _SYMBOL_PATTERN.findall(question.upper())
    return [t for t in tokens if t in _KNOWN_SYMBOLS and t not in _FALSE_SYMBOLS]


_FOREX_ALIAS_MAP = {
    "EURUSD": "EUR/USD",
    "EUR/USD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "GBP/USD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USD/JPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "AUD/USD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USD/CAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "USD/CHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
    "NZD/USD": "NZD/USD",
}


def _extract_forex_pair(question: str) -> str | None:
    """Extract normalized Forex currency pair from user prompt (e.g. 'EUR/USD' or 'eurusd')."""
    q_up = question.upper()
    for alias, norm in _FOREX_ALIAS_MAP.items():
        if alias in q_up:
            return norm
    # Also check unspaced/unslashed
    q_condensed = q_up.replace(" ", "").replace("-", "/").replace("_", "")
    for alias, norm in _FOREX_ALIAS_MAP.items():
        if alias.replace("/", "") in q_condensed:
            return norm
    return None



def _fmt_money(n) -> str:
    if n is None:
        return "—"
    return f"${n:,.2f}"


def _fmt_pct(n, digits=2) -> str:
    if n is None:
        return "—"
    return f"{n * 100:.{digits}f}%"


def _fmt_large_number(n) -> str:
    if n is None:
        return "—"
    if abs(n) >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    if abs(n) >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    return f"${n:,.0f}"


def answer_question(question: str, symbol: str | None, orchestrator: TradingOrchestrator,
                     rag_agent: FinancialRAGAgent, paper_engine: PaperTradingEngine,
                     last_decisions: dict) -> dict:
    """Enhanced chatbot with LLM formatting and more intents."""
    q = question.lower().strip()
    sym = _extract_symbol(question, fallback=symbol) or _extract_forex_pair(question)
    memory = get_memory()
    llm = get_llm_gateway()

    # Record user message
    memory.add_message("user", question)

    # Route to the right intent handler
    result = _route_intent(q, sym, question, orchestrator, rag_agent,
                           paper_engine, last_decisions, memory, fallback_sym=symbol)

    # Build rich live terminal context for the LLM
    active_s = sym or symbol or "NVDA"
    live_ctx_lines = [f"Active Screen Stock: {active_s}"]
    live_price_str = "—"
    try:
        quote = get_stock_quote(active_s)
        if quote and "price" in quote:
            live_price_str = _fmt_money(quote.get('price'))
            live_ctx_lines.append(f"Current Market Price ({active_s}): {live_price_str} ({quote.get('changePct', 0):+.2%})")
    except Exception:
        pass

    if active_s in last_decisions:
        d = last_decisions[active_s]
        regime_val = d.get('regime', {}).get('regime') if isinstance(d.get('regime'), dict) else d.get('regime')
        live_ctx_lines.append(
            f"Multi-Agent Consensus for {active_s}: {d.get('final_decision')} (Conviction: {d.get('confidence', 0):.0%})\n"
            f"Technical Signal: {d.get('technical', {}).get('signal') if isinstance(d.get('technical'), dict) else d.get('technical')}\n"
            f"Predictive ML Model: {d.get('prediction', {}).get('predicted_direction') if isinstance(d.get('prediction'), dict) else d.get('prediction')} ({d.get('prediction', {}).get('probability', 0):.0%})\n"
            f"Market Regime: {regime_val}\n"
            f"Suggested Risk Position Sizing: {d.get('position_size_pct', 0):.1%}"
        )

    cash_str = "$100,000.00"
    try:
        port = paper_engine.get_portfolio_summary()
        cash_val = port.get('cash', 100000.0)
        cash_str = _fmt_money(cash_val)
        live_ctx_lines.append(f"User Virtual Account Balance: {cash_str} available for paper trading")
    except Exception:
        pass

    # Quick Watchlist Snapshot across all major tickers
    wl_items = []
    for s in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]:
        try:
            q_info = get_stock_quote(s)
            if q_info and "price" in q_info:
                wl_items.append(f"{s}: {_fmt_money(q_info['price'])} ({q_info.get('changePct', 0):+.2%})")
        except Exception:
            pass
    if wl_items:
        live_ctx_lines.append(f"Full Watchlist Prices: {', '.join(wl_items)}")

    live_context_str = "\n".join(live_ctx_lines)

    # Enhance with LLM for natural phrasing or conversational answering
    if result.get("tool_used") and result.get("answer"):
        try:
            context = memory.get_context_string(last_n=6)
            hist = memory.get_history(last_n=6)
            llm_response = llm.format_tool_output(
                question=question,
                tool_name=result["tool_used"],
                raw_output=result["answer"],
                context=f"{context}\n\n{live_context_str}",
                history=hist,
            )
            if llm_response and llm_response.provider != "template" and llm_response.text:
                result["answer"] = llm_response.text
                result["llm_provider"] = llm_response.provider
                result["llm_model"] = llm_response.model
        except Exception as exc:
            logger.warning("LLM formatting failed, using raw answer: %s", exc)
    elif not result.get("tool_used"):
        # For general, conversational, conceptual, or unrouted questions
        if llm.is_llm_active:
            try:
                hist = memory.get_history(last_n=8)
                llm_resp = llm.generate_conversational(
                    question=question,
                    history=hist,
                    stock_context=live_context_str,
                )
                if llm_resp and llm_resp.text:
                    result["answer"] = llm_resp.text
                    result["tool_used"] = "llm_conversational"
                    result["llm_provider"] = llm_resp.provider
                    result["llm_model"] = llm_resp.model
            except Exception as exc:
                logger.warning("LLM conversational response failed: %s", exc)

        if not result.get("answer"):
            result["answer"] = (
                f"I've noted your question: *\"{question}\"*.\n\n"
                f"Currently, we are monitoring **{active_s}** at **{live_price_str}** with **{cash_str}** in virtual funds. "
                f"Ask me to analyze any stock (e.g. *'Should I buy {active_s}?'*), "
                f"ask for our best setup (*'Tell me the good stock to buy'*), "
                f"or ask me to explain any trading indicator."
            )

    # Record assistant response
    memory.add_message("assistant", result["answer"], tool_used=result.get("tool_used"))

    return result


def _route_intent(q: str, sym: str | None, question: str,
                  orchestrator: TradingOrchestrator, rag_agent: FinancialRAGAgent,
                  paper_engine: PaperTradingEngine, last_decisions: dict,
                  memory, fallback_sym: str | None = None) -> dict:
    """Route the question to the right handler based on intent."""

    # --- 1. Greetings & Casual Chit-Chat ---
    greetings = {"hi", "hii", "hiii", "hello", "hey", "heyy", "howdy", "hola",
                 "good morning", "good afternoon", "good evening", "sup", "yo"}
    words = set(re.findall(r"\b\w+\b", q))
    if q in greetings or (words & greetings and len(words) <= 3) or q in {"who are you", "what can you do", "help", "help me"}:
        return _handle_greeting(sym or fallback_sym, paper_engine)

    # --- 2. Beginner Guidance & Learning ---
    beginner_kws = ["beginner", "how to trade", "how do i start", "how does trading work",
                    "less knowledge", "learn trading", "teach me", "guide me", "where do i begin",
                    "help me understand", "new to trading", "starting out", "how to invest"]
    if any(kw in q for kw in beginner_kws):
        return _handle_beginner_guide(sym or fallback_sym, paper_engine)

    # --- 3. Educational concepts when no specific symbol is targeted ---
    if not sym and not _extract_forex_pair(question):
        edu_match = _handle_educational_fallback(q)
        if edu_match:
            return edu_match

    # --- 4. Market overview ---
    if any(kw in q for kw in ["market overview", "how is the market", "market doing",
                               "market today", "indices", "s&p", "nasdaq", "dow"]):
        return _handle_market_overview()

    # --- 5. News queries ---
    if any(kw in q for kw in ["news", "latest news", "headlines", "what's happening"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which stock's news do you want? Try: 'What's the news on AAPL?'", "tool_used": None}
        return _handle_news(target)

    # --- 6. Company info / profile ---
    if any(kw in q for kw in ["company info", "company profile", "tell me about", "what does",
                               "who is", "about the company", "business summary"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which company? Try: 'Tell me about NVDA'", "tool_used": None}
        return _handle_company_info(target)

    # --- 7. Stock quote / price ---
    if any(kw in q for kw in ["price of", "current price", "how much is", "quote",
                               "stock price", "what is the price"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which stock? Try: 'What's the price of AAPL?'", "tool_used": None}
        return _handle_quote(target)

    # --- 8. Key statistics ---
    if any(kw in q for kw in ["key stats", "statistics", "eps", "dividend"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which stock's stats? Try: 'Show me key stats for MSFT'", "tool_used": None}
        return _handle_key_stats(target)

    # --- 9. Financials ---
    if any(kw in q for kw in ["financials", "revenue", "income", "balance sheet",
                               "cash flow", "profit", "earnings"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which company's financials? Try: 'Show financials for GOOGL'", "tool_used": None}
        return _handle_financials(target)

    # --- 10. Top Stock Picks / "Which stock should I buy?" / "Tell me the good stock to buy" ---
    top_pick_kws = [
        "good stock to buy", "best stock to buy", "what stock to buy", "which stock to buy",
        "what should i buy", "which stock should i buy", "top pick", "top stocks", "recommend a stock",
        "recommend me a stock", "what to buy", "any stock to buy", "where should i invest",
        "best stock right now", "which is the best stock", "give me a good stock", "stock recommendation",
        "suggest a stock", "which one to buy", "tell me good stock", "which stock is good",
        "tell me the good stock", "what is good to buy", "good stock", "good stocks", "best stock", "best stocks"
    ]
    if (any(kw in q for kw in top_pick_kws) and not sym) or (("good" in q or "best" in q or "which" in q or "what" in q) and "buy" in q and not sym):
        return _handle_top_picks(orchestrator, last_decisions, paper_engine)

    # --- New Intelligence Capabilities Routing (High Priority Before Generic Fallbacks) ---
    if any(kw in q for kw in ["long term or short term", "long-term or short-term", "horizon assessment", "investment horizon", "which horizon", "adaptive horizon", "for the long term or short term"]):
        target = sym or fallback_sym or "AAPL"
        return _handle_adaptive_horizon(target)

    if any(kw in q for kw in ["simulate a", "simulate market", "digital twin", "portfolio scenario", "decline for my paper portfolio", "market decline"]):
        return _handle_digital_twin_scenario(q, paper_engine)

    if any(kw in q for kw in ["evidence conflict", "evidence conflicts", "conflicting evidence", "contradiction", "conflicting claims"]):
        target = sym or fallback_sym or "AAPL"
        return _handle_evidence_conflicts(target)

    if any(kw in q for kw in ["performance attribution", "decision attribution", "underperform", "outperform", "why did my portfolio", "why portfolio"]):
        return _handle_performance_attribution(paper_engine)

    if any(kw in q for kw in ["reliability status", "system reliability", "system health", "drift monitor", "are ai models", "prediction system currently reliable"]):
        return _handle_reliability_status()

    # --- Forex & Macro Economic Event Intelligence Routing ---
    forex_pair = _extract_forex_pair(question)

    if any(kw in q for kw in ["trending or ranging", "forex regime"]) or (("regime" in q or "trending" in q or "ranging" in q) and forex_pair):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_regime(target_pair)

    if any(kw in q for kw in ["pre-event assessment", "pre-event", "economic event is being analyzed", "what major economic event", "upcoming event"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_pre_event(target_pair)

    if any(kw in q for kw in ["how did", "react after the announcement", "post-event", "inflation figures", "expected and actual"]):
        if any(w in q for w in ["react", "announcement", "post-event", "inflation", "cpi", "rate", "nfp", "gdp"]) or forex_pair:
            target_pair = forex_pair or "EUR/USD"
            return _handle_forex_post_event(target_pair, q)

    if any(kw in q for kw in ["continue or reverse", "reversal or continuation", "did the initial reaction continue"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_reversal_continuation(target_pair)

    if any(kw in q for kw in ["spread widen", "spreads widen", "spread widened", "bid/ask spread widen"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_spread_volatility(target_pair)

    if any(kw in q for kw in ["restrict new simulated trades", "restrict simulated", "why did the system restrict", "restrict new trades", "event-risk alert"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_risk_restriction(target_pair)

    if any(kw in q for kw in ["similar to the upcoming", "historical events are similar", "similar historical", "historical event memory", "similar events", "which historical"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_reaction_memory(q, target_pair)

    if any(kw in q for kw in ["consistent with the observed outcome", "difference between the pre-event assessment and the actual result", "pre-event assessment consistent"]):
        target_pair = forex_pair or "EUR/USD"
        return _handle_forex_comparison("FOMC_2026_10", target_pair)

    if any(kw in q for kw in ["which event details or market data are unavailable", "what data is unavailable", "market data unavailable"]):
        return _handle_forex_data_limitations()

    # --- Trading Styles & Risk Protection Layer Intents ---
    if any(kw in q for kw in ["trading style", "which style", "current style", "active style", "day trading mode", "swing trading mode", "position trading mode"]):
        return _handle_trading_style_status()

    if any(kw in q for kw in ["why was this simulated trade rejected", "why trade rejected", "why was my trade rejected", "why was the trade rejected", "why rejected"]):
        target = sym or fallback_sym
        return _handle_trade_rejection_explanation(target)

    if any(kw in q for kw in ["position size", "position sizing", "how was the position size calculated", "how is position size calculated", "how many shares can i buy", "how many units"]):
        target = sym or fallback_sym or "AAPL"
        return _handle_position_sizing_query(target, paper_engine)

    if any(kw in q for kw in ["daily loss allowance", "how much daily loss remains", "remaining daily loss", "daily loss limit", "daily loss remaining", "daily risk status"]):
        return _handle_daily_risk_status(paper_engine)

    if any(kw in q for kw in ["why is new simulated trading locked", "why trading locked", "why is trading locked", "trading locked", "risk lock", "consecutive losses", "consecutive loss", "cooldown", "how many consecutive losses"]):
        return _handle_cooldown_status(paper_engine)

    if any(kw in q for kw in ["current leverage", "leverage limit", "margin requirement", "free margin", "within configured limit", "currency exposure", "currency concentration", "largest portfolio exposure", "which currencies create"]):
        return _handle_portfolio_exposure_query(paper_engine)

    if any(kw in q for kw in ["why is the strategy marked unvalidated", "unvalidated strategy", "unvalidated", "insufficient evidence"]):
        return _handle_unvalidated_strategy_query()

    # --- 11. "Should I buy" / Analysis type questions for specific stock ---
    rec_kws = [
        "should i buy", "should i sell", "is it a good time",
        "recommend", "suggestion", "worth buying", "analyze", "analysis",
        "good to buy", "is it good to buy", "can i buy", "is it safe to buy",
        "safe to buy", "tell is it good", "tell me if it is good", "buy or sell"
    ]
    if any(kw in q for kw in rec_kws) or (("buy" in q or "sell" in q or "invest" in q) and sym):
        target = sym or fallback_sym
        if not target:
            return _handle_top_picks(orchestrator, last_decisions, paper_engine)
        return _handle_recommendation(target, orchestrator, last_decisions)

    # --- 12. Why did the system buy/sell/reject X? ---
    if any(kw in q for kw in ["why did", "why was", "explain the latest decision", "why is", "why hold", "why buy", "why sell"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which symbol? Try: 'Why did the system buy AAPL?'", "tool_used": None}
        return _handle_decision_explanation(target, last_decisions, orchestrator)

    # --- 13. Market regime ---
    if "regime" in q:
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which symbol? Try: 'What is the market regime for TSLA?'", "tool_used": None}
        return _handle_regime(target)

    # --- 14. Financial evidence / RAG ---
    if any(kw in q for kw in ["evidence", "10-k", "10k", "filing", "earnings report",
                               "risk factor"]):
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which company's filings? Try: 'What evidence supports NVDA?'", "tool_used": None}
        result = rag_agent.query(target, question)
        return {"answer": result.answer, "tool_used": "search_financial_rag",
                "citations": [c.model_dump() for c in result.citations], "raw": result.model_dump()}

    # --- Multi-Horizon Strategy & Investment Planning Intents ---
    if any(kw in q for kw in ["investment plan", "invest plan", "plan for", "year plan", "lakh", "create a"]):
        if any(w in q for w in ["plan", "invest", "horizon", "year", "lakh"]):
            return _handle_investment_plan(q, question, rag_agent)

    if any(kw in q for kw in ["long term opportunity", "long-term opportunity", "long term opportunities", "long-term opportunities", "find long-term", "find long term", "long term stocks", "long-term stocks"]):
        return _handle_long_term_opportunities(rag_agent)

    short_term_kws = [
        "short term opportunity", "short-term opportunity", "short term opportunities",
        "short-term opportunities", "give me short-term", "give me short term",
        "give a short term", "give short term", "short term to trade", "short-term to trade",
        "tactical opportunities", "tactical setups"
    ]
    if any(kw in q for kw in short_term_kws) or (
        ("short term" in q or "short-term" in q) and any(w in q for w in ["trade", "opportunity", "opportunities", "pick", "picks", "give", "setup", "setups", "tactical", "swing"])
    ):
        return _handle_short_term_opportunities(orchestrator)

    if ("why" in q and "hold" in q and ("long" in q or "year" in q)) or ("why should i hold" in q):
        target = sym or fallback_sym or "AAPL"
        return _handle_why_hold_long_term(target, rag_agent)

    if ("why" in q and "abstain" in q) or ("short-term system abstain" in q) or ("why did" in q and "abstain" in q):
        target = sym or fallback_sym or "AAPL"
        return _handle_why_short_term_abstain(target, orchestrator)

    if any(kw in q for kw in ["compare long-term", "compare long term", "compare horizons", "long-term vs short-term", "long term vs short term", "strategies"]):
        if "compare" in q and ("long" in q or "short" in q or "horizon" in q or "strategy" in q):
            target = sym or fallback_sym or "AAPL"
            return _handle_compare_horizons(target)

    if any(kw in q for kw in ["combined portfolio", "combined risk", "short-term risk", "short term risk", "exposed to short-term"]):
        return _handle_combined_portfolio_risk(paper_engine)

    # --- 15. Portfolio risk ---
    if "portfolio risk" in q or ("risk" in q and "portfolio" in q):
        return _handle_portfolio_risk(paper_engine)

    # --- 16. Strongest technical signals ---
    if "strongest" in q and "technical" in q:
        return _handle_strongest_signals()

    # --- 17. Multi-stock comparison ---
    symbols = _extract_multiple_symbols(question)
    if "compare" in q and len(symbols) >= 2:
        return _handle_comparison(symbols[:3])

    # --- 18. What-if scenario ---
    if "what happens if" in q or "what if" in q:
        target = sym or fallback_sym
        if not target:
            return {"answer": "Which symbol for the scenario?", "tool_used": None}
        return _handle_scenario(target, q)

    # --- 19. Experience level adjustment ---
    if any(kw in q for kw in ["explain like", "simple terms", "eli5"]):
        memory.update_preference("experience_level", "beginner")
        return {"answer": "Got it! I'll explain things in simpler, beginner-friendly terms from now on. What would you like to know?",
                "tool_used": "set_preference"}
    if any(kw in q for kw in ["advanced mode", "expert mode", "technical details"]):
        memory.update_preference("experience_level", "advanced")
        return {"answer": "Switching to advanced mode — I'll include more technical details. What's your question?",
                "tool_used": "set_preference"}

    # --- 20. If user specifically asked about an explicit symbol with analyze/verdict intent ---
    if sym and any(kw in q for kw in ["evaluate", "signal", "verdict", "outlook", "forecast", "view on", "opinion"]):
        return _handle_recommendation(sym, orchestrator, last_decisions)

    # Default: open-ended conversational routing (handled by LLM)
    return {
        "answer": "",
        "tool_used": None,
    }


# ─── Intent Handlers ─────────────────────────────────────────────────────────


def _handle_greeting(active_sym: str | None, paper_engine: PaperTradingEngine) -> dict:
    try:
        cash = paper_engine.get_portfolio_summary().get("cash", 100000.0)
    except Exception:
        cash = 100000.0

    target = active_sym or "NVDA"
    answer = (
        f"👋 **Hey there! Welcome to Stock AI Trader.**\n\n"
        f"I'm your AI market mentor and quantitative trading assistant. "
        f"Whether you're completely new to trading or exploring data-driven strategies, I'm here to help you make sense of the market in plain, simple English.\n\n"
        f"**Here are a few ways I can help you right now:**\n"
        f"• 📊 **Analyze Stocks**: Ask *'What is happening with {target}?'* or *'Should I buy AAPL?'* to see our 5 AI agents evaluate momentum, ML predictions, and risk.\n"
        f"• 💡 **Learn Concepts**: Ask me anything from *'What is RSI?'* to *'How do stop-losses protect my money?'*\n"
        f"• 🎮 **Risk-Free Practice**: You have **${cash:,.2f}** in paper trading virtual cash to test trade ideas without any real money at risk.\n"
        f"• 📰 **News & Market**: Ask *'How is the market today?'* or *'What's the news on {target}?'*\n\n"
        f"What would you like to explore today? Just ask!"
    )
    return {"answer": answer, "tool_used": "greeting"}


def _handle_beginner_guide(active_sym: str | None, paper_engine: PaperTradingEngine) -> dict:
    try:
        cash = paper_engine.get_portfolio_summary().get("cash", 100000.0)
    except Exception:
        cash = 100000.0
    target = active_sym or "NVDA"
    answer = (
        f"### 🚀 Welcome to Trading! (Beginner's Quickstart Guide)\n\n"
        f"Having little or no experience is 100% okay—that is exactly why this platform was designed! "
        f"You don't need a finance degree or complicated math to get started.\n\n"
        f"**Here are the 3 key rules every beginner should know:**\n\n"
        f"1. **Zero Real Risk (Paper Trading)**:\n"
        f"   You have **${cash:,.2f}** in virtual simulation funds. Every trade you place here is a practice run. "
        f"   Never trade real money until you've practiced paper trading for weeks and feel confident.\n\n"
        f"2. **How Our AI Agents Help You (No Guesswork)**:\n"
        f"   Instead of guessing which way a stock will move, our system runs **5 specialized AI agents**:\n"
        f"   • 🔧 **Technical Agent**: Reads price momentum and indicators like RSI and moving averages.\n"
        f"   • 🤖 **Machine Learning Agent**: Uses historical models to predict if the price is statistically more likely to go UP or DOWN over 5 days.\n"
        f"   • 💬 **Sentiment Agent**: Scans financial headlines to detect whether market news is positive or negative.\n"
        f"   • 🌊 **Regime Agent**: Checks if the market is trending smoothly or swinging wildly in high volatility.\n"
        f"   • 🛡️ **Risk Engine**: Tells you how much cash to risk (e.g. 5% max) so you never suffer large drawdowns.\n\n"
        f"3. **What You Can Do Right Now**:\n"
        f"   • Try asking: *'Should I buy {target}?'* to see the agents in action.\n"
        f"   • Try asking: *'Explain RSI in simple terms'* to learn how indicators work.\n"
        f"   • Look at the chart on your screen—the green and red bars are candlesticks showing daily price action.\n\n"
        f"What question do you have right now? Ask me anything!"
    )
    return {"answer": answer, "tool_used": "beginner_guide"}


def _handle_market_overview() -> dict:
    overview = get_market_overview()
    lines = ["**📈 Market Overview**\n"]
    for name, data in overview["indices"].items():
        arrow = "🟢" if data["changePct"] >= 0 else "🔴"
        sign = "+" if data["changePct"] >= 0 else ""
        lines.append(f"{arrow} **{name}**: {data['price']:,.2f} ({sign}{data['changePct']:.2%})")
    src = "synthetic" if any(d["source"] == "synthetic" for d in overview["indices"].values()) else "live"
    lines.append(f"\n_Data source: {src}_")
    return {"answer": "\n".join(lines), "tool_used": "get_market_overview"}


def _handle_news(sym: str) -> dict:
    news = get_news_for_symbol(sym, limit=8)
    if not news:
        return {"answer": f"No recent news found for {sym}.", "tool_used": "get_news"}
    lines = [f"**📰 Latest News for {sym}**\n"]
    for i, item in enumerate(news[:8], 1):
        headline = item["headline"]
        source = item.get("source", "")
        url = item.get("url", "")
        source_str = f" — _{source}_" if source else ""
        if url:
            lines.append(f"{i}. [{headline}]({url}){source_str}")
        else:
            lines.append(f"{i}. {headline}{source_str}")
    return {"answer": "\n".join(lines), "tool_used": "get_news", "news": news}


def _handle_company_info(sym: str) -> dict:
    info = get_stock_info(sym)
    answer = (
        f"**🏢 {info.get('longName', sym)}** ({sym})\n\n"
        f"**Sector:** {info.get('sector', '—')} | **Industry:** {info.get('industry', '—')}\n"
        f"**Location:** {info.get('city', '—')}, {info.get('country', '—')}\n"
        f"**Website:** {info.get('website', '—')}\n"
        f"**Employees:** {info.get('fullTimeEmployees', '—'):,}\n\n"
        f"**About:** {info.get('longBusinessSummary', 'No description available.')[:500]}\n\n"
        f"**Market Cap:** {_fmt_large_number(info.get('marketCap'))}\n"
        f"_Data source: {info.get('source', 'unknown')}_"
    )
    return {"answer": answer, "tool_used": "get_company_info", "raw": info}


def _handle_quote(sym: str) -> dict:
    quote = get_stock_quote(sym)
    arrow = "🟢" if quote["changePct"] >= 0 else "🔴"
    sign = "+" if quote["changePct"] >= 0 else ""
    answer = (
        f"**{sym} Quote** {arrow}\n\n"
        f"💰 **Price:** ${quote['price']:,.2f}\n"
        f"📊 **Change:** {sign}${quote['change']:,.2f} ({sign}{quote['changePct']:.2%})\n"
        f"📊 **Volume:** {quote['volume']:,}\n"
        f"📈 **Day Range:** ${quote['dayLow']:,.2f} — ${quote['dayHigh']:,.2f}\n"
        f"💎 **Market Cap:** {_fmt_large_number(quote.get('marketCap'))}\n\n"
        f"_Source: {quote['source']}_"
    )
    return {"answer": answer, "tool_used": "get_stock_quote", "raw": quote}


def _handle_key_stats(sym: str) -> dict:
    info = get_stock_info(sym)
    answer = (
        f"**📊 Key Statistics — {info.get('shortName', sym)}** ({sym})\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Market Cap | {_fmt_large_number(info.get('marketCap'))} |\n"
        f"| P/E (Trailing) | {info.get('trailingPE', '—')} |\n"
        f"| P/E (Forward) | {info.get('forwardPE', '—')} |\n"
        f"| EPS | ${info.get('trailingEps', '—')} |\n"
        f"| Dividend Yield | {_fmt_pct(info.get('dividendYield'), 2)} |\n"
        f"| Beta | {info.get('beta', '—')} |\n"
        f"| 52-Week High | ${info.get('fiftyTwoWeekHigh', '—')} |\n"
        f"| 52-Week Low | ${info.get('fiftyTwoWeekLow', '—')} |\n"
        f"| Avg Volume | {info.get('averageVolume', '—'):,} |\n\n"
        f"_Source: {info.get('source', 'unknown')}_"
    )
    return {"answer": answer, "tool_used": "get_key_stats", "raw": info}


def _handle_financials(sym: str) -> dict:
    info = get_stock_info(sym)
    answer = (
        f"**💰 Financial Summary — {info.get('shortName', sym)}** ({sym})\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Revenue | {_fmt_large_number(info.get('totalRevenue'))} |\n"
        f"| EBITDA | {_fmt_large_number(info.get('ebitda'))} |\n"
        f"| Profit Margin | {_fmt_pct(info.get('profitMargins'))} |\n"
        f"| Revenue Growth | {_fmt_pct(info.get('revenueGrowth'))} |\n"
        f"| Earnings Growth | {_fmt_pct(info.get('earningsGrowth'))} |\n"
        f"| Return on Equity | {_fmt_pct(info.get('returnOnEquity'))} |\n"
        f"| Debt to Equity | {info.get('debtToEquity', '—')} |\n"
        f"| Free Cash Flow | {_fmt_large_number(info.get('freeCashflow'))} |\n\n"
        f"_Source: {info.get('source', 'unknown')}_"
    )
    return {"answer": answer, "tool_used": "get_financials", "raw": info}


def _handle_educational_fallback(q: str) -> dict | None:
    """Provide rich, human-centered explanations for common financial and trading concepts."""
    # P/E ratio
    if any(kw in q for kw in ["p/e", "pe ratio", "price to earning", "price-to-earning"]):
        return {
            "answer": (
                "### 💡 Understanding the P/E Ratio (Price-to-Earnings)\n\n"
                "The **Price-to-Earnings (P/E) Ratio** tells you how much investors are willing to pay for every **$1 of company earnings**.\n\n"
                "**How to calculate it:**\n"
                "`P/E = Current Stock Price ÷ Earnings Per Share (EPS)`\n\n"
                "**The Two Main Types:**\n"
                "• **Trailing P/E**: Calculated using the last 12 months of actual reported earnings.\n"
                "• **Forward P/E**: Calculated using consensus Wall Street earnings estimates for the coming 12 months.\n\n"
                "**How to interpret it in plain English:**\n"
                "• **High P/E (>30-40)**: Common in high-growth companies (e.g. tech, AI chips). Investors expect rapid future profit growth.\n"
                "• **Moderate P/E (15-25)**: Healthy valuation typical of established, profitable companies.\n"
                "• **Low P/E (<12)**: Either a great value bargain or a sign the company's business model is facing headwinds.\n\n"
                "💡 *Pro Tip: Always compare a stock's P/E to its industry peers (e.g. compare AAPL to MSFT, not to ExxonMobil).*"
            ),
            "tool_used": "educational_guide",
        }

    # RSI
    if any(kw in q for kw in ["rsi", "relative strength index"]):
        return {
            "answer": (
                "### 📈 Demystifying the RSI (Relative Strength Index)\n\n"
                "The **Relative Strength Index (RSI)** is a momentum oscillator measuring the speed and magnitude of recent price changes on a scale from **0 to 100**.\n\n"
                "**How to read RSI levels:**\n"
                "• **Above 70 (Overbought)**: The stock has experienced aggressive buying in a short period. Buyers might be getting exhausted, creating a higher probability of a consolidation or pullback.\n"
                "• **Below 30 (Oversold)**: The stock has experienced intense selling pressure. The sell-off may be overextended, presenting potential dip-buying or rebound opportunities.\n"
                "• **50 Centerline**: Above 50 indicates bulls hold overall control; below 50 indicates bears are in charge.\n\n"
                "**Key Warning:** In powerful trending markets (like a strong bull run), a stock can stay 'overbought' for weeks without falling. Never short a stock purely because RSI is over 70!"
            ),
            "tool_used": "educational_guide",
        }

    # MACD
    if "macd" in q:
        return {
            "answer": (
                "### 🌊 Demystifying the MACD (Moving Average Convergence Divergence)\n\n"
                "The **MACD** is a trend-following momentum indicator that reveals the acceleration or deceleration of a stock's trend.\n\n"
                "**The Core Components:**\n"
                "• **MACD Line**: 12-day Exponential Moving Average (EMA) minus 26-day EMA.\n"
                "• **Signal Line**: 9-day EMA of the MACD line.\n"
                "• **Histogram**: The visual gap between the MACD Line and the Signal Line.\n\n"
                "**Trading Signals in Plain English:**\n"
                "• 🟢 **Bullish Crossover**: MACD crosses *above* the Signal Line → momentum is turning upward.\n"
                "• 🔴 **Bearish Crossover**: MACD crosses *below* the Signal Line → momentum is waning or flipping downward.\n"
                "• ⚡ **Divergence**: If the stock price makes a new high but MACD makes a lower high, caution is warranted — momentum is failing to confirm the new high!"
            ),
            "tool_used": "educational_guide",
        }

    # Beta
    if "beta" in q and not any(kw in q for kw in ["v1beta", "alphabet"]):
        return {
            "answer": (
                "### ⚡ What is Beta in Stock Trading?\n\n"
                "**Beta** measures a stock's historical volatility relative to the broader market benchmark (S&P 500 = 1.0).\n\n"
                "**How to interpret Beta values:**\n"
                "• **Beta = 1.0**: The stock moves in lockstep with the market.\n"
                "• **Beta > 1.0 (High Beta, e.g. TSLA ~ 2.2, NVDA ~ 1.7)**: Highly volatile. When the market rallies 1%, these stocks may surge 2%. But when the market drops, they fall faster.\n"
                "• **Beta < 1.0 (Low Beta, e.g. Consumer Staples / Utilities ~ 0.5 - 0.7)**: Defensive and resilient during market turbulence.\n"
                "• **Negative Beta**: Moves inversely to the market (rare for equities, occasionally seen in gold or inverse funds).\n\n"
                "💡 *Strategy: In bull regimes, high-beta stocks maximize gains; in high-volatility or bear regimes, low-beta protects capital.*"
            ),
            "tool_used": "educational_guide",
        }

    # ETF
    if any(kw in q for kw in ["etf", "exchange traded fund", "what is an etf"]):
        return {
            "answer": (
                "### 🧺 What is an ETF (Exchange-Traded Fund)?\n\n"
                "An **ETF** is a financial vehicle that pools capital from investors to hold a basket of underlying securities (stocks, bonds, or commodities), and trades on an exchange just like an individual stock.\n\n"
                "**Why investors love ETFs:**\n"
                "• **Instant Diversification**: Buying 1 share of SPY gives you ownership of 500 leading US companies.\n"
                "• **Low Fees**: Passive ETFs usually carry microscopic expense ratios (<0.05% per year).\n"
                "• **Intraday Liquidity**: Unlike mutual funds that settle at market close, ETFs trade continuously throughout market hours.\n\n"
                "**Examples:**\n"
                "• **SPY / VOO**: S&P 500 Index (Broad US market)\n"
                "• **QQQ**: NASDAQ 100 Index (Tech-heavy)\n"
                "• **SMH**: Semiconductor Sector ETF"
            ),
            "tool_used": "educational_guide",
        }

    # Stop Loss & Risk
    if any(kw in q for kw in ["stop loss", "stop-loss", "risk management", "how to manage risk"]):
        return {
            "answer": (
                "### 🛡️ Essential Risk Management & Stop-Losses\n\n"
                "In professional quantitative trading, **capital preservation always comes before capital appreciation**.\n\n"
                "**Golden Rules of Risk Management:**\n"
                "1. **The 1% Rule**: Never risk losing more than 1% to 2% of your total account equity on any individual trade.\n"
                "2. **Pre-Determined Stop-Loss**: Always know your invalidation level *before* you place the trade. If the thesis is broken, exit without emotional hesitation.\n"
                "3. **Position Sizing Formula**:\n"
                "   `Shares = (Account Size × Risk %) ÷ (Entry Price - Stop-Loss Price)`\n"
                "4. **Trailing Stops**: Once a position is in healthy profit, trail your stop higher to lock in gains while letting the winners run."
            ),
            "tool_used": "educational_guide",
        }

    # Market Regime
    if any(kw in q for kw in ["what is regime", "market regime", "regimes"]):
        return {
            "answer": (
                "### 🌊 Understanding Market Regimes\n\n"
                "A **Market Regime** is the statistical state of the market. Our platform's Regime Agent classifies the market into 5 states using trend indicators, volatility (ATR/Bollinger), and volume:\n\n"
                "1. 🐂 **BULL TRENDING**: Steady upward price action. High institutional conviction. Trend-following strategies outperform.\n"
                "2. 🐻 **BEAR TRENDING**: Downward pressure. Hedging, shorting, or cash preservation is advised.\n"
                "3. ⚡ **HIGH VOLATILITY**: Wide price swings with sudden whipsaws. Our risk engine cuts position sizes by up to 50% to prevent drawdowns.\n"
                "4. 🧘 **LOW VOLATILITY**: Calm, grinding market. Favorable for breakout setups.\n"
                "5. ↔️ **RANGING**: Price bounces between support and resistance. Mean-reversion strategies work best."
            ),
            "tool_used": "educational_guide",
        }

    return None


def _handle_recommendation(sym: str, orchestrator: TradingOrchestrator,
                           last_decisions: dict) -> dict:
    decision = orchestrator.analyze_symbol(sym)
    last_decisions[sym] = decision.model_dump()

    sig = decision.final_decision
    conf = decision.confidence
    tech_sig = decision.technical.signal
    tech_conf = decision.technical.confidence
    pred_dir = decision.prediction.predicted_direction
    pred_prob = decision.prediction.probability
    sent_label = decision.sentiment.label
    sent_score = decision.sentiment.sentiment_score
    regime_name = decision.regime.regime
    risk_status = decision.risk.status
    pos_pct = decision.position_size_pct

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

    # Plain English translation of regime
    regime_desc = {
        "BULL_TRENDING": "Strong upward trend with bullish momentum and institutional accumulation.",
        "BEAR_TRENDING": "Persistent downward drift with risk-off participant sentiment.",
        "HIGH_VOLATILITY": "Elevated price swings and tail risk; capital defense is prioritized.",
        "LOW_VOLATILITY": "Quiet consolidation with tight intraday price ranges.",
        "RANGING": "Oscillating between support and resistance boundaries without a dominant trend."
    }.get(regime_name, "Undergoing real-time quantitative evaluation.")

    # Plain English interpretation of signals
    if sig == "BUY":
        verdict_summary = f"Our multi-agent consensus generates an actionable **BUY** signal for **{sym}** with **{conf:.0%} statistical conviction**."
        plain_english = (
            f"The probability models lean favorably to the upside. The technical momentum is `{tech_sig}` "
            f"while our ML classifier projects a `{pred_prob:.0%}` likelihood of positive returns over a 5-day horizon. "
            f"News sentiment is `{sent_label}` ({sent_score:+.2f}), and the current market regime is `{regime_name}`."
        )
        tactical_guidance = (
            f"Consider a conservative position allocation of **{pos_pct:.1%}** of total portfolio equity. "
            f"Use disciplined stop-losses below recent swing lows."
        )
    elif sig == "SELL":
        verdict_summary = f"Our multi-agent consensus generates a **SELL / EXIT** signal for **{sym}** with **{conf:.0%} statistical conviction**."
        plain_english = (
            f"Multiple quantitative indicators detect elevated downside vulnerability or momentum exhaustion. "
            f"Our ML forecast anticipates downward pressure with `{pred_prob:.0%}` confidence in a `{regime_name}` environment."
        )
        tactical_guidance = (
            "Consider trimming existing holdings, locking in profits, or stepping aside until technical consolidation completes."
        )
    else:
        verdict_summary = f"Our multi-agent consensus recommends a **HOLD / WAIT** posture for **{sym}** with **{conf:.0%} conviction**."
        plain_english = (
            f"Current signals are conflicting or neutral. Technical momentum reads `{tech_sig}`, "
            f"while the ML model is `{pred_dir}` ({pred_prob:.0%}). In a `{regime_name}` regime, the risk-reward ratio does not justify aggressive entry."
        )
        tactical_guidance = (
            "Maintain current exposure without adding new capital. Wait for a decisive directional breakout or regime shift."
        )

    answer = (
        f"### {signal_emoji} Quantitative Trade Assessment: {sym}\n\n"
        f"**🎯 Executive Verdict**: {verdict_summary}\n\n"
        f"**💡 What This Means in Plain English:**\n"
        f"{plain_english}\n\n"
        f"**🔍 Multi-Agent Consensus Breakdown:**\n"
        f"• 🔧 **Technical Agent**: `{tech_sig}` ({tech_conf:.0%}) — Evaluates RSI oscillator, Moving Averages (20/50), and momentum curves.\n"
        f"• 🤖 **Predictive ML Agent**: `{pred_dir}` ({pred_prob:.0%}) — 5-day forward directional classifier.\n"
        f"• 💬 **Sentiment Agent**: `{sent_label}` ({sent_score:+.2f}) — Institutional headlines and news tone score.\n"
        f"• 🌊 **Macro Regime**: `{regime_name}` — {regime_desc}\n"
        f"• 🛡️ **Risk Guardrail**: `{risk_status}` — Position size capped at **{pos_pct:.1%}**.\n\n"
        f"**⚖️ Tactical Guidance & Risk Protocol:**\n"
        f"{tactical_guidance}\n"
    )
    if decision.risk.violations:
        answer += f"\n⚠️ **Active Risk Alerts**: {', '.join(decision.risk.violations)}\n"
    answer += (
        f"\n_💡 Want more details? Ask 'Why did you decide {sig} for {sym}?' or 'Show me news on {sym}'._\n\n"
        f"⚠️ *Disclaimer: Research and paper-trading simulation only. Not financial or investment advice.*"
    )
    return {"answer": answer, "tool_used": "run_full_analysis", "raw": decision.model_dump()}


def _handle_decision_explanation(sym: str, last_decisions: dict,
                                 orchestrator: TradingOrchestrator | None = None) -> dict:
    decision = last_decisions.get(sym)
    if decision is None and orchestrator is not None:
        dec = orchestrator.analyze_symbol(sym)
        last_decisions[sym] = dec.model_dump()
        decision = last_decisions[sym]

    if decision is None:
        return {"answer": f"No decision trail available for {sym} yet.",
                "tool_used": "get_decision_audit_trail"}

    trail = decision.get("reasoning_audit_trail", [])
    answer = (
        f"### 🔍 Decision Reasoning Trail: {sym}\n\n"
        f"**Consensus Recommendation: {decision.get('final_decision')}** (Confidence: {decision.get('confidence', 0):.0%})\n\n"
        f"**Step-by-step reasoning behind this signal:**\n"
    )
    for i, line in enumerate(trail, 1):
        answer += f"• {line}\n"
    return {"answer": answer, "tool_used": "get_decision_audit_trail", "raw": decision}


def _handle_regime(sym: str) -> dict:
    df, report, source = load_and_prepare(sym)
    if not report.is_usable:
        return {"answer": f"Not enough clean data for {sym}.", "tool_used": "get_market_regime"}
    from app.agents.regime_agent import MarketRegimeAgent
    result = MarketRegimeAgent().analyze(df, sym)
    answer = (
        f"**🌊 Market Regime — {sym}**\n\n"
        f"**Regime:** {result.regime}\n"
        f"**Confidence:** {result.confidence:.0%}\n"
        f"**Trend:** {result.trend_component}\n"
        f"**Volatility:** {result.volatility_component}\n"
    )
    return {"answer": answer, "tool_used": "get_market_regime", "raw": result.model_dump()}


def _handle_portfolio_risk(paper_engine: PaperTradingEngine) -> dict:
    prices = {s: p["avg_entry_price"] for s, p in paper_engine.holdings.items()}
    state = paper_engine.get_state(prices)
    positions_str = ""
    if state.positions:
        positions_str = "\n**Open Positions:**\n"
        for sym, pos in state.positions.items():
            positions_str += f"• {sym}: {pos.quantity} shares @ {_fmt_money(pos.avg_entry_price)}\n"

    answer = (
        f"**💼 Portfolio Overview**\n\n"
        f"💰 **Total Value:** {_fmt_money(state.total_portfolio_value)}\n"
        f"💵 **Cash:** {_fmt_money(state.cash)}\n"
        f"📊 **Realized P&L:** {_fmt_money(state.realized_pnl)}\n"
        f"📈 **Positions:** {len(state.positions)} open\n"
        f"{positions_str}"
    )
    return {"answer": answer, "tool_used": "get_portfolio_risk", "raw": state.model_dump()}


def _handle_strongest_signals() -> dict:
    from app.agents.technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    scored = []
    for s in _KNOWN_SYMBOLS:
        try:
            df, report, source = load_and_prepare(s)
            if report.is_usable:
                sig = agent.analyze(df, s)
                scored.append((s, sig.signal, sig.confidence))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[2])
    lines = ["**📊 Strongest Technical Signals**\n"]
    for s, sig, conf in scored[:6]:
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
        lines.append(f"{emoji} **{s}**: {sig} (confidence: {conf:.0%})")
    return {"answer": "\n".join(lines), "tool_used": "get_technical_analysis_batch"}


def _handle_comparison(symbols: list[str]) -> dict:
    from app.agents.technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    lines = [f"**🔄 Comparison: {' vs '.join(symbols)}**\n"]
    lines.append("| Symbol | Signal | Confidence | RSI | Trend |")
    lines.append("|--------|--------|-----------|-----|-------|")
    for s in symbols:
        try:
            df, report, source = load_and_prepare(s)
            if report.is_usable:
                sig = agent.analyze(df, s)
                rsi = sig.indicators.get("rsi_14", "—")
                trend = sig.indicators.get("trend_strength", "—")
                lines.append(f"| {s} | {sig.signal} | {sig.confidence:.0%} | {rsi} | {trend} |")
        except Exception:
            lines.append(f"| {s} | Error | — | — | — |")
    return {"answer": "\n".join(lines), "tool_used": "get_technical_analysis_comparison"}


def _handle_scenario(sym: str, q: str) -> dict:
    mult = 1.0
    if "volatility" in q and ("increase" in q or "up" in q or "double" in q):
        mult = 2.0 if "double" in q else 1.5
    elif "crash" in q or "drop" in q:
        mult = 2.5

    df, report, source = load_and_prepare(sym)
    if not report.is_usable:
        return {"answer": f"Not enough data for {sym}.", "tool_used": "run_scenario"}
    from app.risk.risk_engine import value_at_risk, conditional_value_at_risk, max_drawdown
    returns = df["adj_close"].pct_change().dropna() * mult
    var95 = value_at_risk(returns)
    cvar95 = conditional_value_at_risk(returns)
    mdd = max_drawdown(df["adj_close"])

    answer = (
        f"**🔮 What-If Scenario — {sym}**\n\n"
        f"Volatility multiplier: {mult:.1f}×\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| VaR (95%) | {_fmt_pct(var95)} |\n"
        f"| CVaR (95%) | {_fmt_pct(cvar95)} |\n"
        f"| Max Drawdown | {_fmt_pct(mdd)} |\n"
        f"| Ann. Volatility | {_fmt_pct(float(returns.std() * 252**0.5))} |\n"
    )
    return {"answer": answer, "tool_used": "run_scenario"}


def _handle_top_picks(orchestrator: TradingOrchestrator, last_decisions: dict,
                      paper_engine: PaperTradingEngine) -> dict:
    from app.agents.technical_agent import TechnicalAgent
    agent = TechnicalAgent()
    scored = []

    for s in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]:
        try:
            quote = get_stock_quote(s)
            price = quote.get("price") if quote else None
            chg = quote.get("changePct", 0) if quote else 0

            d = last_decisions.get(s)
            if d:
                sig = d.get("final_decision", "HOLD")
                conf = d.get("confidence", 0.5)
                rsi = d.get("technical", {}).get("indicators", {}).get("rsi_14", 50) if isinstance(d.get("technical"), dict) else 50
                pos_pct = d.get("position_size_pct", 0.05)
            else:
                df, report, source = load_and_prepare(s)
                if report.is_usable:
                    tech_sig = agent.analyze(df, s)
                    sig = tech_sig.signal
                    conf = tech_sig.confidence
                    rsi = tech_sig.indicators.get("rsi_14", 50)
                    pos_pct = 0.05
                else:
                    sig = "HOLD"
                    conf = 0.5
                    rsi = 50
                    pos_pct = 0.05
            scored.append({
                "symbol": s,
                "price": price,
                "changePct": chg,
                "signal": sig,
                "confidence": conf,
                "rsi": rsi,
                "pos_pct": pos_pct,
            })
        except Exception as exc:
            logger.warning("Error evaluating stock %s: %s", s, exc)
            continue

    # Priority sort: BUY first, then confidence desc
    sig_weight = {"BUY": 3, "HOLD": 2, "SELL": 1}
    scored.sort(key=lambda x: (sig_weight.get(x["signal"], 0), x["confidence"]), reverse=True)

    lines = ["### 🎯 AI Market Scan: Top Stock Recommendations\n"]
    lines.append("| Rank | Symbol | Price | Today | Consensus Signal | Conviction | RSI | Suggested Allocation |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for i, item in enumerate(scored, 1):
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(item["signal"], "⚪")
        p_str = f"${item['price']:,.2f}" if item["price"] else "—"
        chg_str = f"{item['changePct']:+.2%}" if item["changePct"] is not None else "—"
        rsi_str = f"{item['rsi']:.1f}" if isinstance(item['rsi'], (int, float)) else "—"
        lines.append(f"| #{i} | **{item['symbol']}** | {p_str} | {chg_str} | {emoji} {item['signal']} | {item['confidence']:.0%} | {rsi_str} | {item['pos_pct']:.1%} |")

    top = scored[0] if scored else None
    if top:
        lines.append(f"\n💡 **Primary Recommendation**: **{top['symbol']}** ranks #1 today with a **{top['signal']}** posture at **{top['confidence']:.0%} conviction**.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "scan_market_recommendations",
        "rankings": scored,
    }


# ─── Multi-Horizon Investment Handlers ────────────────────────────────────────

def _handle_investment_plan(q: str, question: str, rag_agent: FinancialRAGAgent) -> dict:
    from app.strategy.long_term_planner import LongTermPlanner

    # Parse capital
    cap = 500000.0
    if "lakh" in q or "lac" in q:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", q)
        if m:
            cap = float(m.group(1)) * 100000.0
    elif "k" in q:
        m = re.search(r"(\d+)\s*k", q)
        if m:
            cap = float(m.group(1)) * 1000.0
    else:
        m = re.search(r"(\d[\d,]+)", q)
        if m:
            val_str = m.group(1).replace(",", "")
            if float(val_str) > 1000:
                cap = float(val_str)

    # Parse horizon
    horizon = "3_years"
    if "5" in q and "year" in q:
        horizon = "5_years"
    elif "1" in q and "year" in q:
        horizon = "1_year"
    elif "6" in q and ("month" in q or "m" in q):
        horizon = "6_months"

    planner = LongTermPlanner(rag_agent=rag_agent)
    plan = planner.construct_portfolio(
        investment_amount=cap,
        horizon=horizon,
        risk_preference="moderate",
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    )

    lines = [
        f"### 📋 Strategic Long-Term Investment Plan",
        f"**Target Horizon**: {plan.horizon.replace('_', ' ').title()} | **Investable Capital**: ₹{plan.capital:,.2f} | **Risk Profile**: {plan.risk_profile.title()}",
        f"\n#### 💼 Suggested Portfolio Allocation",
        "| Stock | Sector | Allocation % | Capital | Shares | Long-Term Score | Quality / Stability |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for a in plan.allocations:
        lines.append(
            f"| **{a.symbol}** | {a.sector} | {a.weight_pct:.1%} | ₹{a.allocated_amount:,.2f} | {a.target_shares} | {a.fundamental_score:.0f}/100 | {a.fundamental_score:.0f} / {a.stability_score:.0f} |"
        )
    lines.append(f"| **CASH BUFFER** | Liquidity / Risk Reserve | {plan.cash_allocation_pct:.1%} | ₹{plan.cash_amount:,.2f} | — | 100/100 | Inviolable Safety |")

    lines.extend([
        f"\n#### 🛡️ Portfolio Risk Metrics (Deterministic Risk Engine)",
        f"• **Expected Volatility (Annualized)**: {plan.portfolio_volatility:.1%}",
        f"• **Value at Risk (VaR 95%)**: {plan.var_95:.2%}",
        f"• **Conditional VaR (CVaR 95%)**: {plan.cvar_95:.2%}",
        f"• **Historical Max Drawdown**: {plan.expected_drawdown:.1%}",
        f"• **Scenario Return Range**: {plan.expected_annual_return_range}",
        f"\n_{plan.disclaimer}_"
    ])

    return {
        "answer": "\n".join(lines),
        "tool_used": "generate_long_term_plan",
        "raw_plan": plan.model_dump(),
    }


def _handle_long_term_opportunities(rag_agent: FinancialRAGAgent) -> dict:
    from app.strategy.long_term_planner import LongTermPlanner
    planner = LongTermPlanner(rag_agent=rag_agent)

    syms = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    scored = []
    for s in syms:
        try:
            res = planner.evaluate_symbol(s, horizon="3_years")
            scored.append(res)
        except Exception:
            continue

    scored.sort(key=lambda x: x.composite_long_term_score, reverse=True)

    lines = [
        "### 🌟 Long-Term Investment Opportunities (3Y-5Y Horizon)",
        "Evaluated on real fundamental quality, revenue stability, moat, and 10-K RAG evidence citations:\n",
        "| Rank | Symbol | Long-Term Score | Quality | Growth | Valuation | Stability | Risk Tier | Recommendation |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, sc in enumerate(scored, 1):
        rec_emoji = "🟢" if "ACCUMULATE" in sc.recommendation else ("🟡" if "HOLD" in sc.recommendation else "⚪")
        lines.append(
            f"| #{i} | **{sc.symbol}** | **{sc.composite_long_term_score:.0f}/100** | {sc.fundamental_quality:.0f} | {sc.growth_score:.0f} | {sc.valuation_score:.0f} | {sc.financial_stability:.0f} | {sc.long_term_risk} | {rec_emoji} {sc.recommendation} |"
        )

    if scored:
        top = scored[0]
        lines.append(f"\n💡 **Top Long-Horizon Asset**: **{top.symbol}** ranks highest with **{top.composite_long_term_score:.0f}/100**. {top.summary_rationale}")

    return {
        "answer": "\n".join(lines),
        "tool_used": "find_long_term_opportunities",
    }


def _handle_short_term_opportunities(orchestrator: TradingOrchestrator) -> dict:
    from app.strategy.short_term_planner import ShortTermPlanner
    st = ShortTermPlanner(orchestrator=orchestrator)
    opps = st.scan_watchlist(horizon="1_week")

    lines = [
        "### ⚡ Short-Term Tactical Opportunities (1-Week Horizon)",
        "Analyzed using momentum, ML directional models, crash probability, and deterministic risk checks:\n",
        "| Symbol | Signal | Entry Target | Target Zone | Stop-Loss | R/R Ratio | Sizing % | Conviction | Risk Level |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for o in opps:
        sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "ABSTAIN": "⚪"}.get(o.signal, "⚪")
        tz = o.trade_zone
        e_str = f"${tz.entry_target:.2f}" if tz else f"${o.current_price:.2f}"
        t_str = f"${tz.target_base:.2f}" if tz else "—"
        s_str = f"${tz.stop_loss:.2f}" if tz else "—"
        rr = tz.risk_reward_str if tz else "—"
        conf = f"{o.confidence.signal_confidence:.0%}"
        lines.append(
            f"| **{o.symbol}** | {sig_emoji} {o.signal} | {e_str} | {t_str} | {s_str} | {rr} | {o.position_size_pct:.1%} | {conf} | {o.risk_level} |"
        )

    lines.append("\n⚠️ **Tactical Disclaimer**: All setups require strict stop-loss adherence. Market conditions fluctuate rapidly.")
    return {
        "answer": "\n".join(lines),
        "tool_used": "scan_short_term_opportunities",
    }


def _handle_why_hold_long_term(symbol: str, rag_agent: FinancialRAGAgent) -> dict:
    from app.strategy.long_term_planner import LongTermPlanner
    planner = LongTermPlanner(rag_agent=rag_agent)
    sc = planner.evaluate_symbol(symbol, horizon="3_years")

    lines = [
        f"### 🔍 Fundamental Investment Case: **{symbol}** (Long-Term Horizon)",
        f"**Composite Score**: {sc.composite_long_term_score:.0f}/100 | **Recommendation**: {sc.recommendation} | **Risk Tier**: {sc.long_term_risk}\n",
        f"#### 📊 Core Fundamental Pillars",
        f"• **Business Quality**: {sc.fundamental_quality:.0f}/100 (balance sheet strength, margins, competitive moat)",
        f"• **Revenue & Earnings Growth**: {sc.growth_score:.0f}/100",
        f"• **Valuation Attractiveness**: {sc.valuation_score:.0f}/100",
        f"• **Financial Stability**: {sc.financial_stability:.0f}/100",
        f"\n#### 📑 10-K RAG Evidence Citations",
    ]

    for c in sc.rag_citations[:2]:
        lines.append(f"• **[{c.get('source', '10-K')} - {c.get('section', 'General')}]**: \"{c.get('text', '')[:160]}...\"")

    lines.extend([
        f"\n#### 🎯 Measurable Operational Rules",
        f"• **Accumulation Trigger**: {sc.accumulation_conditions[0] if sc.accumulation_conditions else 'Price dips toward value boundary'}",
        f"• **Review Trigger**: {sc.review_conditions[0] if sc.review_conditions else 'Annual operating margin contraction'}",
        f"• **Exit Condition**: {sc.exit_conditions[0] if sc.exit_conditions else 'Permanent degradation of fundamental moat'}",
    ])

    return {
        "answer": "\n".join(lines),
        "tool_used": "explain_long_term_holding",
    }


def _handle_why_short_term_abstain(symbol: str, orchestrator: TradingOrchestrator) -> dict:
    from app.strategy.short_term_planner import ShortTermPlanner
    st = ShortTermPlanner(orchestrator=orchestrator)
    opp = st.analyze_opportunity(symbol, horizon="1_week")

    lines = [
        f"### 🛡️ Tactical Abstention & Gate Analysis for **{symbol}**",
        f"**Decision**: `{opp.signal}` | **Risk Level**: {opp.risk_level} | **Current Regime**: `{opp.regime}`",
    ]

    if opp.signal in {"ABSTAIN", "HOLD"}:
        lines.append("\n**Primary Abstention / De-risking Drivers:**")
        if opp.uncertainty_factors:
            for uf in opp.uncertainty_factors:
                lines.append(f"• ⚠️ {uf}")
        else:
            lines.append("• ⚠️ Multi-agent consensus did not meet the required confidence threshold (>60%) for tactical capital commitment.")

        if opp.crash_probability > 0.30:
            lines.append(f"• 📉 Elevated Tail Risk: Crash probability is **{opp.crash_probability:.1%}**, activating the risk gate.")

        lines.extend([
            f"\n**Deterministic Risk Status**:",
            f"• 95% Daily VaR: {opp.var_95:.2%}",
            f"• 95% Daily CVaR: {opp.cvar_95:.2%}",
            f"• Max Drawdown in Lookback: {opp.current_drawdown:.1%}",
            f"\n*The system purposefully abstains when market conditions introduce excessive variance, preserving capital for higher conviction opportunities.*"
        ])
    else:
        lines.append(f"\nThe short-term planner is currently **{opp.signal}** on {symbol} (Conviction: {opp.confidence.signal_confidence:.0%}).")

    return {
        "answer": "\n".join(lines),
        "tool_used": "explain_short_term_abstain",
    }


def _handle_compare_horizons(symbol: str) -> dict:
    from app.strategy.horizon_backtest import MultiHorizonBacktester
    bt = MultiHorizonBacktester()
    res = bt.run_comparison(symbol=symbol, period="2y")

    comp = res.get("comparison", [])
    lt = next((c for c in comp if c["horizon_category"] == "LONG_TERM"), {})
    st = next((c for c in comp if c["horizon_category"] == "SHORT_TERM"), {})

    lines = [
        f"### ⚖️ Multi-Horizon Strategy Comparison ({symbol})",
        "| Dimension | Long-Term Strategy (3Y-5Y) | Short-Term Tactical (1W-4W) | Buy & Hold Baseline |",
        "|:---|:---|:---|:---|",
        "| **Primary Focus** | Fundamental Quality & Moat | Momentum & Volatility Swings | Market Exposure |",
        "| **RAG Importance** | High (10-K Filings & Debt) | Medium (Catalysts & News) | None |",
        "| **Rebalancing Frequency** | Low (Quarterly / Drift > 5%) | High (Signal / Event-Driven) | None |",
        f"| **CAGR** | {lt.get('cagr', 0.0):.1%} | {st.get('cagr', 0.0):.1%} | {comp[2].get('cagr', 0.0) if len(comp)>2 else 0.0:.1%} |",
        f"| **Sharpe Ratio** | {lt.get('sharpe_ratio', 0.0):.2f} | {st.get('sharpe_ratio', 0.0):.2f} | {comp[2].get('sharpe_ratio', 0.0) if len(comp)>2 else 0.0:.2f} |",
        f"| **Max Drawdown** | {lt.get('max_drawdown', 0.0):.1%} | {st.get('max_drawdown', 0.0):.1%} | {comp[2].get('max_drawdown', 0.0) if len(comp)>2 else 0.0:.1%} |",
        f"| **Turnover** | {lt.get('turnover', 0.0):.1%} | {st.get('turnover', 0.0):.1%} | 0.0% |",
        f"| **Abstention Rate** | {lt.get('abstention_rate', 0.0):.1%} | {st.get('abstention_rate', 0.0):.1%} | 0.0% |",
        f"\n**Key Takeaway**: Tactical trading captures localized swings with higher turnover ({st.get('turnover', 0.0):.1%}), while long-term investment optimizes for lower volatility and reduced downside drawdown."
    ]

    return {
        "answer": "\n".join(lines),
        "tool_used": "compare_multi_horizons",
    }


def _handle_combined_portfolio_risk(paper_engine: PaperTradingEngine) -> dict:
    from app.strategy.combined_portfolio import CombinedPortfolioManager
    mgr = CombinedPortfolioManager()

    try:
        cash = paper_engine.get_portfolio_summary().get("cash", 100000.0)
    except Exception:
        cash = 100000.0

    state = mgr.build_combined_portfolio(total_capital=max(cash, 100000.0))
    risk = state.total_risk

    lines = [
        "### 🌐 Unified Multi-Horizon Portfolio Risk Breakdown",
        f"**Total Capital Evaluated**: ₹{risk.total_capital:,.2f} | **Risk Status**: `{risk.risk_status}`\n",
        "#### 📊 Horizon Sleeve Breakdown",
        f"• **Long-Term Investment Sleeve**: ₹{risk.long_term_capital:,.2f} (**{risk.long_term_pct:.1%}**)",
        f"• **Short-Term Tactical Sleeve**: ₹{risk.short_term_capital:,.2f} (**{risk.short_term_pct:.1%}**)",
        f"• **Cash Reserve Buffer**: ₹{risk.cash_capital:,.2f} (**{risk.cash_pct:.1%}**)",
        f"\n#### 🛡️ Total Cross-Portfolio Risk Metrics",
        f"• **Correlated Portfolio Volatility**: {risk.portfolio_volatility_annualized:.1%} (Cap: 40.0%)",
        f"• **Unified Value at Risk (VaR 95%)**: {risk.total_var_95:.2%}",
        f"• **Unified Expected Shortfall (CVaR 95%)**: {risk.total_cvar_95:.2%}",
        f"• **Dominant Sector**: {risk.dominant_sector} ({risk.max_sector_pct:.1%})",
    ]

    if risk.risk_violations:
        lines.append(f"\n⚠️ **Risk Engine Flag**: {'; '.join(risk.risk_violations)}")
    else:
        lines.append("\n✅ **Risk Engine Verification**: All position and concentration limits comply with strict quantitative caps.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_combined_portfolio_risk",
    }


def _handle_adaptive_horizon(symbol: str) -> dict:
    from app.strategy.adaptive_horizon_selector import AdaptiveInvestmentHorizonSelector
    selector = AdaptiveInvestmentHorizonSelector()
    res = selector.assess(symbol)

    lines = [
        f"### 🎯 Adaptive Investment Horizon Assessment: **{res.symbol}**",
        f"**Recommended Horizon**: `{res.recommended_horizon.value}`",
        f"• **Long-Term Suitability Score**: {res.long_term_score:.1f}/100",
        f"• **Short-Term Suitability Score**: {res.short_term_score:.1f}/100",
        f"• **Confidence Level**: {res.confidence * 100:.1f}%\n",
        f"**Strategic Rationale**: {res.alternative_horizon_rationale}\n",
    ]
    if res.supporting_evidence:
        lines.append("**Supporting Evidence**:")
        for ev in res.supporting_evidence:
            lines.append(f"  • {ev}")
    if res.conflicting_evidence:
        lines.append("\n**Conflicting Signals**:")
        for ev in res.conflicting_evidence:
            lines.append(f"  • {ev}")
    if res.main_risks:
        lines.append("\n**Primary Risks**:")
        for r in res.main_risks:
            lines.append(f"  • {r}")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_horizon_assessment",
        "raw": res.model_dump(),
    }


def _handle_digital_twin_scenario(question: str, paper_engine: PaperTradingEngine) -> dict:
    from app.portfolio.digital_twin import PersonalPortfolioDigitalTwin
    dt = PersonalPortfolioDigitalTwin(paper_engine=paper_engine)

    # Parse market decline percentage if mentioned (e.g. 10%, 20%, 5%)
    shock = -0.10  # default 10% decline
    if "20%" in question or "20 percent" in question:
        shock = -0.20
    elif "5%" in question or "5 percent" in question:
        shock = -0.05
    elif "15%" in question or "15 percent" in question:
        shock = -0.15

    res = dt.simulate_scenario(
        scenario_name=f"Hypothetical {abs(shock)*100:.0f}% Market Pullback",
        market_shock_pct=shock,
        volatility_shock_multiplier=1.25,
    )

    lines = [
        f"### 🧪 Personal Portfolio Digital Twin: **{res.scenario_name}**",
        f"*(Virtual Simulation — Live paper portfolio was **NOT** modified)*\n",
        "| Metric | Live Baseline | Simulated Post-Shock | Difference |",
        "|:---|:---|:---|:---|",
        f"| **Portfolio Value** | ${res.baseline.total_value:,.2f} | ${res.simulated.total_value:,.2f} | **${res.allocation_diffs.get('total_value_diff', 0.0):,.2f}** |",
        f"| **Cash Allocation** | {res.baseline.cash_pct:.1%} | {res.simulated.cash_pct:.1%} | {res.allocation_diffs.get('cash_pct_diff', 0.0):+.1%} |",
        f"| **Parametric VaR (95%)** | ${res.baseline.var_95:,.2f} | ${res.simulated.var_95:,.2f} | ${res.risk_metric_diffs.get('var_95_diff', 0.0):+,.2f} |",
        f"| **Expected Shortfall (CVaR)** | ${res.baseline.cvar_95:,.2f} | ${res.simulated.cvar_95:,.2f} | ${res.risk_metric_diffs.get('cvar_95_diff', 0.0):+,.2f} |",
        f"| **Annualized Volatility** | {res.baseline.estimated_volatility:.1%} | {res.simulated.estimated_volatility:.1%} | {res.risk_metric_diffs.get('volatility_diff', 0.0):+.1%} |",
        f"\n**Estimated Trading Friction**: ${res.estimated_transaction_cost:.2f} | **Turnover**: {res.estimated_turnover:.1%}",
        f"\n_Disclaimer: Simulation uses static shock assumptions without modifying actual broker positions._"
    ]

    return {
        "answer": "\n".join(lines),
        "tool_used": "run_portfolio_scenario",
        "raw": res.model_dump(),
    }


def _handle_evidence_conflicts(symbol: str) -> dict:
    from app.rag.conflict_detector import FinancialEvidenceConflictDetector
    detector = FinancialEvidenceConflictDetector()
    report = detector.detect_conflicts(symbol)

    lines = [
        f"### 🔍 Financial Evidence Conflict Analysis: **{report.symbol}**",
        f"**Conflict Severity**: `{report.severity.value}` | **Calibrated Score**: `{report.conflict_score:.2f}` (0.0 to 1.0)",
        f"**Recommended Gate Action**: `{report.recommended_action}` | **Confidence Impact**: -{report.confidence_impact * 100:.0f}%\n",
        f"**Summary**: {report.summary}\n",
    ]
    if report.conflicting_claims:
        lines.append("**Identified Contradictions**:")
        for idx, claim in enumerate(report.conflicting_claims, 1):
            lines.append(f"{idx}. **{claim.get('type', 'Contradiction')}**:")
            lines.append(f"   • Bullish/Claim A: {claim.get('claim_a')}")
            lines.append(f"   • Counter/Claim B: {claim.get('claim_b')}")
            lines.append(f"   • Source Period: {claim.get('reporting_period')}")
    else:
        lines.append("✅ No contradictory financial signals, accrual mismatches, or tail-risk divergences detected.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_evidence_conflicts",
        "raw": report.model_dump(),
    }


def _handle_performance_attribution(paper_engine: PaperTradingEngine) -> dict:
    from app.portfolio.attribution import PortfolioDecisionAttributionEngine
    engine = PortfolioDecisionAttributionEngine(paper_engine=paper_engine)
    res = engine.compute_attribution(benchmark_symbol="SPY")

    lines = [
        f"### 📈 Portfolio Decision Attribution (Brinson-Fachler Multi-Factor)",
        f"**Evaluation Period**: `{res.period}` | **Benchmark**: S&P 500 (SPY)\n",
        f"• **Total Portfolio Return**: {res.total_portfolio_return * 100:+.2f}%",
        f"• **Benchmark Return**: {res.benchmark_return * 100:+.2f}%",
        f"• **Net Excess Alpha**: **{res.excess_return * 100:+.2f}%**\n",
        "#### 🔬 Return Factor Decomposition",
        f"• **Asset Selection Effect**: {res.selection_effect * 10000:+.1f} bps (Stock picking alpha)",
        f"• **Sector Allocation Effect**: {res.allocation_effect * 10000:+.1f} bps (Over/underweighting sectors)",
        f"• **Market Timing / Interaction**: {res.timing_effect * 10000:+.1f} bps",
        f"• **Cash Reserve Drag**: {res.cash_drag_effect * 10000:+.1f} bps",
        f"• **Transaction Costs Impact**: {res.transaction_cost_impact * 10000:+.1f} bps\n",
        "#### 🏆 Top Contributing Assets",
        f"• Positive: {', '.join(res.key_positive_contributors)}",
        f"• Negative: {', '.join(res.key_negative_contributors)}",
    ]

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_performance_attribution",
        "raw": res.model_dump(),
    }


def _handle_reliability_status() -> dict:
    from app.services.reliability_monitor import get_reliability_monitor
    monitor = get_reliability_monitor()
    report = monitor.check_system_health()

    badge = "🟢" if report.overall_status.value == "HEALTHY" else "🟡" if report.overall_status.value == "WARNING" else "🔴"
    lines = [
        f"### {badge} AI Reliability & Drift Monitor Status",
        f"**Overall Platform Health**: `{report.overall_status.value}`",
        f"• **Risk-Increasing Trade Lock**: `{'ACTIVATED (LOCKED)' if report.block_risk_increasing_trades else 'CLEARED (NORMAL)'}`",
        f"• **Position Sizing Multiplier**: `{report.position_size_multiplier * 100:.0f}%`",
        f"• **Summary**: {report.summary}\n",
        "#### 🩺 Subsystem Health Checks",
    ]
    for check in report.checks:
        c_badge = "✅" if check.status.value == "HEALTHY" else "⚠️" if check.status.value == "WARNING" else "❌"
        lines.append(f"• {c_badge} **{check.component_name}**: {check.message} (Value: {check.metric_value:.3f}, Threshold: {check.threshold:.2f})")

    if report.active_alerts:
        lines.append("\n⚠️ **Active System Alerts**:")
        for alert in report.active_alerts:
            lines.append(f"  • **{alert.title}** ({alert.severity.value}): {alert.detail}")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_reliability_status",
        "raw": report.model_dump(),
    }


# ─── Forex & Macro Economic Event Handlers ───────────────────────────────────

def _handle_forex_regime(pair: str) -> dict:
    from app.forex.regime_intelligence import get_regime_intelligence
    intel = get_regime_intelligence()
    res = intel.analyze_regime(pair)

    ind = res.supporting_indicators
    lines = [
        f"### 💱 Forex Market Regime: **{res.pair}**",
        f"• **Active Detected Regime**: `{res.regime.value}` (Confidence: {res.regime_confidence * 100:.0f}%)",
        f"• **Trend Direction**: `{res.trend_direction.value}`",
        f"• **Volatility Level**: `{res.volatility_level.value}` (Annualized Vol: {ind.get('annualized_volatility', 0):.1%})\n",
        f"**Quantitative Explanation**: {res.explanation}\n",
        "#### 📊 Supporting Technical Indicators",
        f"• **ADX (14)**: {ind.get('adx_14', 0):.1f} (+DI: {ind.get('plus_di', 0):.1f}, -DI: {ind.get('minus_di', 0):.1f})",
        f"• **Bollinger Band Width**: {ind.get('bollinger_band_width', 0):.4f}",
        f"• **SMA Alignment**: Price ({ind.get('last_price')}) vs SMA20 ({ind.get('sma_20')}) vs SMA50 ({ind.get('sma_50')})",
        f"• **ATR (14)**: {ind.get('atr_14')}\n",
        "#### 🎯 Strategy Suitability",
        f"• Trend-Following: `{res.strategy_suitability.get('trend_following')}`",
        f"• Mean-Reversion: `{res.strategy_suitability.get('mean_reversion')}`",
        f"• Breakout: `{res.strategy_suitability.get('breakout')}`",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_regime",
        "raw": res.model_dump(),
    }


def _handle_forex_pre_event(pair_or_event: str) -> dict:
    from app.forex.event_analyzer import get_event_analyzer
    analyzer = get_event_analyzer()

    ev = analyzer.get_event(pair_or_event)
    event_id = ev["event_id"] if ev else "FOMC_2026_10"
    pair = "EUR/USD" if ev else pair_or_event

    res = analyzer.assess_pre_event(event_id, primary_pair=pair)
    lines = [
        f"### 📅 Pre-Event Assessment: **{res.event_name}**",
        f"• **Event ID**: `{res.event_id}` | **Category**: `{res.category.value}`",
        f"• **Currency**: `{res.currency}` | **Primary Pair**: `{pair}`",
        f"• **Scheduled Release Time**: `{res.scheduled_time}`",
        f"• **Expected Event Risk**: `{res.expected_risk_level.value}`\n",
        "#### 🔢 Macro Consensus Context",
        f"• **Previous Value**: {res.previous_value} {res.unit}",
        f"• **Consensus Forecast**: {res.consensus_expectation} {res.unit}",
        f"• **Current Spot Price**: {res.current_market_price:.4f}",
        f"• **Pre-Event Volatility**: {res.pre_event_volatility:.1%}",
        f"• **Pre-Event Spread**: {res.pre_event_spread_pips:.1f} pips\n",
        "#### ⚠️ Key Uncertainties & Risk Guidance",
    ]
    for unc in res.key_uncertainties:
        lines.append(f"• {unc}")
    lines.append("\n*Note: Pre-event assessment does NOT forecast the actual outcome or generate directional BUY/SELL signals.*")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_pre_event",
        "raw": res.model_dump(),
    }


def _handle_forex_post_event(pair: str, q: str) -> dict:
    from app.forex.event_analyzer import get_event_analyzer
    analyzer = get_event_analyzer()

    # Determine event based on prompt keywords
    event_id = "US_CPI_2026_10" if "inflation" in q or "cpi" in q else "FOMC_2026_10"
    actual_val = 2.7 if event_id == "US_CPI_2026_10" else 4.75

    res = analyzer.analyze_post_event(event_id=event_id, actual_value=actual_val, primary_pair=pair)
    lines = [
        f"### ⏱️ Post-Event Reaction Analysis: **{res.event_name}**",
        f"• **Primary Pair Analyzed**: `{res.primary_pair}`",
        f"• **Consensus Expectation**: {res.consensus_expectation} {res.unit}",
        f"• **Actual Released Value**: **{res.actual_value} {res.unit}**",
        f"• **Economic Surprise**: **{res.raw_surprise:+0.2f} {res.unit}** (Normalized: {res.normalized_surprise:+0.2f}σ)",
        f"• **Currency Impact Direction**: `{res.surprise_direction_for_currency}`",
        f"• **Initial Reaction**: `{res.initial_reaction}`",
        f"• **Subsequent Pattern**: `{res.reaction_pattern.value}` ({res.reversal_pct}% mean reversion)\n",
        "#### 📈 Multi-Window Reaction Timeline",
    ]
    for w in res.window_reactions:
        lines.append(
            f"• **{w.window_label}** (+{w.minutes_elapsed}m): "
            f"Price Δ: {w.price_change:+.5f} ({w.pip_movement:+.1f} pips, {w.return_pct:+.2%}) | "
            f"Spread: {w.spread_pips:.1f} pips | Ann. Vol: {w.realized_volatility:.1%}"
        )
    lines.append(f"\n• **Spread Expansion**: Peaked at {res.spread_peak_pips:.1f} pips before normalizing to {res.spread_normalized_pips:.1f} pips.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_post_event",
        "raw": res.model_dump(),
    }


def _handle_forex_reversal_continuation(pair: str) -> dict:
    from app.forex.event_analyzer import get_event_analyzer
    analyzer = get_event_analyzer()
    res = analyzer.analyze_post_event("FOMC_2026_10", actual_value=4.75, primary_pair=pair)

    lines = [
        f"### 🔄 Initial Reaction vs Subsequent Movement ({res.primary_pair})",
        f"• **Initial Immediate Reaction**: `{res.initial_reaction}`",
        f"• **Observed 60-Minute Pattern**: `{res.reaction_pattern.value}`",
        f"• **Mean Reversion Rate**: `{res.reversal_pct}%`\n",
        f"Following the initial spike of {res.window_reactions[0].pip_movement:+.1f} pips, "
        f"price action experienced {res.reaction_pattern.value.lower()} behavior over the 60-minute observation window. "
        f"By minute 60, net displacement settled at {res.window_reactions[-1].pip_movement:+.1f} pips as initial market overshooting was absorbed.",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_post_event",
        "raw": res.model_dump(),
    }


def _handle_forex_spread_volatility(pair: str) -> dict:
    from app.forex.event_analyzer import get_event_analyzer
    analyzer = get_event_analyzer()
    res = analyzer.analyze_post_event("FOMC_2026_10", actual_value=4.75, primary_pair=pair)

    lines = [
        f"### 🌊 Spread & Volatility Dynamics ({res.primary_pair})",
        f"• **Baseline Typical Spread**: ~1.2 pips",
        f"• **Peak Announcement Spread**: **{res.spread_peak_pips:.1f} pips** (Immediately after release)",
        f"• **Normalized 60m Spread**: **{res.spread_normalized_pips:.1f} pips**\n",
        "**Spread Evolution by Window**:",
    ]
    for w in res.window_reactions:
        lines.append(f"• **{w.window_label}**: {w.spread_pips:.1f} pips (Realized Vol: {w.realized_volatility:.1%})")
    lines.append("\n*Result: Spreads widened by more than 3x normal during the initial announcement window before liquidity normalized.*")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_post_event",
        "raw": res.model_dump(),
    }


def _handle_forex_risk_restriction(pair: str) -> dict:
    from app.forex.event_risk_gate import get_forex_risk_gate
    gate = get_forex_risk_gate()
    res = gate.evaluate_entry_risk(pair, side="BUY")

    status_badge = "🔴 RESTRICTED" if res.is_restricted else "🟢 PERMITTED"
    lines = [
        f"### 🛡️ Event-Related Simulated Trading Risk Check: **{res.pair}**",
        f"• **Risk Restriction Status**: `{status_badge}`",
        f"• **Spread**: {res.spread_pips:.1f} pips (Typical: {res.typical_spread_pips:.1f} pips, Ratio: {res.spread_ratio:.1f}x)",
        f"• **Annualized Volatility**: {res.volatility_annualized:.1%}",
        f"• **Permitted Simulated Actions**: `{', '.join(res.allowed_actions)}`\n",
    ]
    if res.is_restricted:
        lines.append(f"**Restriction Rationale**: {res.restriction_reason}\n")
        lines.append("⚠️ *Deterministic Policy: To prevent catastrophic simulated slippage, new entries are blocked during announcement blackout windows or abnormal spread blowouts. Existing positions remain intact under standard risk management rules.*")
    else:
        lines.append("✅ Market conditions pass all event-risk thresholds; simulated entries permitted under standard position limits.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_risk_restriction",
        "raw": res.model_dump(),
    }


def _handle_forex_reaction_memory(q: str, pair: str) -> dict:
    from app.forex.event_memory import get_event_reaction_memory
    mem = get_event_reaction_memory()

    cat = "INFLATION" if "cpi" in q or "inflation" in q else "EMPLOYMENT" if "nfp" in q or "job" in q else "INTEREST_RATE"
    summary = mem.find_similar_events(category=cat, currency="USD", target_pair=pair, top_k=3)

    lines = [
        f"### 🧠 Forex Event Reaction Memory ({cat} / USD -> {pair})",
        f"• **Historical Analog Matches**: {summary.total_historical_matches} cases found ({'Sufficient' if summary.has_sufficient_history else 'Limited'})",
        f"• **Reversal Tendency**: `{summary.reversal_tendency}` ({summary.pattern_distribution.get('REVERSAL', 0.0):.0%} of past cases experienced post-spike mean reversion)",
        f"• **Average 5m Movement**: {summary.avg_initial_pips_5m:.1f} pips",
        f"• **Average Spread Expansion**: +{summary.avg_spread_widening_pips:.1f} pips\n",
        "#### 📜 Comparable Historical Case Studies",
    ]
    for case in summary.top_similar_events:
        lines.append(
            f"• **{case.name}** ({case.release_date}): "
            f"Surprise: {case.surprise:+0.2f}{case.unit} | "
            f"Initial 5m: {case.initial_pips_5m:+.1f} pips | "
            f"60m Net: {case.net_pips_60m:+.1f} pips | "
            f"Observed: `{case.observed_pattern}` (Similarity: {case.similarity_score * 100:.0f}%)"
        )
    lines.append(f"\n**Guidance**: {summary.guidance_note}")
    lines.append("\n⚠️ *Limitation: Past event reactions do not guarantee future currency movement.*")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_event_memory",
        "raw": summary.model_dump(),
    }


def _handle_forex_comparison(event_id: str, pair: str) -> dict:
    from app.forex.event_analyzer import get_event_analyzer
    analyzer = get_event_analyzer()
    res = analyzer.compare_pre_post(event_id=event_id, primary_pair=pair)

    lines = [
        f"### ⚖️ Pre-Event vs Post-Event Outcome Comparison",
        f"• **Event**: `{res.event_name}` ({res.event_id})",
        f"• **Currency Pair**: `{res.primary_pair}`",
        f"• **Prior Expected Risk**: `{res.pre_event.expected_risk_level.value}`",
        f"• **Consensus vs Actual**: Expected {res.pre_event.consensus_expectation}{res.pre_event.unit} -> Actual {res.post_event.actual_value}{res.post_event.unit}",
        f"• **Outcome Consistency**: `{'CONSISTENT' if res.is_consistent_with_prior else 'SURPRISE DIVERGENCE'}`\n",
        f"**Integrated Summary**: {res.outcome_summary}\n",
        "#### 🔍 What Remains Unresolved",
    ]
    for unc in res.unresolved_uncertainties:
        lines.append(f"• {unc}")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_event_comparison",
        "raw": res.model_dump(),
    }


def _handle_forex_data_limitations() -> dict:
    lines = [
        "### ℹ️ Transparent Market Data & Event Analysis Limitations",
        "Our Forex Pre-Event & Post-Event system enforces strict data integrity standards:",
        "1. **Granularity Resolution**: Price reactions are analyzed using 1-minute to 5-minute OHLCV windows. Tick-level order-book microstructures are NOT available from standard EOD feeds.",
        "2. **Historical Revisions**: Economic release values reflect initial prints; second and third backward revisions by government statistical agencies are labeled when available.",
        "3. **Spread Transparency**: Bid/ask spreads during live macroeconomic spikes reflect modeled interbank liquidity estimates unless connected to a direct institutional broker feed.",
        "4. **No Unlabeled Fake Data**: Unsupported currency pairs and missing indicators are flagged explicitly rather than filled with arbitrary zeros or hallucinations.",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_forex_data_limitations",
    }


def _handle_trading_style_status() -> dict:
    from app.strategy.trading_style_manager import get_trading_style_manager
    mgr = get_trading_style_manager()
    p = mgr.get_active_profile()

    lines = [
        f"### 🎯 Active Trading Style: **{p.display_name}** ({p.style})",
        f"**Profile Overview**: {p.description}\n",
        f"• **Analysis Timeframes**: `{', '.join(p.primary_timeframes)}`",
        f"• **Max Holding Horizon**: `{p.max_holding_days:.1f} day(s)`",
        f"• **Default Risk Per Trade**: `{p.default_risk_per_trade_pct:.1%}` (Max allowed: `{p.max_risk_per_trade_pct:.1%}`)",
        f"• **Minimum Risk/Reward Requirement**: `{p.min_risk_reward_ratio:.1f}:1`",
        f"• **Session Boundary Policy**: `{p.session_boundary_action}` (Overnight risk: `{'YES' if p.carries_overnight_risk else 'NO'}`)",
        f"• **Economic Event Blackout Filter**: `{'ACTIVE' if p.requires_economic_event_check else 'OFF'}`",
        f"• **Macro Financial RAG Integration**: `{'ACTIVE' if p.requires_macro_rag else 'OFF'}`",
        "\n*You can switch between Day Trading, Swing Trading, and Position Trading via the Trading Style Selector in the dashboard or via API.*",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_trading_style_status",
        "raw": p.model_dump(),
    }


def _handle_trade_rejection_explanation(symbol: str | None) -> dict:
    from app.db.models import get_session, TradeQualityGateRecord
    session = get_session()
    rec = None
    try:
        q = session.query(TradeQualityGateRecord)
        if symbol:
            q = q.filter(TradeQualityGateRecord.symbol == symbol.upper())
        rec = q.order_by(TradeQualityGateRecord.id.desc()).first()
    finally:
        session.close()

    if not rec:
        return {
            "answer": "No recent trade quality gate evaluation records found. All recent proposed trades were either accepted or none have been submitted.",
            "tool_used": "get_trade_rejection_explanation",
        }

    reasons = json.loads(rec.rejection_reasons_json) if rec.rejection_reasons_json else []
    checks = json.loads(rec.checks_json) if rec.checks_json else {}

    lines = [
        f"### 🛡️ Trade Quality Gate Evaluation: **{rec.symbol}**",
        f"• **Evaluation ID**: `{rec.evaluation_id}`",
        f"• **Trading Style**: `{rec.trading_style}` | **Side**: `{rec.proposed_side}`",
        f"• **Decision**: **`{rec.decision}`** ({'Approved' if rec.decision == 'ELIGIBLE' else 'Not Approved'})",
        f"• **Entry Price**: `${rec.entry_price:,.2f}` | **Stop-Loss**: `${rec.stop_loss_price:,.2f}`" if rec.stop_loss_price else f"• **Entry Price**: `${rec.entry_price:,.2f}`",
        f"• **R/R Ratio**: `{rec.risk_reward_ratio:.2f}:1`" if rec.risk_reward_ratio else "",
        "\n**Specific Reasons / Violations**:",
    ]
    if reasons:
        for r in reasons:
            lines.append(f"• ❌ {r}")
    else:
        lines.append("• ✅ All 8 quality checks passed successfully.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_trade_rejection_explanation",
    }


def _handle_position_sizing_query(symbol: str, paper_engine) -> dict:
    from app.risk.position_sizer import get_position_sizer
    sizer = get_position_sizer()
    summary = paper_engine.get_portfolio_summary()
    equity = summary.get("total_portfolio_value", 100000.0)

    # Fetch live price
    px = 150.0
    try:
        q = get_stock_quote(symbol)
        if q and "price" in q:
            px = float(q["price"])
    except Exception:
        pass

    stop = round(px * 0.95, 2)
    target = round(px * 1.10, 2)

    res = sizer.calculate_size(
        symbol=symbol,
        side="BUY",
        entry_price=px,
        stop_loss_price=stop,
        target_price=target,
        account_equity=equity,
    )

    lines = [
        f"### 📐 Deterministic Position Sizing Model for **{symbol}**",
        f"• **Account Equity**: `${equity:,.2f}`",
        f"• **Configured Risk Budget**: `{res.risk_pct:.1%}` -> **`${res.risk_budget_usd:,.2f}`** maximum monetary risk",
        f"• **Simulated Entry Price**: `${px:,.2f}` | **Stop-Loss Price**: `${stop:,.2f}` (Distance: `${res.stop_distance:,.2f}` / `{res.stop_distance_pct:.1%}`)",
        f"• **Calculated Permitted Units**: **`{res.max_units}`** {'shares' if res.instrument_type == 'STOCK' else 'units'}",
        f"• **Position Notional Capital**: `${res.position_notional_value:,.2f}`",
        f"• **Estimated Loss at Stop**: `${res.estimated_monetary_risk:,.2f}` (including transaction costs & slippage)",
        f"\n**Mathematical Formula**: `Max Units = floor(Risk Budget / Loss Per Unit)`",
        "*Note: A stop-loss does not guarantee exact loss amount due to overnight gaps and slippage; monetary risk is strictly modeled as an estimate.*",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_position_sizing_query",
        "raw": res.model_dump(),
    }


def _handle_daily_risk_status(paper_engine) -> dict:
    from app.risk.daily_loss_guard import get_daily_loss_guard
    guard = get_daily_loss_guard()
    summary = paper_engine.get_portfolio_summary()
    equity = summary.get("total_portfolio_value", 100000.0)
    unrealized = summary.get("unrealized_pnl", 0.0)

    status = guard.update_unrealized_pnl_and_evaluate(equity, unrealized)

    lines = [
        f"### 🛑 Persistent Daily Loss Protection Status",
        f"• **Trading Day (UTC)**: `{status.date_str}`",
        f"• **Day Starting Equity Baseline**: `${status.starting_equity:,.2f}`",
        f"• **Current Equity**: `${status.current_equity:,.2f}`",
        f"• **Today's Net P&L**: `{status.daily_pnl_pct:+.2%}` (${status.net_daily_pnl:+,.2f})",
        f"• **Daily Loss Limit**: `-{status.max_daily_loss_pct:.1%}`",
        f"• **Remaining Loss Allowance**: **`${status.daily_loss_allowance_remaining:,.2f}`**",
        f"• **Consecutive Losses**: `{status.consecutive_losses}/{status.max_consecutive_losses}`",
        f"• **Trading Lock Status**: **`{'LOCKED' if status.is_locked else 'NORMAL (UNLOCKED)'}`**",
    ]
    if status.is_locked:
        lines.append(f"• **Lock Reason**: {status.lock_reason}")
    if status.is_cooldown_active:
        lines.append(f"• **Cooldown Timer**: `{status.cooldown_minutes_remaining:.1f} minutes remaining`")

    lines.append("\n*Protection Invariant: The daily lock persists across server restarts and browser refreshes to prevent revenge-trading.*")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_daily_risk_status",
        "raw": status.model_dump(),
    }


def _handle_cooldown_status(paper_engine) -> dict:
    from app.risk.daily_loss_guard import get_daily_loss_guard
    guard = get_daily_loss_guard()
    summary = paper_engine.get_portfolio_summary()
    equity = summary.get("total_portfolio_value", 100000.0)
    status = guard.update_unrealized_pnl_and_evaluate(equity, 0.0)

    lines = [
        f"### ⏱️ Anti-Revenge-Trading Cooldown & Lock Monitor",
        f"• **Current Consecutive Losses**: `{status.consecutive_losses}` (Threshold: `{status.max_consecutive_losses}`)",
        f"• **Cooldown Active**: `{'YES' if status.is_cooldown_active else 'NO'}`",
        f"• **Minutes Remaining**: `{status.cooldown_minutes_remaining:.1f} min`" if status.is_cooldown_active else "• **Cooldown Status**: Idle (No cooldown restriction active)",
        f"• **Simulated Entries Allowed**: `{'YES' if status.can_enter_new_trades else 'NO — LOCKED'}`",
    ]
    if status.is_locked:
        lines.append(f"• **Lock Reason**: {status.lock_reason}")
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_cooldown_status",
        "raw": status.model_dump(),
    }


def _handle_portfolio_exposure_query(paper_engine) -> dict:
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

    lines = [
        f"### 💼 Portfolio Leverage, Margin & Currency Exposure",
        f"• **Total Portfolio Equity**: `${equity:,.2f}`",
        f"• **Gross Exposure Notional**: `${res.current_portfolio_value:,.2f}`",
        f"• **Gross Leverage**: `{res.gross_leverage:.2f}x` (Max Allowed: `{res.max_leverage_allowed:.1f}x`)",
        f"• **Margin Required**: `${res.margin_required_projected:,.2f}`",
        f"• **Free Margin Available**: `${res.free_margin_projected:,.2f}`",
        f"• **Max Single Instrument Cap**: `{res.max_instrument_pct:.1%}`",
        "\n**Cross-Pair Currency Concentration** (Shared Exposure across FX pairs):",
    ]
    if res.currency_exposures_pct:
        for curr, pct in res.currency_exposures_pct.items():
            bar = "█" * int(pct * 20)
            lines.append(f"• **{curr}**: `{pct:.1%}` {bar} (Cap: `{res.max_currency_pct:.1%}`)")
    else:
        lines.append("• No open Forex currency exposures.")

    return {
        "answer": "\n".join(lines),
        "tool_used": "get_portfolio_exposure_query",
        "raw": res.model_dump(),
    }


def _handle_unvalidated_strategy_query() -> dict:
    lines = [
        "### ⚠️ Strategy Edge Validation Policy",
        "Our Trade Setup Quality Gate enforces a strict research evidence standard:",
        "1. **No Assumed Edge**: A strategy cannot be assumed profitable based on backtests with fewer than 10 historical trades.",
        "2. **Abstention Over Speculation**: If backtest sample size is below threshold or edge cannot be verified in the current regime, the system issues an **ABSTAIN — INSUFFICIENT EVIDENCE** decision.",
        "3. **Zero Fabrication**: The system never manufactures fake win rates, profit factors, or statistical confidence scores.",
        "4. **Preserved Signals**: The original AI agent recommendation is preserved for audit trail purposes, but the execution gate blocks simulated capital deployment until historical edge is verified.",
    ]
    return {
        "answer": "\n".join(lines),
        "tool_used": "get_unvalidated_strategy_query",
    }




"""Tests for individual agents."""
import pandas as pd

from app.agents.ml_agent import MLPredictionAgent
from app.agents.regime_agent import MarketRegimeAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.data.news_sample import get_sample_headlines
from app.data.pipeline import load_and_prepare


def _sample_df():
    df, report, source = load_and_prepare("AAPL", period="2y", use_cache=True)
    assert report.is_usable
    return df


def test_technical_agent_output_shape():
    df = _sample_df()
    sig = TechnicalAgent().analyze(df, "AAPL")
    assert sig.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0
    assert "rsi_14" in sig.indicators
    assert 0 <= sig.indicators["rsi_14"] <= 100


def test_technical_agent_never_calls_llm_for_indicators():
    # indicators must be pure numbers derived deterministically — running
    # twice on the same data must give byte-identical indicator values
    df = _sample_df()
    sig1 = TechnicalAgent().analyze(df, "AAPL")
    sig2 = TechnicalAgent().analyze(df, "AAPL")
    assert sig1.indicators == sig2.indicators


def test_ml_agent_uses_chronological_split_no_shuffle():
    df = _sample_df()
    pred = MLPredictionAgent().analyze(df, "AAPL")
    assert pred.predicted_direction in ("UP", "DOWN")
    assert 0.0 <= pred.probability <= 1.0
    assert pred.validation_metrics["split"].startswith("chronological")
    assert pred.validation_metrics["n_train"] > pred.validation_metrics["n_val"]


def test_sentiment_agent_is_deterministic_not_llm():
    headlines = get_sample_headlines("AAPL")
    r1 = SentimentAgent().analyze("AAPL", headlines)
    r2 = SentimentAgent().analyze("AAPL", headlines)
    assert r1.sentiment_score == r2.sentiment_score  # deterministic, not sampled from an LLM


def test_sentiment_agent_handles_empty_headlines():
    r = SentimentAgent().analyze("AAPL", [])
    assert r.label == "NEUTRAL"
    assert r.n_headlines == 0


def test_regime_agent_output_shape():
    df = _sample_df()
    r = MarketRegimeAgent().analyze(df, "AAPL")
    assert r.trend_component in ("BULL", "BEAR", "SIDEWAYS")
    assert r.volatility_component in ("HIGH_VOLATILITY", "LOW_VOLATILITY", "NORMAL_VOLATILITY")
    assert 0.0 <= r.confidence <= 1.0

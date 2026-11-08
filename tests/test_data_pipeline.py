"""Tests for the market data provider + validation/cleaning pipeline."""
import numpy as np
import pandas as pd

from app.data.pipeline import validate_and_clean, engineer_features, load_and_prepare
from app.data.provider import SyntheticProvider


def test_synthetic_provider_returns_valid_ohlcv():
    df = SyntheticProvider().get_historical("AAPL", period="1y", interval="1d")
    assert not df.empty
    assert set(["open", "high", "low", "close", "adj_close", "volume"]).issubset(df.columns)
    assert (df["high"] >= df["low"]).all()


def test_synthetic_provider_is_reproducible():
    df1 = SyntheticProvider().get_historical("AAPL", period="6mo")
    df2 = SyntheticProvider().get_historical("AAPL", period="6mo")
    pd.testing.assert_frame_equal(df1, df2)


def test_validate_and_clean_removes_duplicates():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": np.arange(10) + 100.0, "high": np.arange(10) + 101.0,
        "low": np.arange(10) + 99.0, "close": np.arange(10) + 100.5,
        "adj_close": np.arange(10) + 100.5, "volume": np.arange(10) * 1000 + 1000,
    }, index=idx)
    df = pd.concat([df, df.iloc[[0]]])  # inject a duplicate
    clean, report = validate_and_clean(df, "TEST")
    assert report.duplicates_removed == 1
    assert len(clean) == 10


def test_validate_and_clean_drops_non_positive_prices():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 101, -5, 103, 104], "high": [101, 102, 1, 104, 105],
        "low": [99, 100, -6, 102, 103], "close": [100.5, 101.5, 0, 103.5, 104.5],
        "adj_close": [100.5, 101.5, 0, 103.5, 104.5], "volume": [1000] * 5,
    }, index=idx)
    clean, report = validate_and_clean(df, "TEST")
    assert report.non_positive_prices_dropped == 1
    assert len(clean) == 4


def test_engineer_features_adds_expected_columns():
    df = SyntheticProvider().get_historical("MSFT", period="1y")
    feat = engineer_features(df)
    for col in ["return_1d", "log_return_1d", "rolling_vol_20d", "dollar_volume"]:
        assert col in feat.columns


def test_load_and_prepare_falls_back_gracefully():
    # yfinance is unreachable in the test sandbox network, so this exercises
    # the full fallback -> validate -> feature-engineer -> cache path
    df, report, source = load_and_prepare("AAPL", period="1y", use_cache=False)
    assert report.is_usable
    assert source in ("yfinance", "synthetic", "cache")
    assert len(df) > 30

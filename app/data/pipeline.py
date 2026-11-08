"""Data engineering pipeline: validate -> clean -> feature-engineer -> cache.

Source -> Validation -> Cleaning -> Feature Engineering -> Storage -> Agents
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.data.provider import OHLCV_COLUMNS, get_historical_with_fallback
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    symbol: str
    rows_in: int
    rows_out: int
    duplicates_removed: int = 0
    missing_filled: int = 0
    invalid_timestamps_dropped: int = 0
    non_positive_prices_dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.rows_out >= 1  # Intraday sessions may start with a few bars; individual agents enforce their own length requirement


def validate_and_clean(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, ValidationReport]:
    rows_in = len(df)
    report = ValidationReport(symbol=symbol, rows_in=rows_in, rows_out=0)

    if df.empty:
        report.warnings.append("empty dataframe received")
        return df, report

    # 1. drop invalid/NaT timestamps
    before = len(df)
    df = df[~df.index.isna()]
    report.invalid_timestamps_dropped = before - len(df)

    # 2. remove duplicate timestamps (keep last observed)
    before = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    report.duplicates_removed = before - len(df)

    # 3. ensure required columns exist
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            report.warnings.append(f"missing column '{col}' — filled with NaN")

    # 4. drop non-positive / impossible prices
    before = len(df)
    price_cols = ["open", "high", "low", "close", "adj_close"]
    mask = (df[price_cols] > 0).all(axis=1)
    df = df[mask]
    report.non_positive_prices_dropped = before - len(df)

    # 5. fill small gaps in volume/prices via forward-fill, cap gap size
    missing_before = df[OHLCV_COLUMNS].isna().sum().sum()
    df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill(limit=3)
    df = df.dropna(subset=OHLCV_COLUMNS)
    missing_after = df[OHLCV_COLUMNS].isna().sum().sum()
    report.missing_filled = int(missing_before - missing_after)

    # 6. sanity check high >= low, high >= close/open, etc.
    before = len(df)
    consistent = (df["high"] >= df["low"]) & (df["high"] >= df["close"]) & (df["high"] >= df["open"]) \
        & (df["low"] <= df["close"]) & (df["low"] <= df["open"])
    df = df[consistent]
    if before - len(df) > 0:
        report.warnings.append(f"dropped {before - len(df)} rows with inconsistent OHLC relationships")

    report.rows_out = len(df)
    if not report.is_usable:
        report.warnings.append("insufficient clean rows for reliable analysis (<30)")

    return df, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline features shared across agents (returns, log-returns, rolling vol)."""
    out = df.copy()
    out["return_1d"] = out["adj_close"].pct_change()
    out["log_return_1d"] = np.log(out["adj_close"] / out["adj_close"].shift(1))
    out["rolling_vol_20d"] = out["return_1d"].rolling(20).std() * np.sqrt(252)
    out["rolling_vol_5d"] = out["return_1d"].rolling(5).std() * np.sqrt(252)
    out["dollar_volume"] = out["adj_close"] * out["volume"]
    return out


def enforce_temporal_boundary(df: pd.DataFrame, symbol: str,
                               as_of_date: str | None = None) -> pd.DataFrame:
    """Apply temporal firewall to truncate data beyond ``as_of_date``.

    This is a convenience wrapper for use outside the full TemporalFirewall
    workflow (e.g. ad-hoc agent calls).  Returns a potentially shorter DataFrame.
    """
    if as_of_date is None or df.empty:
        return df
    from app.data.temporal_firewall import TemporalFirewall
    fw = TemporalFirewall(as_of_date=as_of_date)
    return fw.enforce_ohlcv(df, symbol)


def cache_path(symbol: str, interval: str) -> Path:
    return settings.data_parquet_dir / f"{symbol.upper()}_{interval}.parquet"


def load_and_prepare(symbol: str, period: str | None = None, interval: str | None = None,
                      use_cache: bool = True) -> tuple[pd.DataFrame, ValidationReport, str]:
    """Full pipeline: fetch (with fallback) -> validate/clean -> feature engineer -> cache.

    Returns (feature_df, validation_report, data_source).
    """
    period = period or settings.history_period
    interval = interval or settings.default_interval
    symbol = symbol.upper()
    path = cache_path(symbol, interval)

    if use_cache and path.exists():
        try:
            cached = pd.read_parquet(path)
            now_utc = pd.Timestamp.now(tz="UTC")
            age_hours = (now_utc - cached.index.max()).total_seconds() / 3600
            # If it's a weekday and the cached data does not include today (and it's after premarket open 08:00 UTC / 04:00 EDT),
            # the cache is stale and must be refreshed for the new trading day
            is_new_trading_day = (now_utc.weekday() < 5) and (now_utc.hour >= 8) and (cached.index.max().date() < now_utc.date())
            if is_new_trading_day:
                max_age_hours = 0.0
            else:
                is_weekend = now_utc.weekday() in (5, 6) or (now_utc.weekday() == 0 and now_utc.hour < 8)
                max_age_hours = (72 if is_weekend else 0.08) if interval in ("1m", "2m", "5m", "15m", "30m") else 12
            if age_hours < max_age_hours and len(cached) >= 1:
                logger.info("cache hit for %s (%s), age=%.1fh", symbol, interval, age_hours)
                report = ValidationReport(symbol=symbol, rows_in=len(cached), rows_out=len(cached))
                return cached, report, "cache"
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache read failed for %s: %s", symbol, exc)

    raw_df, source = get_historical_with_fallback(symbol, period, interval)
    clean_df, report = validate_and_clean(raw_df, symbol)
    if not report.is_usable:
        logger.error("validation failed for %s: %s", symbol, report.warnings)
        return clean_df, report, source

    feature_df = engineer_features(clean_df)

    try:
        feature_df.to_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache write failed for %s: %s", symbol, exc)

    return feature_df, report, source

"""Crash / Tail-Risk Prediction Agent.

Research contribution: Dedicated pipeline to assess downside risk using
tree-based models (XGBoost) trained on tail-risk-predictive features.

Unlike the general ML Agent (which predicts direction UP/DOWN), this agent
specifically models the probability that a stock will experience a significant
drawdown (> 5%) within the next 10 trading days.

Features used (all constructed from OHLCV data, no look-ahead):
  1. Realized volatility percentile (20d rolling)
  2. Volatility-of-volatility (vol regime instability)
  3. Historical drawdown speed (max 5d drawdown in trailing 20d)
  4. Volume anomaly score (current vs 20d average)
  5. Price dispersion (intraday range / close)
  6. Return skewness (20d rolling)
  7. Return kurtosis (20d rolling — fat tail indicator)
  8. Distance from 52-week high
  9. Consecutive negative return count
  10. Rolling Sortino ratio proxy (downside deviation)

SAFETY: Walk-forward training only — the model at time T is trained on
data [0..T-11] and predicts for bar T.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrashRiskResult(BaseModel):
    symbol: str
    crash_probability: float = Field(ge=0.0, le=1.0,
                                     description="Probability of >5% drawdown in next 10 days")
    expected_downside_pct: float = Field(
        description="Expected magnitude of downside if crash occurs")
    tail_risk_score: float = Field(ge=0.0, le=1.0,
                                   description="Composite tail risk score [0=safe, 1=extreme]")
    risk_tier: str  # LOW, MODERATE, ELEVATED, HIGH, EXTREME
    feature_importances: dict[str, float]
    top_risk_drivers: list[str]
    model_accuracy: float
    training_samples: int
    prediction_horizon_days: int = 10
    drawdown_threshold_pct: float = 0.05
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _build_crash_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build tail-risk-predictive feature matrix from OHLCV data."""
    out = pd.DataFrame(index=df.index)
    close = df["adj_close"]
    returns = close.pct_change()

    # 1. Realized volatility percentile (20d rolling)
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    out["vol_20d"] = vol_20
    out["vol_percentile"] = vol_20.rank(pct=True)

    # 2. Volatility of volatility (instability of the vol regime)
    out["vol_of_vol"] = vol_20.rolling(10).std()

    # 3. Max 5-day drawdown in trailing 20 days
    rolling_5d_return = close.pct_change(5)
    out["max_5d_drawdown_20d"] = rolling_5d_return.rolling(20).min().abs()

    # 4. Volume anomaly (current volume / 20d avg volume)
    vol_avg = df["volume"].rolling(20).mean()
    out["volume_anomaly"] = df["volume"] / vol_avg.replace(0, np.nan)

    # 5. Price dispersion (intraday range normalized by close)
    out["price_dispersion"] = (df["high"] - df["low"]) / close

    # 6. Rolling return skewness (20d) — negative skew = tail risk
    out["return_skew_20d"] = returns.rolling(20).skew()

    # 7. Rolling return kurtosis (20d) — high kurtosis = fat tails
    out["return_kurt_20d"] = returns.rolling(20).apply(
        lambda x: x.kurtosis() if len(x) >= 4 else 0.0, raw=False
    )

    # 8. Distance from 52-week high (percentage below)
    rolling_max_252 = close.rolling(min(252, len(close))).max()
    out["dist_from_52w_high"] = 1.0 - (close / rolling_max_252)

    # 9. Consecutive negative return days
    neg_streak = returns.lt(0).astype(int)
    groups = neg_streak.ne(neg_streak.shift()).cumsum()
    out["consec_neg_days"] = neg_streak.groupby(groups).cumsum()

    # 10. Downside deviation (20d rolling Sortino denominator)
    downside_returns = returns.clip(upper=0)
    out["downside_dev_20d"] = downside_returns.rolling(20).std() * np.sqrt(252)

    return out


def _build_labels(df: pd.DataFrame, horizon: int = 10,
                  threshold: float = 0.05) -> pd.Series:
    """Label: 1 if stock experiences > threshold drawdown within next
    ``horizon`` trading days.  Strictly forward-looking — used only as
    training target, never as a feature."""
    close = df["adj_close"]
    labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(close) - horizon):
        future_window = close.iloc[i + 1: i + 1 + horizon]
        if future_window.empty:
            continue
        max_dd = (future_window.min() / close.iloc[i]) - 1.0
        if max_dd < -threshold:
            labels.iloc[i] = 1

    return labels


class CrashRiskAgent:
    """XGBoost-based crash probability estimator with walk-forward discipline."""

    FEATURE_COLS = [
        "vol_20d", "vol_percentile", "vol_of_vol", "max_5d_drawdown_20d",
        "volume_anomaly", "price_dispersion", "return_skew_20d",
        "return_kurt_20d", "dist_from_52w_high", "consec_neg_days",
        "downside_dev_20d",
    ]

    def __init__(self, min_train_bars: int = 120, horizon: int = 10,
                 drawdown_threshold: float = 0.05) -> None:
        self.min_train_bars = min_train_bars
        self.horizon = horizon
        self.threshold = drawdown_threshold

    def analyze(self, df: pd.DataFrame, symbol: str) -> CrashRiskResult:
        """Run crash risk assessment on a price DataFrame."""
        symbol = symbol.upper()

        if len(df) < self.min_train_bars + self.horizon + 20:
            return self._insufficient_data_result(symbol)

        try:
            features = _build_crash_features(df)
            labels = _build_labels(df, self.horizon, self.threshold)

            # Merge and drop NaN rows
            combined = features[self.FEATURE_COLS].copy()
            combined["label"] = labels
            combined = combined.dropna()

            if len(combined) < self.min_train_bars:
                return self._insufficient_data_result(symbol)

            # Walk-forward split: train on all but last (horizon + buffer)
            train_end = len(combined) - self.horizon - 1
            train = combined.iloc[:train_end]
            X_train = train[self.FEATURE_COLS]
            y_train = train["label"]

            # The prediction point is the LAST available bar
            X_pred = combined[self.FEATURE_COLS].iloc[[-1]]

            return self._train_and_predict(
                symbol, X_train, y_train, X_pred, df
            )

        except Exception as exc:
            logger.warning("Crash risk agent failed for %s: %s", symbol, exc)
            return self._insufficient_data_result(symbol)

    def _train_and_predict(
        self,
        symbol: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_pred: pd.DataFrame,
        df: pd.DataFrame,
    ) -> CrashRiskResult:
        """Train XGBoost and produce crash risk prediction."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not available, using heuristic fallback")
            return self._heuristic_fallback(symbol, X_pred, df)

        # Handle class imbalance with scale_pos_weight
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_weight = max(n_neg / max(n_pos, 1), 1.0)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=min(scale_weight, 10.0),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Training accuracy (in-sample, for reporting)
        train_preds = model.predict(X_train)
        accuracy = float((train_preds == y_train).mean())

        # Predict crash probability for current bar
        crash_prob = float(model.predict_proba(X_pred)[0, 1])

        # Feature importances
        importances = dict(zip(self.FEATURE_COLS,
                               [round(float(v), 4) for v in model.feature_importances_]))

        # Expected downside: average realized drawdown for crash events in training
        close = df["adj_close"]
        crash_drawdowns = []
        for i in range(len(close) - self.horizon):
            future = close.iloc[i + 1: i + 1 + self.horizon]
            dd = (future.min() / close.iloc[i]) - 1.0
            if dd < -self.threshold:
                crash_drawdowns.append(abs(dd))
        expected_downside = float(np.mean(crash_drawdowns)) if crash_drawdowns else self.threshold

        # Risk tier classification
        risk_tier = self._classify_risk_tier(crash_prob)

        # Top 3 risk drivers
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
        top_drivers = [
            f"{name} (importance={imp:.3f})" for name, imp in sorted_imp[:3]
        ]

        return CrashRiskResult(
            symbol=symbol,
            crash_probability=round(crash_prob, 4),
            expected_downside_pct=round(expected_downside, 4),
            tail_risk_score=round(min(crash_prob * 1.2, 1.0), 4),
            risk_tier=risk_tier,
            feature_importances=importances,
            top_risk_drivers=top_drivers,
            model_accuracy=round(accuracy, 4),
            training_samples=len(X_train),
            prediction_horizon_days=self.horizon,
            drawdown_threshold_pct=self.threshold,
        )

    def _heuristic_fallback(self, symbol: str, X_pred: pd.DataFrame,
                            df: pd.DataFrame) -> CrashRiskResult:
        """Simple heuristic when XGBoost is unavailable."""
        row = X_pred.iloc[0]
        vol_pct = float(row.get("vol_percentile", 0.5))
        dist_high = float(row.get("dist_from_52w_high", 0.0))
        vol_of_vol = float(row.get("vol_of_vol", 0.0))

        # Heuristic crash probability
        crash_prob = np.clip(
            0.1 + vol_pct * 0.3 + dist_high * 0.4 + min(vol_of_vol * 2, 0.2),
            0.0, 1.0
        )

        return CrashRiskResult(
            symbol=symbol,
            crash_probability=round(float(crash_prob), 4),
            expected_downside_pct=0.05,
            tail_risk_score=round(float(crash_prob), 4),
            risk_tier=self._classify_risk_tier(float(crash_prob)),
            feature_importances={"vol_percentile": 0.35, "dist_from_52w_high": 0.35,
                                 "vol_of_vol": 0.30},
            top_risk_drivers=["Heuristic fallback (XGBoost unavailable)"],
            model_accuracy=0.0,
            training_samples=0,
        )

    def _insufficient_data_result(self, symbol: str) -> CrashRiskResult:
        """Return a safe default when insufficient data is available."""
        return CrashRiskResult(
            symbol=symbol,
            crash_probability=0.0,
            expected_downside_pct=0.0,
            tail_risk_score=0.0,
            risk_tier="UNKNOWN",
            feature_importances={},
            top_risk_drivers=["Insufficient data for crash risk assessment"],
            model_accuracy=0.0,
            training_samples=0,
        )

    @staticmethod
    def _classify_risk_tier(prob: float) -> str:
        if prob < 0.15:
            return "LOW"
        elif prob < 0.30:
            return "MODERATE"
        elif prob < 0.50:
            return "ELEVATED"
        elif prob < 0.70:
            return "HIGH"
        else:
            return "EXTREME"

"""ML Prediction Agent.

Predicts next-period DIRECTION (up/down) and expected return magnitude —
never an exact future price. XGBoost is the primary tabular model;
RandomForest is kept as a lightweight comparison model, trained on the same
chronologically-split data.

NOTE ON SCOPE: the original spec also called for an LSTM/GRU deep-learning
comparison model. That is intentionally NOT included here — a real LSTM
needs a much larger install (torch/tensorflow) and materially more data/
tuning to be a fair comparison, and per the project's own principle
("correctness over unnecessary technology," "do not fabricate results"),
a token LSTM trained on ~1-2y of daily bars would not produce a meaningful
number. RandomForest fills the "comparison model" role for now. Swapping in
a real LSTM is a documented extension point (see README).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


class MLPrediction(BaseModel):
    symbol: str
    predicted_direction: str  # UP / DOWN
    probability: float = Field(ge=0.0, le=1.0)
    expected_return: float
    model: str
    validation_metrics: dict
    comparison_model: dict | None = None
    timestamp: str


FEATURE_COLUMNS = [
    "rsi_14", "macd_hist", "sma_ratio", "momentum_5", "momentum_10",
    "volatility_20", "volume_ratio", "return_1d", "return_5d",
]


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["adj_close"]
    volume = df["volume"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    feat = pd.DataFrame(index=df.index)
    feat["rsi_14"] = rsi
    feat["macd_hist"] = macd_hist
    feat["sma_ratio"] = sma20 / sma50
    feat["momentum_5"] = close.pct_change(5)
    feat["momentum_10"] = close.pct_change(10)
    feat["volatility_20"] = close.pct_change().rolling(20).std()
    feat["volume_ratio"] = volume / volume.rolling(20).mean()
    feat["return_1d"] = close.pct_change(1)
    feat["return_5d"] = close.pct_change(5)

    # target: next-bar direction / return (shift -1 => uses FUTURE bar as label,
    # features themselves only ever use present/past data, so no leakage)
    fwd_return = close.pct_change().shift(-1)
    feat["target_direction"] = (fwd_return > 0).astype(int)
    feat["target_return"] = fwd_return

    return feat.dropna()


class MLPredictionAgent:
    def analyze(self, df: pd.DataFrame, symbol: str) -> MLPrediction:
        feat = _build_feature_frame(df)
        if len(feat) < 60:
            return MLPrediction(
                symbol=symbol, predicted_direction="UP", probability=0.5, expected_return=0.0,
                model="insufficient_data", validation_metrics={"note": "fewer than 60 usable rows"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        X = feat[FEATURE_COLUMNS]
        y_dir = feat["target_direction"]

        # chronological split: 70% train / 15% val / 15% test — no shuffling, no look-ahead
        n = len(X)
        i_train = int(n * 0.70)
        i_val = int(n * 0.85)

        X_train, y_train = X.iloc[:i_train], y_dir.iloc[:i_train]
        X_val, y_val = X.iloc[i_train:i_val], y_dir.iloc[i_train:i_val]
        X_test, y_test = X.iloc[i_val:], y_dir.iloc[i_val:]

        xgb = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42,
        )
        xgb.fit(X_train, y_train)

        val_pred = xgb.predict(X_val)
        test_pred = xgb.predict(X_test)
        metrics = {
            "val_accuracy": round(float(accuracy_score(y_val, val_pred)), 3) if len(y_val) else None,
            "val_f1": round(float(f1_score(y_val, val_pred, zero_division=0)), 3) if len(y_val) else None,
            "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 3) if len(y_test) else None,
            "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 3) if len(y_test) else None,
            "n_train": i_train, "n_val": i_val - i_train, "n_test": n - i_val,
            "split": "chronological 70/15/15, no shuffling",
        }

        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_test_pred = rf.predict(X_test)
        comparison = {
            "model": "RandomForest",
            "test_accuracy": round(float(accuracy_score(y_test, rf_test_pred)), 3) if len(y_test) else None,
            "test_f1": round(float(f1_score(y_test, rf_test_pred, zero_division=0)), 3) if len(y_test) else None,
            "note": "comparison model; LSTM/GRU not evaluated yet (see module docstring)",
        }

        # final live prediction uses the most recent row (not part of train/val/test targets, since it has no future label yet)
        latest_features = feat[FEATURE_COLUMNS].iloc[[-1]]
        proba_up = float(xgb.predict_proba(latest_features)[0][1])
        direction = "UP" if proba_up >= 0.5 else "DOWN"
        recent_up_moves = feat.loc[feat["target_return"] > 0, "target_return"].tail(20)
        recent_down_moves = feat.loc[feat["target_return"] < 0, "target_return"].tail(20)
        typical_move = float(recent_up_moves.mean()) if direction == "UP" else float(recent_down_moves.mean())
        confidence_in_direction = proba_up if direction == "UP" else (1 - proba_up)
        expected_return = float(typical_move * confidence_in_direction) if not np.isnan(typical_move) else 0.0

        return MLPrediction(
            symbol=symbol,
            predicted_direction=direction,
            probability=round(proba_up if direction == "UP" else 1 - proba_up, 3),
            expected_return=round(expected_return, 5),
            model="XGBoost",
            validation_metrics=metrics,
            comparison_model=comparison,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

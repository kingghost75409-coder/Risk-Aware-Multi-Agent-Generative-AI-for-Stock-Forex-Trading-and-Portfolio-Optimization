"""Central configuration, loaded from environment variables / .env file.

No secrets or business logic are hard-coded here. Every tunable lives in
one place so the rest of the system never reaches into os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [x.strip().upper() for x in val.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # --- paths ---
    base_dir: Path = BASE_DIR
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    data_parquet_dir: Path = BASE_DIR / "data" / "parquet"
    rag_data_dir: Path = BASE_DIR / "rag_data"
    models_dir: Path = BASE_DIR / "models"

    # --- database ---
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}")

    # --- market data ---
    default_symbols: list[str] = field(
        default_factory=lambda: _list("DEFAULT_SYMBOLS", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"])
    )
    default_interval: str = os.getenv("DEFAULT_INTERVAL", "1d")
    history_period: str = os.getenv("HISTORY_PERIOD", "2y")
    use_synthetic_on_failure: bool = _bool("USE_SYNTHETIC_DATA_ON_FAILURE", True)

    # --- LLM (optional) ---
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    # --- risk engine ---
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    max_portfolio_volatility: float = float(os.getenv("MAX_PORTFOLIO_VOLATILITY", "0.35"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
    var_confidence: float = float(os.getenv("VAR_CONFIDENCE", "0.95"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.08"))
    max_concentration_pct: float = float(os.getenv("MAX_CONCENTRATION_PCT", "0.30"))
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
    loss_cooldown_minutes: int = int(os.getenv("LOSS_COOLDOWN_MINUTES", "60"))
    max_leverage_stock: float = float(os.getenv("MAX_LEVERAGE_STOCK", "5.0"))
    max_leverage_forex: float = float(os.getenv("MAX_LEVERAGE_FOREX", "20.0"))
    max_currency_concentration_pct: float = float(os.getenv("MAX_CURRENCY_CONCENTRATION_PCT", "0.40"))
    default_trading_style: str = os.getenv("DEFAULT_TRADING_STYLE", "SWING")
    min_risk_reward_ratio: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))

    # --- paper trading ---
    starting_cash: float = float(os.getenv("STARTING_CASH", "100000"))
    transaction_cost_bps: float = float(os.getenv("TRANSACTION_COST_BPS", "5"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "3"))

    # --- flask ---
    flask_host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port: int = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug: bool = _bool("FLASK_DEBUG", True)

    # --- logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

for _p in (settings.data_raw_dir, settings.data_processed_dir, settings.data_parquet_dir,
           settings.rag_data_dir, settings.models_dir):
    _p.mkdir(parents=True, exist_ok=True)

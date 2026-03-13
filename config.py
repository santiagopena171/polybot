"""
Central configuration — loaded once at startup.
All values come from the .env file (never hardcoded).
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Wallet / Polymarket ────────────────────────
    polygon_private_key: SecretStr = Field(...)
    polymarket_api_key: str = Field(default="")
    polymarket_api_secret: SecretStr = Field(default=SecretStr(""))
    polymarket_api_passphrase: SecretStr = Field(default=SecretStr(""))

    # ── LLM ───────────────────────────────────────
    openai_api_key: SecretStr = Field(...)
    openai_model: str = Field(default="gpt-4o")

    # ── External data ──────────────────────────────
    news_api_key: SecretStr = Field(default=SecretStr(""))

    # ── Bot parameters ─────────────────────────────
    min_edge: float = Field(default=0.07)
    max_kelly_fraction: float = Field(default=0.25)
    max_trade_size_usdc: float = Field(default=50.0)
    markets_to_scan: int = Field(default=30)
    scan_interval_seconds: int = Field(default=300)
    dry_run: bool = Field(default=True)

    # ── Logging ────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/bot.log")

    # ── Polymarket endpoints (rarely changed) ──────
    clob_host: str = Field(default="https://clob.polymarket.com")
    gamma_host: str = Field(default="https://gamma-api.polymarket.com")


# Singleton — import this everywhere
settings = Settings()

"""Application configuration and environment validation."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, validator


REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_SECRET_TOKEN",
    "PAPER_MODE",
    "LIVE",
    "TZ",
    "LOG_LEVEL",
    "WEBHOOK_RPS",
    "WEBHOOK_BURST",
    "PORT",
]


class ConfigError(RuntimeError):
    """Raised when the environment configuration is invalid."""


class Settings(BaseModel):
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_secret_token: str = Field(..., alias="TELEGRAM_SECRET_TOKEN")
    paper_mode: bool = Field(..., alias="PAPER_MODE")
    live: bool = Field(..., alias="LIVE")
    timezone: str = Field(..., alias="TZ")
    log_level: str = Field(..., alias="LOG_LEVEL")
    webhook_rps: float = Field(..., alias="WEBHOOK_RPS")
    webhook_burst: int = Field(..., alias="WEBHOOK_BURST")
    port: int = Field(..., alias="PORT")
    daily_loss_limit: float = Field(1000.0, alias="DAILY_LOSS_LIMIT")
    data_dir: str = Field("./run", alias="DATA_DIR")

    class Config:
        allow_population_by_field_name = True
        anystr_strip_whitespace = True

    @validator("telegram_bot_token", "telegram_secret_token", "timezone", "log_level")
    def non_empty(cls, value: str) -> str:  # noqa: D417
        if not value:
            raise ValueError("must not be empty")
        return value

    @validator("webhook_rps")
    def validate_rps(cls, value: float) -> float:  # noqa: D417
        if value <= 0:
            raise ValueError("WEBHOOK_RPS must be positive")
        return value

    @validator("webhook_burst")
    def validate_burst(cls, value: int) -> int:  # noqa: D417
        if value <= 0:
            raise ValueError("WEBHOOK_BURST must be positive")
        return value

    @validator("port")
    def validate_port(cls, value: int) -> int:  # noqa: D417
        if not (0 < value < 65536):
            raise ValueError("PORT must be a valid TCP port")
        return value

    @validator("daily_loss_limit")
    def validate_loss_limit(cls, value: float) -> float:  # noqa: D417
        if value <= 0:
            raise ValueError("DAILY_LOSS_LIMIT must be positive")
        return value

    def ensure_modes(self) -> None:
        if not self.paper_mode:
            raise ConfigError("PAPER_MODE must be true")
        if self.live:
            raise ConfigError("LIVE must be false")


def _missing_required_env() -> list[str]:
    return [key for key in REQUIRED_ENV_VARS if key not in os.environ or os.environ[key] == ""]


def load_settings() -> Settings:
    missing = _missing_required_env()
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(sorted(missing))}")
    try:
        env_values = {
            field.alias: os.environ.get(field.alias)
            for field in Settings.__fields__.values()
            if os.environ.get(field.alias) is not None
        }
        settings = Settings(**env_values)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    settings.ensure_modes()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]


def is_environment_valid() -> tuple[bool, Optional[str]]:
    try:
        reset_settings_cache()
        get_settings()
    except ConfigError as exc:
        return False, str(exc)
    return True, None

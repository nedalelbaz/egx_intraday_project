"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from fastapi import FastAPI, HTTPException, status

from . import config
from .config import ConfigError
from .ledger import LedgerStore
from .logging_utils import configure_logging
from .rate_limit import RateLimiter
from .risk import RiskManager
from .webhook import router as webhook_router

app = FastAPI(title="Paper Telegram Webhook", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    try:
        settings = config.get_settings()
    except ConfigError as exc:
        app.state.startup_error = str(exc)
        logging.getLogger(__name__).error("startup configuration error", extra={"error": str(exc)})
        return

    configure_logging(settings)
    ledger_store = LedgerStore(settings.data_dir)
    limiter = RateLimiter(settings.webhook_rps, settings.webhook_burst)
    risk_manager = RiskManager(settings.daily_loss_limit, ledger_store)

    app.state.settings = settings
    app.state.rate_limiter = limiter
    app.state.risk_manager = risk_manager
    app.state.startup_error = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    valid, reason = config.is_environment_valid()
    if not valid:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=reason or "invalid configuration")
    return {"status": "ok"}


app.include_router(webhook_router)

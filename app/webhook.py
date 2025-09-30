"""Telegram webhook endpoint and dependencies."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status

from .config import Settings
from .rate_limit import RateLimiter
from .risk import RiskManager

router = APIRouter()


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Configuration unavailable")
    return settings


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Rate limiter unavailable")
    return limiter


def get_risk_manager(request: Request) -> RiskManager:
    manager = getattr(request.app.state, "risk_manager", None)
    if manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Risk manager unavailable")
    return manager


@router.post("/webhook/{token}")
async def telegram_webhook(
    token: str,
    request: Request,
    payload: Dict[str, Any] = Body(...),
    secret_header: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
    settings: Settings = Depends(get_settings),
    limiter: RateLimiter = Depends(get_rate_limiter),
    risk_manager: RiskManager = Depends(get_risk_manager),
) -> Dict[str, Any]:
    if token != settings.telegram_bot_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    if secret_header != settings.telegram_secret_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")

    trade = payload.get("trade") if isinstance(payload, dict) else None
    trade = trade if isinstance(trade, dict) else {}
    accepted = risk_manager.process_trade(trade)
    if not accepted:
        raise HTTPException(status_code=status.HTTP_423_LOCKED, detail="Trade rejected by risk manager")

    return {"status": "ok"}

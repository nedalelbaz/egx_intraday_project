"""
Webhook service for the EGX intraday assistant.

This FastAPI app proxies Telegram webhook, health, status and metrics endpoints
to the EGX intraday assistant. It dynamically locates the assistant module
(egx_intraday_project/egx_intraday_enhanced.py) unless ASSISTANT_PATH env var is set.
"""

import datetime as _dt
import importlib.util
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Attempt to import the assistant
def _load_assistant():
    """Dynamically locate and load the EGX intraday assistant."""
    assistant = None
    cfg: Dict[str, Any] = {}
    try:
        from pathlib import Path
        assistant_path = os.environ.get("ASSISTANT_PATH")
        if assistant_path:
            candidate = Path(assistant_path)
        else:
            here = Path(__file__).resolve()
            candidate = None
            for parent in here.parents:
                path = parent / "egx_intraday_project" / "egx_intraday_enhanced.py"
                if path.exists():
                    candidate = path
                    break
        if candidate and candidate.exists():
            spec = importlib.util.spec_from_file_location("egx_intraday_enhanced_dynamic", str(candidate))
            module = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            import sys
            sys.modules[spec.name] = module  # type: ignore
            spec.loader.exec_module(module)  # type: ignore
            EGXAssistant = getattr(module, "EGXAssistant")
            load_config = getattr(module, "load_config")
            from pathlib import Path as _Path
            cfg_env = os.environ.get("CONFIG_PATH")
            cfg_path = _Path(cfg_env) if cfg_env else candidate.parent / "config.yaml"
            cfg = load_config(str(cfg_path))
            assistant = EGXAssistant(cfg)
    except Exception as exc:
        logger.warning(f"Failed to initialise assistant: {exc}")
    return assistant, cfg

_assistant, _cfg = _load_assistant()

app = FastAPI(title="EGX Intraday Webhook", version=os.getenv("COMMIT_SHA", "0.0.0"))

@app.get("/healthz", summary="Health check")
async def healthz() -> Dict[str, Any]:
    """Return a basic liveness and readiness check."""
    return {
        "status": "ok",
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "assistant_ready": _assistant is not None,
        "version": os.getenv("COMMIT_SHA", "unknown"),
    }

@app.get("/status", summary="Assistant status")
async def status() -> Dict[str, Any]:
    """Return a minimal status snapshot."""
    if _assistant is None:
        return {"assistant_ready": False, "message": "Assistant not initialised"}
    try:
        return {
            "assistant_ready": True,
            "equity": getattr(_assistant.trader, "equity", None),
            "open_positions": len(getattr(_assistant.trader, "open_positions", [])),
            "today_pnl": getattr(_assistant.trader, "today_pnl", None),
            "timestamp_cairo": _dt.datetime.now(getattr(_assistant, "tz", _dt.timezone.utc)).isoformat(),
        }
    except Exception as exc:
        logger.error(f"Failed to compute status: {exc}")
        return {"assistant_ready": False, "error": "Error computing status"}

@app.post("/webhook/{token}", summary="Telegram webhook")
async def telegram_webhook(token: str, request: Request) -> JSONResponse:
    """Handle incoming Telegram bot updates."""
    configured_token = os.environ.get("TELEGRAM_BOT_TOKEN") or _cfg.get("alerts", {}).get("telegram_bot_token")
    if configured_token and token != configured_token:
        raise HTTPException(status_code=404, detail="Invalid token")
    try:
        payload = await request.json()
    except Exception:
        payload = None
    logger.info("Received Telegram update: %s", payload)
    if _assistant is not None and payload is not None:
        try:
            _handler = getattr(_assistant.tg, "handle_update", None)
        if callable(handler):
            handler(payload)
        # else, ignore unhandled updates
        except Exception as exc:
            logger.error(f"Error handling Telegram update: {exc}")
    return JSONResponse(content={})

@app.get("/metrics", summary="Operational metrics")
async def metrics() -> Dict[str, Any]:
    """Return a small set of operational metrics."""
    data: Dict[str, Any] = {
        "bar_age_seconds": 0,
        "last_cycle_time_seconds": 0,
        "provider_error_rate": 0.0,
        "open_positions_count": 0,
        "version": os.getenv("COMMIT_SHA", "unknown"),
    }
    if _assistant is not None:
        try:
            last_ts = getattr(_assistant, "_last_bar_ts", None)
            if last_ts:
                age = (_dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc) - last_ts).total_seconds()
                data["bar_age_seconds"] = max(age, 0)
        except Exception:
            pass
        try:
            data["last_cycle_time_seconds"] = float(getattr(_assistant, "_last_cycle_time", 0))
        except Exception:
            pass
        try:
            data["provider_error_rate"] = float(getattr(_assistant, "_provider_error_rate", 0))
        except Exception:
            pass
        try:
            data["open_positions_count"] = len(getattr(_assistant.trader, "open_positions", []))
        except Exception:
            pass
    return data

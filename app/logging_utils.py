"""Structured logging utilities with secret redaction."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

from pythonjsonlogger import jsonlogger

from .config import Settings

SECRET_PATTERN = re.compile(r"([A-Za-z0-9_:-]{10,})")
REDACTED = "***REDACTED***"


class RedactingJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter that redacts token-like values."""

    def json_record(self, message: str, extra: Dict[str, Any], record: logging.LogRecord) -> Dict[str, Any]:
        data = super().json_record(message, extra, record)
        for key, value in list(data.items()):
            data[key] = self._redact_value(value)
        return data

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("http"):
                return SECRET_PATTERN.sub(REDACTED, value)
            return REDACTED if SECRET_PATTERN.fullmatch(value) else SECRET_PATTERN.sub(REDACTED, value)
        if isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact_value(v) for v in value]
        return value


def configure_logging(settings: Settings) -> None:
    """Configure root logger with JSON output and redaction."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    handler = logging.StreamHandler()
    fmt = RedactingJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    root_logger.addHandler(handler)

    os.environ.setdefault("TZ", settings.timezone)

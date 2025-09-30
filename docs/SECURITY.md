# Security Overview

## Authentication & Authorization

* Webhook path includes `TELEGRAM_BOT_TOKEN`; mismatches return HTTP 404 to avoid disclosure.
* Header `X-Telegram-Bot-Api-Secret-Token` must equal `TELEGRAM_SECRET_TOKEN`; mismatches return HTTP 403.
* Rate limiting mitigates abuse by limiting requests per IP using `WEBHOOK_RPS` and `WEBHOOK_BURST`.

## Configuration Safety

* Strict environment validation requires PAPER mode (`PAPER_MODE=true`, `LIVE=false`).
* Missing or invalid configuration causes `/healthz` to return HTTP 500, preventing accidental live usage.
* `TZ`, `LOG_LEVEL`, and port/rate limit values must be explicitly configured to reduce misconfiguration risk.

## Logging

* Structured JSON logs with automatic redaction prevent secrets or token-like strings from leaking.
* Log level is controlled by `LOG_LEVEL`; prefer `INFO` or `WARNING` in production.

## Data Handling

* Ledgers (`trades.jsonl`, `reserve.jsonl`) are append-only JSONL files generated at runtime.
* No secrets are written to disk; only simulated trade metadata and reserve transfers are stored.
* Configure `DATA_DIR` to point to secured storage; avoid committing ledger files.

## Dependency & Runtime Hardening

* Dependencies pinned in `requirements.txt` for deterministic builds.
* Docker image runs as non-root `appuser` and exposes only port 8000.
* Docker `HEALTHCHECK` polls `/healthz` to ensure readiness before traffic is routed.

## Secret Management

* Provide Telegram tokens via environment variables or managed secret stores (Render dashboard, etc.).
* Rotate tokens periodically and re-register the webhook using `scripts/register_webhook_example.sh`.
* Never log or hardcode secrets within the repository.

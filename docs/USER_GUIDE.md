# User Guide

Welcome to the PAPER-only Telegram webhook service. This guide explains how to interact with the API safely in a simulated environment.

## 1. Overview

The service receives Telegram webhook updates, validates their authenticity, applies rate limits and risk controls, and records simulated trades in append-only ledgers. It never performs live trading actions.

## 2. API Endpoints

### `GET /healthz`

* **Purpose**: Verify configuration integrity.
* **200 OK**: Environment is fully configured for PAPER mode.
* **500 Internal Server Error**: One or more required environment variables are missing or invalid.

### `POST /webhook/{TELEGRAM_BOT_TOKEN}`

* **Authentication**:
  * Path token **must** match `TELEGRAM_BOT_TOKEN`.
  * Header `X-Telegram-Bot-Api-Secret-Token` **must** match `TELEGRAM_SECRET_TOKEN`.
* **Rate Limiting**: In-memory token bucket keyed by client IP using `WEBHOOK_RPS` and `WEBHOOK_BURST`.
* **Payload**: Expect JSON with an optional `trade` object containing `trade_id` and `realized_pnl`.
* **Responses**:
  * `200 OK` – trade accepted and logged.
  * `403 Forbidden` – missing or incorrect secret header.
  * `404 Not Found` – path token mismatch.
  * `423 Locked` – trade rejected by risk manager (daily loss cap).
  * `429 Too Many Requests` – rate limit exceeded.

## 3. Trade Simulation Workflow

1. Telegram forwards updates to the webhook URL.
2. The service validates the token and secret header.
3. Rate limiter enforces traffic policy.
4. Risk manager applies the daily loss cap.
5. Accepted trades are appended to `trades.jsonl`; rejections include the reason.
6. End-of-day settlement (manual or scheduled) transfers 2% of positive realized profit to the reserve ledger.

## 4. Logs and Observability

* Logs are emitted in JSON with sensitive values redacted.
* `LOG_LEVEL` controls verbosity (`INFO` recommended in production).
* Inspect logs for rate limit events, risk rejections, and settlement summaries.

## 5. Ledgers

Ledgers are generated at runtime under `DATA_DIR/ledgers/`:

* `trades.jsonl` – all accepted trades, rejections, and settlements.
* `reserve.jsonl` – reserve allocations recorded during settlements.

Each line is an individual JSON object with an ISO timestamp.

## 6. Best Practices

* Keep `PAPER_MODE=true` and `LIVE=false` at all times.
* Rotate Telegram tokens regularly; use `scripts/register_webhook_example.sh` as a reference.
* Schedule a daily settlement job (cron or external task runner) to maintain the reserve ledger.
* Review ledgers to assess simulated performance and adherence to risk controls.

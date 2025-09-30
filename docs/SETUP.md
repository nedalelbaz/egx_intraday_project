# Setup Guide

Follow these steps to run the PAPER Telegram webhook service locally.

## 1. Prerequisites

* Python 3.11+
* Virtual environment tool (e.g., `venv` or `virtualenv`)
* Telegram bot token and secret token

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy `.env.example` to `.env` and edit the values:

```bash
cp .env.example .env
```

Minimum required variables:

* `TELEGRAM_BOT_TOKEN`
* `TELEGRAM_SECRET_TOKEN`
* `PAPER_MODE` (set to `true`)
* `LIVE` (set to `false`)
* `TZ` (e.g., `UTC`)
* `LOG_LEVEL` (e.g., `INFO`)
* `WEBHOOK_RPS`
* `WEBHOOK_BURST`
* `PORT`

Optional:

* `DAILY_LOSS_LIMIT` (defaults to `1000`)
* `DATA_DIR` (defaults to `./run`)

Export the variables prior to running the application:

```bash
set -a
source .env
set +a
```

## 5. Run Tests

```bash
pytest -q
```

## 6. Start the Server

```bash
uvicorn app.main:app --port "$PORT"
```

The `/healthz` endpoint returns 200 only when the environment is correctly configured for PAPER mode.

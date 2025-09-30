# Deploying to Render

This guide walks through deploying the PAPER Telegram webhook to Render using the provided `render.yaml`.

## 1. Prerequisites

* Render account with access to create Web Services.
* Telegram bot token and secret token.
* GitHub or GitLab repository containing this project.

## 2. Configure Environment Variables

On the Render dashboard, create a new **Web Service** from your repository. Render reads `render.yaml` and preconfigures baseline variables:

* `PAPER_MODE=true`
* `LIVE=false`
* `TZ=UTC`
* `LOG_LEVEL=INFO`
* `WEBHOOK_RPS=1.0`
* `WEBHOOK_BURST=5`
* `DATA_DIR=/var/data`

Add the following secrets manually under **Environment Variables**:

* `TELEGRAM_BOT_TOKEN`
* `TELEGRAM_SECRET_TOKEN`
* `PORT` (Render sets this automatically; leave as `0` or `${PORT}` for runtime injection)
* Optional overrides for rate limits or loss limits as needed.

## 3. Deploy

1. Click **Create Web Service**.
2. Render builds the Docker image using the provided Dockerfile.
3. Once live, verify health:
   ```bash
   curl https://<service-name>.onrender.com/healthz
   ```
   A 200 response indicates valid PAPER configuration.

## 4. Register Telegram Webhook

After deployment, set the Telegram webhook endpoint to the Render URL:

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_SECRET_TOKEN=... \
WEBHOOK_URL=https://<service-name>.onrender.com ./scripts/register_webhook_example.sh
```

## 5. Operations

* Render stores logs in the dashboard with JSON formatting and redaction.
* Ledgers persist under `/var/data/ledgers/`; configure a persistent disk if you need retention across deploys.
* Redeployments automatically reapply configuration. Health checks fail fast if `LIVE=true` or required variables are missing.

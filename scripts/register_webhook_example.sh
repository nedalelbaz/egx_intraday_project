#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_SECRET_TOKEN:-}" || -z "${WEBHOOK_URL:-}" ]]; then
  echo "TELEGRAM_BOT_TOKEN, TELEGRAM_SECRET_TOKEN, and WEBHOOK_URL must be set" >&2
  exit 1
fi

curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"${WEBHOOK_URL}/${TELEGRAM_BOT_TOKEN}\", \"secret_token\": \"${TELEGRAM_SECRET_TOKEN}\"}"

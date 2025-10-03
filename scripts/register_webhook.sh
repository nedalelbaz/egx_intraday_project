#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PUBLIC_URL:-}" ]]; then
  echo "Usage: PUBLIC_URL=https://yourdomain.com bash $0"
  exit 1
fi

# Load token from .env if not set
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-$(grep TELEGRAM_BOT_TOKEN .env | cut -d '=' -f2)}
if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
  echo "Please set TELEGRAM_BOT_TOKEN in environment or .env file."
  exit 1
fi

# Register webhook with Telegram
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=${PUBLIC_URL}/${TELEGRAM_BOT_TOKEN}" | jq .
echo "Webhook registered to ${PUBLIC_URL}/${TELEGRAM_BOT_TOKEN}"

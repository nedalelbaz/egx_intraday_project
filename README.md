# EGX Intraday Project

This repository contains the implementation for the EGX intraday data project.

## Setup

1. Clone or download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your Telegram bot token and chat ID.

## Usage

### Telegram Notifications

Create a Telegram bot using @BotFather and obtain your `TELEGRAM_BOT_TOKEN`. Use @get_id_bot or another method to get your `TELEGRAM_CHAT_ID`.

In your GitHub repository, add these secrets under Settings → Secrets and variables → Actions:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

The GitHub Actions workflow in `.github/workflows/ci.yml` will send a message to your chat when CI runs.

### Local Test

Use the helper script to send a test message:

```bash
python scripts/send_test_message.py
```

## CI

The CI pipeline performs linting, type checking, tests, and sends a Telegram notification with the build status.

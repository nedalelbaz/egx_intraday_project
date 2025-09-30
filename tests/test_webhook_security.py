import os
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

ENV = {
    "TELEGRAM_BOT_TOKEN": "token1234567890",
    "TELEGRAM_SECRET_TOKEN": "secret1234567890",
    "PAPER_MODE": "true",
    "LIVE": "false",
    "TZ": "UTC",
    "LOG_LEVEL": "INFO",
    "WEBHOOK_RPS": "1.0",
    "WEBHOOK_BURST": "1",
    "PORT": "8000",
    "DAILY_LOSS_LIMIT": "1000",
}


def prepare_env(tmp_path):
    for key, value in ENV.items():
        os.environ[key] = value
    os.environ["DATA_DIR"] = str(tmp_path)


@pytest.fixture
def client(tmp_path):
    prepare_env(tmp_path)
    config.reset_settings_cache()
    with TestClient(app) as test_client:
        yield test_client


def test_path_token_must_match(client):
    response = client.post(
        "/webhook/wrong",
        headers={"X-Telegram-Bot-Api-Secret-Token": ENV["TELEGRAM_SECRET_TOKEN"]},
        json={"trade": {"trade_id": "t1", "realized_pnl": 0}},
    )
    assert response.status_code == 404


def test_secret_header_required(client):
    response = client.post(
        f"/webhook/{ENV['TELEGRAM_BOT_TOKEN']}",
        json={"trade": {"trade_id": "t1", "realized_pnl": 0}},
    )
    assert response.status_code == 403


def test_secret_header_must_match(client):
    response = client.post(
        f"/webhook/{ENV['TELEGRAM_BOT_TOKEN']}",
        headers={"X-Telegram-Bot-Api-Secret-Token": "incorrect"},
        json={"trade": {"trade_id": "t1", "realized_pnl": 0}},
    )
    assert response.status_code == 403


def test_rate_limit_enforced(client):
    headers = {"X-Telegram-Bot-Api-Secret-Token": ENV["TELEGRAM_SECRET_TOKEN"]}
    url = f"/webhook/{ENV['TELEGRAM_BOT_TOKEN']}"
    first = client.post(url, headers=headers, json={"trade": {"trade_id": "t1", "realized_pnl": 1}})
    assert first.status_code == 200
    second = client.post(url, headers=headers, json={"trade": {"trade_id": "t2", "realized_pnl": 1}})
    assert second.status_code == 429

import os
from fastapi.testclient import TestClient

from app import config
from app.main import app


REQUIRED = {
    "TELEGRAM_BOT_TOKEN": "token1234567890",
    "TELEGRAM_SECRET_TOKEN": "secret1234567890",
    "PAPER_MODE": "true",
    "LIVE": "false",
    "TZ": "UTC",
    "LOG_LEVEL": "INFO",
    "WEBHOOK_RPS": "1.0",
    "WEBHOOK_BURST": "5",
    "PORT": "8000",
    "DAILY_LOSS_LIMIT": "1000",
}


def set_env(tmp_path):
    for key, value in REQUIRED.items():
        os.environ[key] = value
    os.environ["DATA_DIR"] = str(tmp_path)


def test_healthz_missing_env(monkeypatch):
    for key in REQUIRED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    config.reset_settings_cache()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 500
    assert "Missing required" in response.json()["detail"]


def test_healthz_valid_env(tmp_path, monkeypatch):
    set_env(tmp_path)
    config.reset_settings_cache()
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

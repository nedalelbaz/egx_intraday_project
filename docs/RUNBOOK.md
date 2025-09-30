# Runbook

Operational procedures for the PAPER Telegram webhook service.

## 1. Health Monitoring

* Endpoint: `GET /healthz`
* Success: `{"status": "ok"}`
* Failure (HTTP 500): Missing/invalid environment variables or PAPER/LIVE misconfiguration.

**Response Actions**:
1. Inspect Render or process logs for the `startup configuration error` message.
2. Confirm all required environment variables are present and valid.
3. Ensure `PAPER_MODE=true` and `LIVE=false`.
4. Redeploy after correcting configuration.

## 2. Rate Limit Alerts

* Symptom: HTTP 429 responses in logs or Telegram retries.
* Mitigation:
  1. Verify `WEBHOOK_RPS` and `WEBHOOK_BURST` match expected traffic.
  2. Investigate sudden spikes; ensure Telegram is not retrying due to upstream failures.
  3. Adjust limits cautiously while staying within simulated constraints.

## 3. Risk Rejections

* Symptom: HTTP 423 with message "Trade rejected by risk manager".
* Logs: JSON entry with `status="rejected"` and `reason="daily-loss-cap"`.
* Remediation:
  1. Review `DATA_DIR/ledgers/trades.jsonl` for the rejected trade.
  2. Evaluate simulated strategy performance; consider lowering risk exposure.
  3. If the cap is too restrictive, adjust `DAILY_LOSS_LIMIT` responsibly and redeploy.

## 4. End-of-Day Settlement

* Recommended to trigger via scheduled job:
  ```bash
  python -c "from app.risk import RiskManager; from app.ledger import LedgerStore; import os; mgr = RiskManager(float(os.environ['DAILY_LOSS_LIMIT']), LedgerStore(os.environ['DATA_DIR'])); mgr.end_of_day_settlement()"
  ```
* Verifies reserve transfers in `reserve.jsonl`.
* Settlement entries also appear in `trades.jsonl` with `status="settled"`.

## 5. Secret Rotation

1. Generate new `TELEGRAM_BOT_TOKEN` and `TELEGRAM_SECRET_TOKEN`.
2. Update environment variables (Render dashboard or `.env`).
3. Redeploy the service.
4. Run `scripts/register_webhook_example.sh` to register the new webhook URL.

## 6. Incident Response

1. **Identify** – Monitor metrics, health checks, and logs for anomalies.
2. **Contain** – Disable the Telegram webhook (`deleteWebhook`) if secrets are compromised.
3. **Remediate** – Rotate secrets, verify PAPER mode enforcement, redeploy.
4. **Review** – Audit ledger entries to confirm no unauthorized actions were recorded.

## 7. Testing and Releases

* Run `pytest -q` prior to deployment.
* Review `docs/RELEASE_NOTES.md` for change history.
* Update documentation alongside code changes for operational clarity.

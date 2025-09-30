# Risk Controls

This PAPER service enforces conservative limits to simulate disciplined trading operations.

## Daily Loss Cap

* `DAILY_LOSS_LIMIT` defines the maximum cumulative realized loss allowed per day.
* Each incoming trade carries a `realized_pnl` value (positive or negative).
* If the running total for a day would fall below `-DAILY_LOSS_LIMIT`, the trade is rejected (HTTP 423) and logged with `reason="daily-loss-cap"`.

## Reserve Allocation

* End-of-day settlement transfers **2%** of positive daily realized profit to a reserve ledger.
* Settlement appends two entries:
  * `reserve.jsonl` – reserve transfer (`day`, `transfer`, `source_profit`).
  * `trades.jsonl` – settlement summary with `status="settled"`.
* Trigger settlement manually or via scheduled automation using `RiskManager.end_of_day_settlement()`.

## Ledgers

* Location: `DATA_DIR/ledgers/`
* Files: `trades.jsonl` and `reserve.jsonl`
* Format: JSON Lines with ISO-8601 timestamps (UTC).
* Behavior: Append-only; entries are never modified or deleted by the service.

## Mode Enforcement

* PAPER mode requires `PAPER_MODE=true` and `LIVE=false`.
* Startup validation fails if `LIVE=true`, causing `/healthz` to report HTTP 500.
* All trading actions remain simulated; no external execution occurs.

## Audit Recommendations

* Review ledgers regularly for anomalies and repeated risk rejections.
* Compare reserve transfers against daily profit to ensure the 2% rule executes.
* Archive ledgers periodically for historical analysis and compliance records.

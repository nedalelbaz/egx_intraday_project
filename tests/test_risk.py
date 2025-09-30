from app.ledger import LedgerStore
from app.risk import RiskManager, current_day


def test_daily_loss_cap(tmp_path):
    store = LedgerStore(str(tmp_path))
    manager = RiskManager(daily_loss_limit=1000, ledger_store=store)

    accepted = manager.process_trade({"trade_id": "t1", "realized_pnl": -400})
    assert accepted is True

    rejected = manager.process_trade({"trade_id": "t2", "realized_pnl": -700})
    assert rejected is False

    trades = store.read_trades()
    accepted_entry = next(entry for entry in trades if entry["trade_id"] == "t1")
    rejected_entry = next(entry for entry in trades if entry.get("status") == "rejected")

    assert accepted_entry["status"] == "accepted"
    assert rejected_entry["reason"] == "daily-loss-cap"


def test_end_of_day_settlement(tmp_path):
    store = LedgerStore(str(tmp_path))
    manager = RiskManager(daily_loss_limit=1000, ledger_store=store)
    today = current_day()

    manager.process_trade({"trade_id": "t3", "realized_pnl": 1500, "day": today})
    result = manager.end_of_day_settlement(today)

    assert result is not None
    assert result["transfer"] == 30.0  # 2% of 1500
    reserve_entries = store.read_reserve()
    assert reserve_entries[-1]["transfer"] == 30.0
    trades = store.read_trades()
    settlement_entry = next(entry for entry in trades if entry.get("status") == "settled")
    assert settlement_entry["transfer"] == 30.0

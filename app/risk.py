"""Risk management primitives for the PAPER webhook."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, Optional

from .ledger import LedgerStore


def current_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class RiskManager:
    """Apply daily loss caps and manage reserve transfers."""

    def __init__(self, daily_loss_limit: float, ledger_store: LedgerStore) -> None:
        self.daily_loss_limit = daily_loss_limit
        self.ledger_store = ledger_store
        self._lock = threading.Lock()
        self._daily_realized: Dict[str, float] = {}

    def process_trade(self, trade: Dict[str, float | str]) -> bool:
        """Validate and record a trade instruction.

        Returns ``True`` when the trade is accepted, ``False`` when rejected.
        """

        realized = float(trade.get("realized_pnl", 0.0))
        day = str(trade.get("day") or current_day())

        entry = {
            "day": day,
            "realized_pnl": realized,
            "trade_id": trade.get("trade_id"),
            "status": "accepted",
        }

        with self._lock:
            running = self._daily_realized.get(day, 0.0)
            prospective = running + realized
            if prospective < -self.daily_loss_limit:
                entry["status"] = "rejected"
                entry["reason"] = "daily-loss-cap"
                self.ledger_store.record_trade(entry)
                return False
            self._daily_realized[day] = prospective
            self.ledger_store.record_trade(entry)
            return True

    def end_of_day_settlement(self, day: Optional[str] = None) -> Optional[dict[str, float | str]]:
        """Transfer 2% of positive daily realized profit to the reserve ledger."""

        day_key = day or current_day()
        with self._lock:
            realized = self._daily_realized.get(day_key, 0.0)
            if realized <= 0:
                return None
            transfer_amount = round(realized * 0.02, 2)
            reserve_entry = {
                "day": day_key,
                "transfer": transfer_amount,
                "source_profit": realized,
            }
            self.ledger_store.record_reserve_transfer(reserve_entry)
            settlement_entry = {
                "day": day_key,
                "status": "settled",
                "transfer": transfer_amount,
                "source_profit": realized,
            }
            self.ledger_store.record_trade(settlement_entry)
            return reserve_entry

    def get_daily_realized(self, day: Optional[str] = None) -> float:
        day_key = day or current_day()
        with self._lock:
            return self._daily_realized.get(day_key, 0.0)

"""Append-only JSONL ledger utilities."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


class LedgerStore:
    """Manage append-only ledgers for trades and reserves."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.ledger_dir = self.base_dir / "ledgers"
        self.trade_path = self.ledger_dir / "trades.jsonl"
        self.reserve_path = self.ledger_dir / "reserve.jsonl"
        self._lock = threading.Lock()
        self._initialise_files()

    def _initialise_files(self) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.trade_path, self.reserve_path):
            if not path.exists():
                path.touch()

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        payload = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def record_trade(self, entry: dict[str, Any]) -> None:
        record = {"timestamp": utc_now_iso(), **entry}
        self._append(self.trade_path, record)

    def record_reserve_transfer(self, entry: dict[str, Any]) -> None:
        record = {"timestamp": utc_now_iso(), **entry}
        self._append(self.reserve_path, record)

    def read_trades(self) -> List[dict[str, Any]]:
        return list(_read_jsonl(self.trade_path))

    def read_reserve(self) -> List[dict[str, Any]]:
        return list(_read_jsonl(self.reserve_path))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[dict[str, Any]]:
    entries: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries

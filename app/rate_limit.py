"""Simple in-memory token bucket rate limiter."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class TokenBucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """Token bucket limiter keyed by arbitrary identifier."""

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate = rate
        self.burst = burst
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True if the request is within limits for ``key``."""

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(tokens=float(self.burst), last_refill=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_refill)
                bucket.tokens = min(float(self.burst), bucket.tokens + elapsed * self.rate)
                bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

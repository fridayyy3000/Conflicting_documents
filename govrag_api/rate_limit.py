"""Simple in-memory sliding-window rate limiter (single-process, demo-scale only)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()

            if len(hits) >= self.max_requests:
                return False

            hits.append(now)
            return True

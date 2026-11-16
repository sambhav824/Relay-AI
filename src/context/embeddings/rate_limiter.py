"""Relay AI — Provider-Aware In-Process Rate Limiter.

Prevents sending excessive concurrent requests to embedding APIs.
Supports thread-safe concurrent access, cancellation, and deterministic testing.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Callable, Deque, Optional

logger = logging.getLogger("RelayAI.RateLimiter")


class RateLimitCancelledError(Exception):
    """Raised when waiting for a rate limiter slot is cancelled."""
    pass


class RateLimiter:
    """Thread-safe sliding-window request rate limiter.

    Tracks timestamps of recent requests within a 60-second window to enforce
    requests-per-minute (RPM) limits across concurrent threads.
    Guarantees the lock is never held while sleeping.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be positive, got {requests_per_minute}.")

        self.requests_per_minute = requests_per_minute
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, cancellation_event: Optional[threading.Event] = None) -> float:
        """Block if necessary until a request slot is available.

        Releases internal lock before sleeping so concurrent callers do not block.
        Checks cancellation before and after sleep.

        Returns:
            The total duration waited in seconds (0.0 if slot was immediately available).
        """
        total_waited = 0.0

        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                raise RateLimitCancelledError("Rate limiter wait cancelled by caller.")

            with self._lock:
                now = self.time_fn()
                window_start = now - 60.0

                # Purge timestamps outside the 60-second rolling window
                while self._timestamps and self._timestamps[0] <= window_start:
                    self._timestamps.popleft()

                # If slot is available, claim it immediately inside the lock
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return total_waited

                # Capacity reached: calculate minimum wait time
                earliest = self._timestamps[0]
                wait_time = max(0.001, (earliest + 60.0) - now)

            # --- Lock is explicitly RELEASED here before sleeping ---
            if cancellation_event is not None and cancellation_event.is_set():
                raise RateLimitCancelledError("Rate limiter wait cancelled before sleep.")

            logger.debug(
                "Rate limit reached (%d RPM). Throttling for %.3fs (lock released)",
                self.requests_per_minute,
                wait_time,
            )

            self.sleep_fn(wait_time)
            total_waited += wait_time

            if cancellation_event is not None and cancellation_event.is_set():
                raise RateLimitCancelledError("Rate limiter wait cancelled after sleep.")

    def reset(self) -> None:
        """Clear all tracked timestamps thread-safely."""
        with self._lock:
            self._timestamps.clear()

    @property
    def current_load(self) -> int:
        """Return the number of requests executed in the last 60 seconds."""
        with self._lock:
            now = self.time_fn()
            window_start = now - 60.0
            while self._timestamps and self._timestamps[0] <= window_start:
                self._timestamps.popleft()
            return len(self._timestamps)

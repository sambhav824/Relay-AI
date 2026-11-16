"""Relay AI — Resilient Retry and Exponential Backoff for Embedding Generation.

Handles HTTP 429 rate limits, HTTP 5xx server errors, connection drops,
and timeouts with configurable exponential backoff, jitter, and Retry-After parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import email.utils
import logging
import random
import re
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type

from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Backward-compatibility aliases
ProviderAuthError = ProviderAuthenticationError
ProviderNotFoundError = ModelNotFoundError

logger = logging.getLogger("RelayAI.Retry")


class RetryAfterExceededError(ProviderRateLimitError):
    """Raised when server Retry-After duration exceeds maximum permitted synchronous wait."""

    def __init__(self, message: str, retry_after: float, max_delay: float, provider_id: str = ""):
        super().__init__(message, provider_id=provider_id)
        self.retry_after = retry_after
        self.max_delay = max_delay


# Non-retryable errors: client errors that will never succeed on immediate retry
NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ProviderAuthenticationError,
    ProviderAuthError,
    ModelNotFoundError,
    ProviderNotFoundError,
    ValueError,
    TypeError,
    KeyError,
)

# Retryable transient errors
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ProviderRateLimitError,
    ProviderConnectionError,
    ProviderTimeoutError,
)


def extract_retry_after(exc: Exception, now_fn: Optional[Callable[[], float]] = None) -> Optional[float]:
    """Extract Retry-After duration in seconds from exception attributes or message.

    Supports:
    1. Integer or float seconds on `exc.retry_after`
    2. Response headers `Retry-After` (both numeric seconds and RFC 7231 / RFC 2822 HTTP-date)
    3. Regular expression matches on error messages (e.g. 'Retry-After: 12.5s' or 'try again in 5s')
    """
    # 1. Attribute on exception object
    if hasattr(exc, "retry_after") and isinstance(exc.retry_after, (int, float)):
        return max(0.0, float(exc.retry_after))

    # 2. Look for Retry-After in response headers attribute if available
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        headers = response.headers
        retry_val = headers.get("retry-after") or headers.get("Retry-After")
        if retry_val:
            parsed = parse_retry_after_header(str(retry_val), now_fn=now_fn)
            if parsed is not None:
                return parsed

    # 3. Regular expression match from error message
    msg = str(exc)
    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            pass

    match_sec = re.search(r"please try again in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if match_sec:
        try:
            return max(0.0, float(match_sec.group(1)))
        except ValueError:
            pass

    return None


def parse_retry_after_header(value: str, now_fn: Optional[Callable[[], float]] = None) -> Optional[float]:
    """Parse HTTP Retry-After header value as either numeric seconds or HTTP-date string."""
    val_str = value.strip()
    # 1. Try numeric seconds
    try:
        sec = float(val_str)
        return max(0.0, sec)
    except ValueError:
        pass

    # 2. Try RFC 7231 / RFC 2822 HTTP date
    try:
        target_dt = email.utils.parsedate_to_datetime(val_str)
        if target_dt is not None:
            now_ts = now_fn() if now_fn is not None else time.time()
            now_dt = datetime.datetime.fromtimestamp(now_ts, tz=datetime.timezone.utc)
            delta = (target_dt - now_dt).total_seconds()
            return max(0.0, float(delta))
    except Exception:
        pass

    return None


@dataclass
class RetryConfig:
    """Configuration for exponential backoff and retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter_ratio: float = 0.2  # +/- 20%
    retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS
    non_retryable_exceptions: Tuple[Type[Exception], ...] = NON_RETRYABLE_EXCEPTIONS
    sleep_fn: Callable[[float], None] = time.sleep
    random_fn: Callable[[float, float], float] = random.uniform
    now_fn: Optional[Callable[[], float]] = None

    def is_retryable(self, exc: Exception) -> bool:
        """Determine whether an exception should be retried."""
        if isinstance(exc, self.non_retryable_exceptions):
            return False

        # Check for explicit non-retryable keywords or HTTP error status codes
        msg = str(exc).lower()
        non_retryable_indicators = (
            "401",
            "403",
            "404",
            "unauthorized",
            "authentication",
            "forbidden",
            "invalid api key",
            "model not found",
            "not found",
            "bad request",
        )
        if any(ind in msg for ind in non_retryable_indicators):
            return False

        if isinstance(exc, self.retryable_exceptions):
            return True

        # Check for transient HTTP error status codes or substrings in ProviderError
        transient_indicators = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "timeout",
            "connection",
            "overloaded",
            "service unavailable",
        )
        if any(code in msg for code in transient_indicators):
            return True

        return False

    def compute_delay(self, attempt: int, exc: Optional[Exception] = None) -> float:
        """Compute the backoff delay with jitter in seconds.

        Guarantees:
        - Never retries earlier than a valid server-provided Retry-After value.
        - Raises `RetryAfterExceededError` if Retry-After exceeds `max_delay`.
        - When Retry-After is supplied, jitter only adds delay (never subtracts below requested wait).
        """
        retry_after = extract_retry_after(exc, now_fn=self.now_fn) if exc is not None else None

        if retry_after is not None and retry_after > 0:
            if retry_after > self.max_delay:
                raise RetryAfterExceededError(
                    f"Retry-After ({retry_after:.1f}s) exceeds maximum permitted synchronous wait ({self.max_delay:.1f}s).",
                    retry_after=retry_after,
                    max_delay=self.max_delay,
                )
            # Never retry earlier than requested: add jitter only upwards
            jitter = abs(self.random_fn(0.0, self.jitter_ratio * retry_after))
            return retry_after + jitter

        # Exponential backoff: base_delay * 2^attempt
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))

        # Add configurable jitter (+/- jitter_ratio)
        jitter = self.random_fn(-self.jitter_ratio * delay, self.jitter_ratio * delay)
        return max(0.0, delay + jitter)

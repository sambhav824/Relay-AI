from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


class UsageLimitExceeded(Exception):
    """Raised when a configured request or spending limit would be exceeded."""


@dataclass
class UsageLimits:
    """User-configurable guardrails; zero disables a cost/request-count limit."""

    max_input_tokens: int = 8192
    max_output_tokens: int = 1024
    max_request_cost_usd: float = 0.0
    daily_budget_usd: float = 0.0
    monthly_budget_usd: float = 0.0
    max_requests_per_day: int = 0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    warning_fraction: float = 0.8

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]]) -> "UsageLimits":
        limits = cls()
        if values:
            for name in asdict(limits):
                if name in values:
                    setattr(limits, name, type(getattr(limits, name))(values[name]))
        limits.max_input_tokens = max(1, limits.max_input_tokens)
        limits.max_output_tokens = max(1, limits.max_output_tokens)
        limits.warning_fraction = min(1.0, max(0.0, limits.warning_fraction))
        for name in (
            "max_request_cost_usd",
            "daily_budget_usd",
            "monthly_budget_usd",
            "input_cost_per_million",
            "output_cost_per_million",
        ):
            setattr(limits, name, max(0.0, getattr(limits, name)))
        limits.max_requests_per_day = max(0, limits.max_requests_per_day)
        return limits


def estimate_input_tokens(messages: Iterable[Union[Dict[str, Any], Any]]) -> int:
    """Conservative, tokenizer-independent estimate used only for preflight limits."""
    characters = 0
    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        characters += len(str(content))
    return math.ceil(characters / 4)


class UsageTracker:
    """Thread-safe, local usage ledger with per-user daily/monthly accounting."""

    def __init__(self, database_path: Optional[Union[str, Path]] = None):
        if database_path is None:
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
            database_path = Path(base) / "RelayAI" / "usage.sqlite3"
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_usage (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens_est INTEGER NOT NULL,
                    output_tokens_limit INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    actual_input_tokens INTEGER,
                    actual_output_tokens INTEGER,
                    actual_cost REAL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.database_path), timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, limits: UsageLimits) -> Optional[float]:
        if limits.input_cost_per_million <= 0 or limits.output_cost_per_million <= 0:
            return None
        return (
            input_tokens * limits.input_cost_per_million
            + output_tokens * limits.output_cost_per_million
        ) / 1_000_000

    def begin_request(
        self,
        user_id: str,
        model: str,
        messages: Iterable[Union[Dict[str, Any], Any]],
        output_limit: int,
        limits: UsageLimits,
    ) -> str:
        input_tokens = estimate_input_tokens(messages)
        if input_tokens > limits.max_input_tokens:
            raise UsageLimitExceeded(
                f"Estimated input ({input_tokens} tokens) exceeds the configured "
                f"per-request limit ({limits.max_input_tokens} tokens)."
            )

        request_cost = self.estimate_cost(input_tokens, output_limit, limits)
        cost_controls_enabled = any(
            (limits.max_request_cost_usd, limits.daily_budget_usd, limits.monthly_budget_usd)
        )
        if cost_controls_enabled and request_cost is None:
            raise UsageLimitExceeded(
                "Configure input and output prices per million tokens before enabling USD budgets."
            )
        request_cost = request_cost or 0.0
        if limits.max_request_cost_usd and request_cost > limits.max_request_cost_usd:
            raise UsageLimitExceeded("Estimated request cost exceeds the configured per-request budget.")

        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        request_id = str(uuid.uuid4())
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT
                    SUM(CASE WHEN started_at >= ? THEN 1 ELSE 0 END) AS daily_requests,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_cost, estimated_cost) ELSE 0 END) AS daily_cost,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_cost, estimated_cost) ELSE 0 END) AS monthly_cost
                FROM ai_usage WHERE user_id = ?
                """,
                (day_start, day_start, month_start, user_id),
            ).fetchone()
            if limits.max_requests_per_day and (row["daily_requests"] or 0) >= limits.max_requests_per_day:
                raise UsageLimitExceeded("The configured daily request limit has been reached.")
            if limits.daily_budget_usd and (row["daily_cost"] or 0) + request_cost > limits.daily_budget_usd:
                raise UsageLimitExceeded("This request would exceed the configured daily budget.")
            if limits.monthly_budget_usd and (row["monthly_cost"] or 0) + request_cost > limits.monthly_budget_usd:
                raise UsageLimitExceeded("This request would exceed the configured monthly budget.")
            db.execute(
                """
                INSERT INTO ai_usage
                    (request_id, user_id, started_at, model, input_tokens_est,
                     output_tokens_limit, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, user_id, now.isoformat(), model, input_tokens, output_limit, request_cost),
            )
        return request_id

    def complete_request(
        self,
        request_id: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        limits: UsageLimits,
    ) -> None:
        actual_cost = None
        if input_tokens is not None and output_tokens is not None:
            actual_cost = self.estimate_cost(input_tokens, output_tokens, limits)
        with self._lock, self._connect() as db:
            db.execute(
                """
                UPDATE ai_usage
                SET actual_input_tokens = ?, actual_output_tokens = ?, actual_cost = ?
                WHERE request_id = ?
                """,
                (input_tokens, output_tokens, actual_cost, request_id),
            )

    def summary(self, user_id: str, limits: Optional[UsageLimits] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self._lock, self._connect() as db:
            row = db.execute(
                """
                SELECT
                    SUM(CASE WHEN started_at >= ? THEN 1 ELSE 0 END) AS daily_requests,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_cost, estimated_cost) ELSE 0 END) AS daily_cost,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_cost, estimated_cost) ELSE 0 END) AS monthly_cost,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_input_tokens, input_tokens_est) ELSE 0 END) AS daily_input_tokens,
                    SUM(CASE WHEN started_at >= ? THEN COALESCE(actual_output_tokens, 0) ELSE 0 END) AS daily_output_tokens,
                    SUM(CASE WHEN started_at >= ? THEN actual_input_tokens END) AS daily_actual_input_tokens,
                    SUM(CASE WHEN started_at >= ? THEN actual_output_tokens END) AS daily_actual_output_tokens
                FROM ai_usage WHERE user_id = ?
                """,
                (day_start, day_start, month_start, day_start, day_start, day_start, day_start, user_id),
            ).fetchone()
        result = {key: (row[key] or 0) for key in row.keys()}
        result["warning"] = False
        if limits:
            result["warning"] = bool(
                (limits.daily_budget_usd and result["daily_cost"] >= limits.daily_budget_usd * limits.warning_fraction)
                or (limits.monthly_budget_usd and result["monthly_cost"] >= limits.monthly_budget_usd * limits.warning_fraction)
                or (limits.max_requests_per_day and result["daily_requests"] >= limits.max_requests_per_day * limits.warning_fraction)
            )
        return result

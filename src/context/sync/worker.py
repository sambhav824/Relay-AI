"""Background Supabase PostgREST sync worker."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from urllib.parse import quote

import httpx
from PySide6.QtCore import QThread, Signal

from src.context.auth.manager import AuthManager
from src.context.sync.queue import SyncQueueManager

logger = logging.getLogger(__name__)

SYNC_TABLES = {
    "profiles", "projects", "conversations", "messages", "memories",
    "documents", "document_chunks", "smart_routing_logs",
    "model_comparisons", "comparison_slots", "team_runs", "team_step_records",
}


class SyncWorker(QThread):
    """Uploads queued mutations without blocking the Qt application thread.

    The worker uses the signed-in user's access token only. It never accepts or
    stores a Supabase service-role key.
    """

    item_synced = Signal(str)
    item_failed = Signal(str, str)
    queue_changed = Signal(int)
    sync_error = Signal(str)

    def __init__(
        self,
        queue: SyncQueueManager,
        auth_manager: AuthManager,
        supabase_url: str,
        supabase_anon_key: str,
        *,
        poll_interval: float = 60.0,
        batch_size: int = 20,
        http_client: Optional[httpx.Client] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not supabase_url or not supabase_anon_key:
            raise ValueError("Supabase URL and anon key are required")
        if poll_interval <= 0 or batch_size < 1:
            raise ValueError("poll_interval and batch_size must be positive")
        self.queue = queue
        self.auth_manager = auth_manager
        self.supabase_url = supabase_url.rstrip("/")
        self.anon_key = supabase_anon_key
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._client = http_client or httpx.Client(timeout=20.0)
        self._owns_client = http_client is None
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._stop_event.set()
        self.requestInterruption()

    def close(self) -> None:
        self.stop()
        if self._owns_client:
            self._client.close()

    def run(self) -> None:
        """Poll and process batches until stopped."""
        while not self._stop_event.is_set() and not self.isInterruptionRequested():
            try:
                self.run_once()
            except Exception as exc:  # keep the worker alive after recoverable failures
                logger.exception("Sync cycle failed")
                self.sync_error.emit(str(exc))
            self._stop_event.wait(self.poll_interval)

    def run_once(self) -> int:
        """Process one batch and return the number of successful uploads."""
        session = self.auth_manager.session
        if session is None or session.user is None:
            return 0
        user_id = session.user.id
        if not user_id:
            return 0

        # get_access_token refreshes expiring sessions through AuthManager.
        token = self.auth_manager.get_access_token()
        if not token:
            return 0

        items = self.queue.claim_batch(self.batch_size)
        completed = 0
        for item in items:
            if self._stop_event.is_set() or self.isInterruptionRequested():
                break
            if item["user_id"] != user_id:
                self.queue.mark_failed(item["id"], "Queue owner does not match active account", retryable=False)
                self.item_failed.emit(item["id"], "Queue owner does not match active account")
                continue
            try:
                self._send_item(item, token, user_id)
                self.queue.mark_completed(item["id"])
                completed += 1
                self.item_synced.emit(item["id"])
            except httpx.RequestError as exc:
                self.queue.mark_failed(item["id"], str(exc), retryable=True)
                self.item_failed.emit(item["id"], str(exc))
                self._backoff(item["retry_count"])
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status == 429 or status >= 500
                detail = f"Supabase returned HTTP {status}: {exc.response.text[:500]}"
                self.queue.mark_failed(item["id"], detail, retryable=retryable)
                self.item_failed.emit(item["id"], detail)
                if retryable:
                    self._backoff(item["retry_count"])
            except (ValueError, TypeError) as exc:
                self.queue.mark_failed(item["id"], str(exc), retryable=False)
                self.item_failed.emit(item["id"], str(exc))
        self.queue_changed.emit(self.queue.pending_count())
        return completed

    def _send_item(self, item: dict, token: str, user_id: str) -> None:
        table = item["table_name"]
        if table not in SYNC_TABLES:
            raise ValueError(f"Table is not enabled for synchronization: {table}")
        base = f"{self.supabase_url}/rest/v1/{quote(table, safe='')}"
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        operation = item["operation"]
        record_id = item["record_id"]
        if operation in {"INSERT", "UPDATE"}:
            payload = dict(item["payload"])
            payload["id"] = record_id
            if "user_id" in payload and payload["user_id"] != user_id:
                raise ValueError("Payload user_id does not match active account")
            payload["user_id"] = user_id

        if operation == "INSERT":
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
            response = self._client.post(base, json=payload, headers=headers)
        elif operation == "UPDATE":
            headers["Prefer"] = "return=minimal"
            response = self._client.patch(
                base, params={"id": f"eq.{record_id}", "user_id": f"eq.{user_id}"},
                json=payload, headers=headers,
            )
        elif operation == "DELETE":
            headers["Prefer"] = "return=minimal"
            response = self._client.delete(
                base, params={"id": f"eq.{record_id}", "user_id": f"eq.{user_id}"},
                headers=headers,
            )
        else:
            raise ValueError(f"Unsupported sync operation: {operation}")
        response.raise_for_status()

    def _backoff(self, retry_count: int) -> None:
        delay = min(30.0, 2.0 ** max(0, retry_count))
        self._stop_event.wait(delay)

"""Integration tests for Milestone 10.2: Cloud Sync, Outbox, and UI Integration."""

from __future__ import annotations

import json
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

import httpx
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from src.context.auth.client import AuthSession, AuthUser
from src.context.auth.manager import AuthManager
from src.context.db.connection import initialize_database
from src.context.repository import ContextRepository
from src.context.sync.queue import SyncQueueManager
from src.context.sync.worker import SyncWorker
from src.ui.main_window import MainWindow

# Ensure QApplication exists for Qt tests
_app = QApplication.instance() or QApplication([])


class TestSyncIntegration(unittest.TestCase):
    """Integration test suite verifying local mutations -> outbox -> sync worker -> UI."""

    def setUp(self) -> None:
        self.db = initialize_database(":memory:")
        self.conn = self.db.get_connection()
        self.queue = SyncQueueManager(self.conn)
        self.user_id = "00000000-0000-0000-0000-000000000001"
        self.repo = ContextRepository(self.conn, sync_queue=self.queue, default_user_id=self.user_id)

        # Mock AuthManager with active session
        self.auth_manager = MagicMock(spec=AuthManager)
        self.mock_user = AuthUser(id=self.user_id, email="tester@example.com")
        self.mock_session = AuthSession(
            access_token="mock_valid_jwt_token",
            refresh_token="mock_refresh_token",
            expires_in=3600,
            expires_at=9999999999,
            user=self.mock_user,
        )
        self.auth_manager.session = self.mock_session
        self.auth_manager.get_access_token.return_value = "mock_valid_jwt_token"

        self.supabase_url = "https://mockproject.supabase.co"
        self.anon_key = "mock_anon_key"

    def tearDown(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

    def test_local_mutation_enqueues_outbox(self) -> None:
        """Verify that local repository mutations atomically insert outbox records."""
        self.assertEqual(self.queue.pending_count(), 0)

        # 1. Create conversation
        conv_id = self.repo.create_conversation(title="Project Alpha Discussion", model_id="gpt-4o")
        self.assertIsNotNone(conv_id)
        self.assertEqual(self.queue.pending_count(), 1)

        # 2. Add message
        msg_id = self.repo.add_message(
            conversation_id=conv_id,
            role="user",
            content="Hello cloud sync outbox",
            token_count=12,
        )
        self.assertIsNotNone(msg_id)
        self.assertEqual(self.queue.pending_count(), 2)

        # 3. Add memory
        mem_id = self.repo.add_memory(category="USER_PREFERENCE", content="Prefers dark theme")
        self.assertIsNotNone(mem_id)
        self.assertEqual(self.queue.pending_count(), 3)

        # Inspect queue batch
        items = self.queue.claim_batch(10)
        self.assertEqual(len(items), 3)

        tables = [item["table_name"] for item in items]
        self.assertEqual(tables, ["conversations", "messages", "memories"])
        self.assertEqual(items[0]["record_id"], conv_id)
        self.assertEqual(items[0]["payload"]["title"], "Project Alpha Discussion")
        self.assertEqual(items[1]["record_id"], msg_id)
        self.assertEqual(items[1]["payload"]["content"], "Hello cloud sync outbox")

    def test_outbox_to_sync_worker_success(self) -> None:
        """Verify SyncWorker processes outbox entries and marks them completed."""
        self.repo.create_conversation(title="Sync Worker Test")
        self.assertEqual(self.queue.pending_count(), 1)

        captured_requests = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(201, json={"status": "ok"})

        transport = httpx.MockTransport(mock_handler)
        client = httpx.Client(transport=transport)

        worker = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            http_client=client,
        )

        completed = worker.run_once()
        self.assertEqual(completed, 1)
        self.assertEqual(self.queue.pending_count(), 0)
        self.assertEqual(len(captured_requests), 1)

        req = captured_requests[0]
        self.assertEqual(req.method, "POST")
        self.assertIn("/rest/v1/conversations", str(req.url))
        self.assertEqual(req.headers["Authorization"], "Bearer mock_valid_jwt_token")
        self.assertEqual(req.headers["apikey"], self.anon_key)
        self.assertIn("resolution=merge-duplicates", req.headers.get("Prefer", ""))

        body = json.loads(req.content.decode("utf-8"))
        self.assertEqual(body["user_id"], self.user_id)
        self.assertEqual(body["title"], "Sync Worker Test")

    def test_failed_synchronization_and_retry(self) -> None:
        """Verify transient network failures are retried and permanent failures can be requeued."""
        self.repo.create_conversation(title="Retry Test")

        # 1. Transient failure (500 Server Error)
        def fail_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client_fail = httpx.Client(transport=httpx.MockTransport(fail_handler))
        worker = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            http_client=client_fail,
        )

        completed = worker.run_once()
        self.assertEqual(completed, 0)
        # Should remain pending with retry_count incremented
        self.assertEqual(self.queue.pending_count(), 1)

        # 2. Permanent client error (400 Bad Request)
        def bad_req_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Invalid schema column")

        client_bad = httpx.Client(transport=httpx.MockTransport(bad_req_handler))
        worker_bad = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            http_client=client_bad,
        )

        worker_bad.run_once()
        # Item failed permanently
        self.assertEqual(self.queue.pending_count(), 0)
        failed_items = self.queue.failed_items()
        self.assertEqual(len(failed_items), 1)
        self.assertIn("HTTP 400", failed_items[0]["last_error"])

        # 3. Manual retry
        requeued = self.queue.retry_failed()
        self.assertEqual(requeued, 1)
        self.assertEqual(self.queue.pending_count(), 1)

    def test_duplicate_operation_prevention(self) -> None:
        """Verify mutations for the same record are coalesced to prevent duplicate entries."""
        conv_id = "test-conv-dedup"
        self.repo.create_conversation(title="Initial Title", conversation_id=conv_id)
        self.assertEqual(self.queue.pending_count(), 1)

        # Update while pending in queue: should coalesce into single INSERT with updated payload
        self.repo.update_conversation(conversation_id=conv_id, title="Updated Title")
        self.assertEqual(self.queue.pending_count(), 1)

        items = self.queue.claim_batch(10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["operation"], "INSERT")
        self.assertEqual(items[0]["payload"]["title"], "Updated Title")

    def test_network_disconnection_and_recovery(self) -> None:
        """Verify worker wakes up immediately on network recovery via trigger_sync."""
        self.repo.create_conversation(title="Network Test")

        is_online = False
        captured = []

        def intermittent_handler(request: httpx.Request) -> httpx.Response:
            if not is_online:
                raise httpx.ConnectError("No internet connection")
            captured.append(request)
            return httpx.Response(201, json={"status": "ok"})

        client = httpx.Client(transport=httpx.MockTransport(intermittent_handler))
        worker = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            poll_interval=300.0,
            http_client=client,
        )

        # Attempt 1: Offline
        worker.run_once()
        self.assertEqual(len(captured), 0)
        self.assertEqual(self.queue.pending_count(), 1)

        # Network Restored!
        is_online = True
        worker.trigger_sync()

        # Run once again
        worker.run_once()
        self.assertEqual(len(captured), 1)
        self.assertEqual(self.queue.pending_count(), 0)

    def test_user_level_isolation(self) -> None:
        """Verify worker strictly rejects queuing or uploading items belonging to other users."""
        other_user = "99999999-9999-9999-9999-999999999999"
        self.queue.stage(
            user_id=other_user,
            table_name="conversations",
            record_id="foreign_conv_1",
            operation="INSERT",
            payload={"title": "Foreign Data"},
        )

        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        worker = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            http_client=client,
        )

        completed = worker.run_once()
        self.assertEqual(completed, 0)

        # The item must have failed permanently due to ownership mismatch
        failed = self.queue.failed_items()
        self.assertEqual(len(failed), 1)
        self.assertIn("does not match active account", failed[0]["last_error"])

    def test_ui_sync_status_and_lifecycle(self) -> None:
        """Verify UI shows sync status and cleans up worker safely on window close."""
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(201)))
        worker = SyncWorker(
            queue=self.queue,
            auth_manager=self.auth_manager,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            http_client=client,
        )

        window = MainWindow(
            sync_worker=worker,
            auth_manager=self.auth_manager,
            sync_queue=self.queue,
            context_repo=self.repo,
        )

        # Button exists and initial status is Synced
        self.assertTrue(hasattr(window, "sync_status_btn"))
        self.assertIn("Synced", window.sync_status_btn.text())

        # When queue has pending items, UI updates
        self.repo.create_conversation(title="UI Test")
        window._on_sync_queue_changed(1)
        self.assertIn("Syncing", window.sync_status_btn.text())

        # When error occurs, UI updates
        window._on_sync_error("Network timeout")
        self.assertIn("Sync Error", window.sync_status_btn.text())

        # Test closeEvent cooperative shutdown
        close_event = QCloseEvent()
        window.closeEvent(close_event)
        self.assertTrue(window._is_shutting_down)
        self.assertTrue(worker._stop_event.is_set())

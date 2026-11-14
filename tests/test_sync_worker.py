"""Unit tests for the authenticated Supabase background sync worker."""

import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.context.sync.queue import SyncQueueManager
from src.context.sync.worker import SyncWorker


class FakeResponse:
    def __init__(self, status_code=204, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            request = httpx.Request("POST", "https://example.supabase.co/rest/v1/projects")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


class TestSyncWorker(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.queue = SyncQueueManager(self.conn)
        self.auth = Mock()
        self.auth.session = SimpleNamespace(user=SimpleNamespace(id="user-1"))
        self.auth.get_access_token.return_value = "user-jwt"
        self.http = Mock()
        self.http.post.return_value = FakeResponse()
        self.http.patch.return_value = FakeResponse()
        self.http.delete.return_value = FakeResponse()
        self.worker = SyncWorker(
            self.queue, self.auth, "https://example.supabase.co", "public-anon-key",
            http_client=self.http,
        )

    def tearDown(self):
        self.worker.close()
        self.conn.close()

    def test_insert_uses_user_jwt_and_owner_id(self):
        queue_id = self.queue.stage(
            user_id="user-1", table_name="projects", record_id="project-1",
            operation="INSERT", payload={"name": "Alpha"},
        )
        self.assertEqual(self.worker.run_once(), 1)
        self.http.post.assert_called_once()
        args, kwargs = self.http.post.call_args
        self.assertEqual(args[0], "https://example.supabase.co/rest/v1/projects")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer user-jwt")
        self.assertEqual(kwargs["headers"]["apikey"], "public-anon-key")
        self.assertEqual(kwargs["json"]["id"], "project-1")
        self.assertEqual(kwargs["json"]["user_id"], "user-1")
        self.assertEqual(self.queue.pending_count(), 0)

    def test_queue_item_for_another_user_is_not_sent(self):
        self.queue.stage(
            user_id="other-user", table_name="projects", record_id="project-2",
            operation="INSERT", payload={"name": "Private"},
        )
        self.assertEqual(self.worker.run_once(), 0)
        self.http.post.assert_not_called()
        self.assertEqual(self.queue.failed_items()[0]["user_id"], "other-user")

    def test_server_error_is_retryable(self):
        self.http.post.return_value = FakeResponse(503, "temporarily unavailable")
        self.queue.stage(
            user_id="user-1", table_name="projects", record_id="project-3",
            operation="INSERT", payload={"name": "Retry"},
        )
        self.assertEqual(self.worker.run_once(), 0)
        row = self.conn.execute("SELECT status,retry_count FROM sync_queue").fetchone()
        self.assertEqual(row[0], "PENDING")
        self.assertEqual(row[1], 1)

    def test_client_error_is_permanent(self):
        self.http.post.return_value = FakeResponse(400, "invalid payload")
        self.queue.stage(
            user_id="user-1", table_name="projects", record_id="project-4",
            operation="INSERT", payload={"name": "Bad"},
        )
        self.assertEqual(self.worker.run_once(), 0)
        self.assertEqual(self.queue.failed_items()[0]["last_error"].split(":")[0], "Supabase returned HTTP 400")

    def test_no_active_session_does_not_claim_queue(self):
        self.auth.session = None
        self.queue.stage(
            user_id="user-1", table_name="projects", record_id="project-5",
            operation="INSERT", payload={"name": "Offline"},
        )
        self.assertEqual(self.worker.run_once(), 0)
        self.http.post.assert_not_called()
        self.assertEqual(self.queue.pending_count(), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the durable local Supabase synchronization outbox."""

import json
import sqlite3
import unittest

from src.context.sync.queue import SyncQueueManager


class TestSyncQueueManager(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.queue = SyncQueueManager(self.conn)

    def tearDown(self):
        self.conn.close()

    def stage(self, operation="INSERT", payload=None, record_id="r-1"):
        return self.queue.stage(
            user_id="user-1", table_name="projects", record_id=record_id,
            operation=operation, payload=payload or {"name": "Alpha"},
        )

    def rows(self):
        return self.conn.execute(
            "SELECT id,operation,payload_json,status,retry_count FROM sync_queue ORDER BY created_at,id"
        ).fetchall()

    def test_stage_insert(self):
        queue_id = self.stage()
        self.assertTrue(queue_id)
        row = self.rows()[0]
        self.assertEqual(row[1], "INSERT")
        self.assertEqual(json.loads(row[2])["name"], "Alpha")
        self.assertEqual(row[3], "PENDING")

    def test_pending_insert_then_update_coalesces_to_insert(self):
        self.stage("INSERT", {"name": "Alpha"})
        self.stage("UPDATE", {"name": "Beta"})
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "INSERT")
        self.assertEqual(json.loads(rows[0][2])["name"], "Beta")

    def test_pending_insert_then_delete_cancels(self):
        self.stage("INSERT")
        result = self.stage("DELETE", {})
        self.assertIsNone(result)
        self.assertEqual(self.rows(), [])

    def test_pending_update_then_delete_becomes_delete(self):
        self.stage("UPDATE", {"name": "Alpha"})
        self.stage("DELETE", {})
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "DELETE")

    def test_claim_batch_marks_in_flight(self):
        queue_id = self.stage()
        batch = self.queue.claim_batch(limit=10)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["id"], queue_id)
        self.assertEqual(batch[0]["payload"]["name"], "Alpha")
        self.assertEqual(self.rows()[0][3], "IN_FLIGHT")

    def test_completed_item_is_removed(self):
        queue_id = self.stage()
        self.queue.claim_batch()
        self.queue.mark_completed(queue_id)
        self.assertEqual(self.rows(), [])

    def test_retryable_failure_returns_to_pending(self):
        queue_id = self.stage()
        self.queue.claim_batch()
        self.queue.mark_failed(queue_id, "temporary network issue", retryable=True)
        row = self.rows()[0]
        self.assertEqual(row[3], "PENDING")
        self.assertEqual(row[4], 1)

    def test_permanent_failure_can_be_manually_retried(self):
        queue_id = self.stage()
        self.queue.claim_batch()
        self.queue.mark_failed(queue_id, "invalid row", retryable=False)
        self.assertEqual(self.rows()[0][3], "FAILED")
        self.assertEqual(self.queue.retry_failed(queue_id), 1)
        self.assertEqual(self.rows()[0][3], "PENDING")

    def test_stage_participates_in_outer_transaction(self):
        self.conn.execute("BEGIN")
        self.stage()
        self.conn.rollback()
        self.assertEqual(self.rows(), [])

    def test_new_change_can_queue_behind_in_flight_change(self):
        self.stage("INSERT", {"name": "Alpha"})
        self.queue.claim_batch()
        self.stage("UPDATE", {"name": "Beta"})
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row[1] for row in rows}, {"INSERT", "UPDATE"})

    def test_invalid_operation_rejected(self):
        with self.assertRaises(ValueError):
            self.stage("UPSERT")


if __name__ == "__main__":
    unittest.main()

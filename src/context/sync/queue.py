"""Durable SQLite outbox for synchronizing Relay AI records with Supabase."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional


class SyncQueueManager:
    """Stages local mutations in a transactional, deduplicating outbox.

    The caller supplies the same SQLite connection used by the local context DB.
    Queue changes use SAVEPOINTs so they participate in an outer transaction
    without committing it.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_queue (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('INSERT','UPDATE','DELETE')),
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 5,
                last_error TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING','IN_FLIGHT','FAILED','COMPLETED')),
                locked_until REAL,
                UNIQUE(user_id, table_name, record_id)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_queue_status_created "
            "ON sync_queue(status, created_at ASC)"
        )
        self.connection.commit()

    @staticmethod
    def _validate_identifier(value: str, label: str) -> str:
        if not value or not value.replace("_", "").isalnum():
            raise ValueError(f"Invalid {label}: {value!r}")
        return value

    def stage(
        self,
        *,
        user_id: str,
        table_name: str,
        record_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
    ) -> Optional[str]:
        """Stage or coalesce one mutation. Returns its queue id, or None if canceled."""
        self._validate_identifier(table_name, "table name")
        if not user_id or not record_id:
            raise ValueError("user_id and record_id are required")
        operation = operation.upper()
        if operation not in {"INSERT", "UPDATE", "DELETE"}:
            raise ValueError("operation must be INSERT, UPDATE, or DELETE")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        data = dict(payload or {})
        if operation != "DELETE":
            payload_json = json.dumps(data, separators=(",", ":"), sort_keys=True)
        else:
            payload_json = json.dumps(data, separators=(",", ":"), sort_keys=True)

        conn = self.connection
        conn.execute("SAVEPOINT sync_queue_stage")
        try:
            row = conn.execute(
                """SELECT id, operation FROM sync_queue
                   WHERE user_id=? AND table_name=? AND record_id=?
                     AND status IN ('PENDING','FAILED')""",
                (user_id, table_name, record_id),
            ).fetchone()

            if row:
                queue_id, previous = row[0], row[1]
                if previous == "INSERT" and operation == "DELETE":
                    conn.execute("DELETE FROM sync_queue WHERE id=?", (queue_id,))
                    conn.execute("RELEASE SAVEPOINT sync_queue_stage")
                    return None
                if previous == "INSERT" and operation == "UPDATE":
                    next_operation = "INSERT"
                elif previous == "DELETE" and operation == "INSERT":
                    next_operation = "UPDATE"
                else:
                    next_operation = operation
                conn.execute(
                    """UPDATE sync_queue SET operation=?, payload_json=?,
                       status='PENDING', retry_count=0, last_error=NULL,
                       locked_until=NULL, max_retries=?, updated_at=CURRENT_TIMESTAMP
                       WHERE id=?""",
                    (next_operation, payload_json, max_retries, queue_id),
                )
            else:
                queue_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO sync_queue
                       (id,user_id,table_name,record_id,operation,payload_json,max_retries)
                       VALUES (?,?,?,?,?,?,?)""",
                    (queue_id, user_id, table_name, record_id, operation, payload_json, max_retries),
                )
            conn.execute("RELEASE SAVEPOINT sync_queue_stage")
            return queue_id
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT sync_queue_stage")
            conn.execute("RELEASE SAVEPOINT sync_queue_stage")
            raise

    def claim_batch(self, limit: int = 20, lease_seconds: int = 120) -> List[Dict[str, Any]]:
        """Atomically claim pending or expired in-flight entries in creation order."""
        if limit < 1 or lease_seconds < 1:
            raise ValueError("limit and lease_seconds must be positive")
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """UPDATE sync_queue SET status='PENDING', locked_until=NULL
                   WHERE status='IN_FLIGHT' AND locked_until < strftime('%s','now')"""
            )
            rows = conn.execute(
                """SELECT id,user_id,table_name,record_id,operation,payload_json,
                          retry_count,max_retries
                   FROM sync_queue WHERE status='PENDING'
                   ORDER BY created_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
            ids = [row[0] for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""UPDATE sync_queue SET status='IN_FLIGHT',
                        locked_until=strftime('%s','now') + ?
                        WHERE id IN ({placeholders})""",
                    (lease_seconds, *ids),
                )
            conn.commit()
            return [
                {
                    "id": row[0], "user_id": row[1], "table_name": row[2],
                    "record_id": row[3], "operation": row[4],
                    "payload": json.loads(row[5]), "retry_count": row[6],
                    "max_retries": row[7],
                }
                for row in rows
            ]
        except Exception:
            conn.rollback()
            raise

    def mark_completed(self, queue_id: str) -> None:
        self.connection.execute("DELETE FROM sync_queue WHERE id=?", (queue_id,))
        self.connection.commit()

    def mark_failed(self, queue_id: str, error: str, *, retryable: bool) -> None:
        row = self.connection.execute(
            "SELECT retry_count,max_retries FROM sync_queue WHERE id=?", (queue_id,)
        ).fetchone()
        if row is None:
            return
        retries, maximum = int(row[0]), int(row[1])
        status = "PENDING" if retryable and retries + 1 < maximum else "FAILED"
        self.connection.execute(
            """UPDATE sync_queue SET retry_count=retry_count+1, last_error=?,
               status=?, locked_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (str(error)[:2000], status, queue_id),
        )
        self.connection.commit()

    def retry_failed(self, queue_id: Optional[str] = None) -> int:
        """Manually requeue failed items; returns the number changed."""
        if queue_id:
            cursor = self.connection.execute(
                """UPDATE sync_queue SET status='PENDING', retry_count=0,
                   last_error=NULL, locked_until=NULL WHERE id=? AND status='FAILED'""",
                (queue_id,),
            )
        else:
            cursor = self.connection.execute(
                """UPDATE sync_queue SET status='PENDING', retry_count=0,
                   last_error=NULL, locked_until=NULL WHERE status='FAILED'"""
            )
        self.connection.commit()
        return cursor.rowcount

    def pending_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM sync_queue WHERE status IN ('PENDING','IN_FLIGHT')"
        ).fetchone()
        return int(row[0])

    def failed_items(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT id,user_id,table_name,record_id,operation,retry_count,last_error
               FROM sync_queue WHERE status='FAILED' ORDER BY updated_at DESC"""
        ).fetchall()
        return [
            {"id": r[0], "user_id": r[1], "table_name": r[2], "record_id": r[3],
             "operation": r[4], "retry_count": r[5], "last_error": r[6]}
            for r in rows
        ]

"""Relay AI Universal Context Engine — Database Migration Runner.

Provides schema versioning, tracking, and incremental migrations for SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable, Dict, List, Tuple

from src.context.db.schema import (
    CORE_TABLES_DDL,
    DEFAULT_DATA_DML,
    FTS5_TABLES_DDL,
    INDEXES_DDL,
    SCHEMA_VERSION,
    TRIGGERS_DDL,
)

logger = logging.getLogger(__name__)


def _migration_v1(conn: sqlite3.Connection) -> None:
    """Migration 1: Initial schema for Universal Context Engine."""
    cursor = conn.cursor()
    cursor.executescript(CORE_TABLES_DDL)
    cursor.executescript(FTS5_TABLES_DDL)
    cursor.executescript(TRIGGERS_DDL)
    cursor.executescript(INDEXES_DDL)
    cursor.executescript(DEFAULT_DATA_DML)


def _migration_v2(conn: sqlite3.Connection) -> None:
    """Migration 2: Durable local outbox for cloud synchronization."""
    conn.execute(
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
            locked_until REAL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sync_queue_status_created "
        "ON sync_queue(status, created_at ASC)"
    )


MIGRATIONS: Dict[int, Tuple[str, Callable[[sqlite3.Connection], None]]] = {
    1: (
        "Initial UCE schema: projects, conversations, messages, memories, documents, chunks, and FTS5 indices",
        _migration_v1,
    ),
    2: (
        "Durable local sync queue with retry state and lease recovery",
        _migration_v2,
    ),
}


class MigrationManager:
    """Manages schema migrations for the SQLite context database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_migration_table(self) -> None:
        """Create schema_migrations tracking table if it doesn't already exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def get_applied_versions(self) -> List[int]:
        """Return list of schema version integers that have already been applied."""
        self.ensure_migration_table()
        cursor = self.conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version ASC;"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_current_version(self) -> int:
        """Return the highest applied schema version, or 0 if none applied."""
        applied = self.get_applied_versions()
        return max(applied) if applied else 0

    def apply_pending_migrations(self) -> int:
        """Apply all pending migrations in order.

        Returns:
            The highest applied version number after migration.
        """
        self.ensure_migration_table()
        applied = set(self.get_applied_versions())
        pending_versions = sorted([v for v in MIGRATIONS if v not in applied])

        if not pending_versions:
            logger.debug("Database schema is up to date (version %d).", self.get_current_version())
            return self.get_current_version()

        for version in pending_versions:
            desc, migration_fn = MIGRATIONS[version]
            logger.info("Applying database migration %d: %s", version, desc)
            try:
                # Wrap migration in transaction
                with self.conn:
                    migration_fn(self.conn)
                    self.conn.execute(
                        "INSERT INTO schema_migrations (version, description) VALUES (?, ?);",
                        (version, desc),
                    )
            except Exception as e:
                logger.error("Failed to apply migration %d (%s): %s", version, desc, e)
                raise

        current = self.get_current_version()
        logger.info("Database schema successfully migrated to version %d.", current)
        return current

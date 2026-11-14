"""Tests for Relay AI Universal Context Engine — Database Layer (Phase 10.1)."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import unittest

from src.context.db.connection import (
    ContextDatabase,
    check_database_integrity,
    configure_connection_pragmas,
    get_db_connection,
    get_default_db_path,
    initialize_database,
)
from src.context.db.migrations import MigrationManager
from src.context.db.schema import SCHEMA_VERSION


class TestContextDatabase(unittest.TestCase):
    """Test suite for ContextDatabase, connection lifecycle, pragmas, and migrations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="relay_test_db_")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_db_path(self) -> None:
        """Verify default db path points to RelayAI directory and relay_context.db."""
        path = get_default_db_path()
        self.assertIsInstance(path, Path)
        self.assertEqual(path.name, "relay_context.db")
        self.assertIn("RelayAI", str(path))

    def test_in_memory_initialization(self) -> None:
        """Verify in-memory database initializes with all core tables and migrations."""
        db = ContextDatabase(":memory:")
        db.initialize()

        row = db.fetchone("SELECT version FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,))
        self.assertIsNotNone(row)
        self.assertEqual(row["version"], SCHEMA_VERSION)

        # Verify tables exist
        tables = [r[0] for r in db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")]
        expected = [
            "schema_migrations",
            "projects",
            "conversations",
            "messages",
            "memories",
            "documents",
            "document_chunks",
            "chunks_fts",
            "memories_fts",
        ]
        for tbl in expected:
            self.assertIn(tbl, tables, f"Expected table {tbl} was not created.")

        db.close()

    def test_file_based_initialization_wal_mode(self) -> None:
        """Verify file-based database activates WAL mode and synchronous NORMAL."""
        db_path = Path(self.temp_dir) / "test_context.db"
        db = initialize_database(db_path)

        journal_mode = db.fetchone("PRAGMA journal_mode;")[0]
        self.assertEqual(journal_mode.lower(), "wal")

        sync_mode = db.fetchone("PRAGMA synchronous;")[0]
        # NORMAL is 1 in SQLite
        self.assertIn(sync_mode, (1, "1", "NORMAL"))

        db.close()

    def test_pragmas_enforced(self) -> None:
        """Verify foreign_keys, busy_timeout, and recursive_triggers are active."""
        db = initialize_database(":memory:")

        fk = db.fetchone("PRAGMA foreign_keys;")[0]
        self.assertEqual(fk, 1)

        rt = db.fetchone("PRAGMA recursive_triggers;")[0]
        self.assertEqual(rt, 1)

        timeout = db.fetchone("PRAGMA busy_timeout;")[0]
        self.assertGreaterEqual(timeout, 5000)

        db.close()

    def test_default_global_project_seeded(self) -> None:
        """Verify the 'global' default project is seeded automatically."""
        db = initialize_database(":memory:")
        row = db.fetchone("SELECT id, name, description FROM projects WHERE id = 'global'")
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "Global / General")
        db.close()

    def test_migration_idempotency(self) -> None:
        """Verify applying migrations multiple times is a clean no-op."""
        db = ContextDatabase(":memory:")
        db.initialize()

        conn = db.get_connection()
        mm = MigrationManager(conn)
        v1 = mm.get_current_version()
        self.assertEqual(v1, SCHEMA_VERSION)

        # Run again
        v2 = mm.apply_pending_migrations()
        self.assertEqual(v2, SCHEMA_VERSION)

        applied = mm.get_applied_versions()
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0], SCHEMA_VERSION)

        db.close()

    def test_foreign_key_cascades(self) -> None:
        """Verify deleting a project cascades to conversations, messages, documents, and memories."""
        db = initialize_database(":memory:")

        # 1. Create a custom project
        db.execute(
            "INSERT INTO projects (id, name, description) VALUES (?, ?, ?)",
            ("proj_alpha", "Project Alpha", "Test project"),
        )

        # 2. Add conversation and messages
        db.execute(
            "INSERT INTO conversations (id, project_id, title) VALUES (?, ?, ?)",
            ("conv_1", "proj_alpha", "Chat 1"),
        )
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
            ("msg_1", "conv_1", "user", "Hello Alpha"),
        )

        # 3. Add document and chunks
        db.execute(
            "INSERT INTO documents (id, project_id, file_name, file_path, file_hash, file_size_bytes, mime_type, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("doc_1", "proj_alpha", "test.txt", "/test.txt", "h1", 100, "text/plain", "INDEXED"),
        )
        db.execute(
            "INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content) "
            "VALUES (?, ?, ?, ?, ?)",
            ("chunk_1", "doc_1", "proj_alpha", 0, "Alpha chunk text"),
        )

        # 4. Add memory
        db.execute(
            "INSERT INTO memories (id, project_id, category, content, content_hash, provenance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("mem_1", "proj_alpha", "PROJECT_FACT", "Alpha uses Python", "mh1", "USER_EXPLICIT"),
        )

        # Verify records exist
        self.assertIsNotNone(db.fetchone("SELECT id FROM conversations WHERE id = 'conv_1'"))
        self.assertIsNotNone(db.fetchone("SELECT id FROM messages WHERE id = 'msg_1'"))
        self.assertIsNotNone(db.fetchone("SELECT id FROM documents WHERE id = 'doc_1'"))
        self.assertIsNotNone(db.fetchone("SELECT id FROM document_chunks WHERE id = 'chunk_1'"))
        self.assertIsNotNone(db.fetchone("SELECT id FROM memories WHERE id = 'mem_1'"))

        # 5. Delete project -> verify all cascade deleted
        db.execute("DELETE FROM projects WHERE id = 'proj_alpha'")

        self.assertIsNone(db.fetchone("SELECT id FROM conversations WHERE id = 'conv_1'"))
        self.assertIsNone(db.fetchone("SELECT id FROM messages WHERE id = 'msg_1'"))
        self.assertIsNone(db.fetchone("SELECT id FROM documents WHERE id = 'doc_1'"))
        self.assertIsNone(db.fetchone("SELECT id FROM document_chunks WHERE id = 'chunk_1'"))
        self.assertIsNone(db.fetchone("SELECT id FROM memories WHERE id = 'mem_1'"))

        db.close()

    def test_fts5_triggers_synchronization(self) -> None:
        """Verify triggers keep chunks_fts and memories_fts in sync on insert, update, and delete."""
        db = initialize_database(":memory:")

        # Add project and document
        db.execute("INSERT INTO projects (id, name) VALUES ('p1', 'P1')")
        db.execute(
            "INSERT INTO documents (id, project_id, file_name, file_path, file_hash, file_size_bytes, mime_type, status) "
            "VALUES ('d1', 'p1', 'f.txt', '/f.txt', 'h', 10, 'text/plain', 'INDEXED')"
        )

        # 1. Insert chunk -> triggers chunks_fts insert
        db.execute(
            "INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content) "
            "VALUES ('c1', 'd1', 'p1', 0, 'Quantum computing algorithms and supercomputers')"
        )
        res = db.fetchall("SELECT chunk_id, content FROM chunks_fts WHERE chunks_fts MATCH 'quantum'")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "c1")

        # 2. Update chunk -> triggers chunks_fts update
        db.execute(
            "UPDATE document_chunks SET content = 'Neural network architectures and deep transformers' WHERE id = 'c1'"
        )
        old_match = db.fetchall("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'quantum'")
        self.assertEqual(len(old_match), 0)

        new_match = db.fetchall("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'transformers'")
        self.assertEqual(len(new_match), 1)

        # 3. Delete chunk -> triggers chunks_fts delete
        db.execute("DELETE FROM document_chunks WHERE id = 'c1'")
        del_match = db.fetchall("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'transformers'")
        self.assertEqual(len(del_match), 0)

        # 4. Memories FTS trigger verification
        db.execute(
            "INSERT INTO memories (id, project_id, category, content, content_hash, provenance) "
            "VALUES ('m1', 'p1', 'PROJECT_FACT', 'Always use strict type hints in TypeScript', 'h2', 'USER_EXPLICIT')"
        )
        mem_res = db.fetchall("SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'TypeScript'")
        self.assertEqual(len(mem_res), 1)
        self.assertEqual(mem_res[0]["memory_id"], "m1")

        # Update memory
        db.execute("UPDATE memories SET content = 'Prefer dataclasses in modern Python code' WHERE id = 'm1'")
        self.assertEqual(len(db.fetchall("SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'TypeScript'")), 0)
        self.assertEqual(len(db.fetchall("SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'dataclasses'")), 1)

        # Delete memory
        db.execute("DELETE FROM memories WHERE id = 'm1'")
        self.assertEqual(len(db.fetchall("SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'dataclasses'")), 0)

        db.close()

    def test_recursive_cascade_cleans_fts(self) -> None:
        """Verify deleting a project cascades to document_chunks and cleans chunks_fts via recursive triggers."""
        db = initialize_database(":memory:")

        db.execute("INSERT INTO projects (id, name) VALUES ('proj_temp', 'Temp')")
        db.execute(
            "INSERT INTO documents (id, project_id, file_name, file_path, file_hash, file_size_bytes, mime_type, status) "
            "VALUES ('d_temp', 'proj_temp', 'temp.txt', '/temp.txt', 'h', 10, 'text/plain', 'INDEXED')"
        )
        db.execute(
            "INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content) "
            "VALUES ('c_temp', 'd_temp', 'proj_temp', 0, 'Specialized aerodynamic propulsion dynamics')"
        )

        match_before = db.fetchall("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'aerodynamic'")
        self.assertEqual(len(match_before), 1)

        # Deleting the parent project should cascade and trigger deletion from FTS
        db.execute("DELETE FROM projects WHERE id = 'proj_temp'")

        match_after = db.fetchall("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'aerodynamic'")
        self.assertEqual(len(match_after), 0)

        db.close()

    def test_transaction_rollback_on_error(self) -> None:
        """Verify an exception inside db.connection() rolls back uncommitted changes."""
        db = initialize_database(":memory:")

        try:
            with db.connection() as conn:
                conn.execute(
                    "INSERT INTO projects (id, name, description) VALUES (?, ?, ?)",
                    ("p_fail", "Rollback Project", "Should not exist"),
                )
                raise RuntimeError("Simulated transaction failure")
        except RuntimeError:
            pass

        row = db.fetchone("SELECT id FROM projects WHERE id = 'p_fail'")
        self.assertIsNone(row, "Transaction was not rolled back after exception.")

        db.close()

    def test_concurrent_threads_wal_safety(self) -> None:
        """Verify multiple threads can concurrently write and read without database locked errors."""
        db_path = Path(self.temp_dir) / "concurrent_wal.db"
        db = initialize_database(db_path)

        def worker_task(worker_id: int) -> int:
            # Each thread creates project and adds memories
            proj_id = f"worker_proj_{worker_id}"
            db.execute("INSERT INTO projects (id, name) VALUES (?, ?)", (proj_id, f"Project {worker_id}"))

            for i in range(10):
                mem_id = f"mem_{worker_id}_{i}"
                db.execute(
                    "INSERT INTO memories (id, project_id, category, content, content_hash, provenance) "
                    "VALUES (?, ?, 'USER_PREFERENCE', ?, ?, 'USER_EXPLICIT')",
                    (mem_id, proj_id, f"Preference {i} from worker {worker_id}", f"hash_{worker_id}_{i}"),
                )

            # Read back count
            row = db.fetchone("SELECT COUNT(*) as cnt FROM memories WHERE project_id = ?", (proj_id,))
            return row["cnt"]

        num_threads = 6
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        for res in results:
            self.assertEqual(res, 10)

        total_row = db.fetchone("SELECT COUNT(*) as total FROM memories WHERE project_id != 'global'")
        self.assertEqual(total_row["total"], num_threads * 10)

        db.close()

    def test_corruption_detection_and_recovery(self) -> None:
        """Verify corrupted database file is detected, renamed to .corrupted, and a healthy DB initialized."""
        db_path = Path(self.temp_dir) / "corruptible.db"

        # Write corrupted garbage data into the file
        with open(db_path, "wb") as f:
            f.write(b"NOT_A_VALID_SQLITE_DATABASE_HEADER_DATA_1234567890")

        self.assertTrue(db_path.exists())

        # Initialize database — should detect corruption, rename, and succeed
        db = ContextDatabase(db_path)
        db.initialize()

        # Database should now be functional
        row = db.fetchone("SELECT id, name FROM projects WHERE id = 'global'")
        self.assertIsNotNone(row)
        self.assertEqual(row["id"], "global")

        # Verify a corrupted file was archived in the directory
        archived_files = list(Path(self.temp_dir).glob("corruptible_corrupted_*.db"))
        self.assertGreaterEqual(len(archived_files), 1, "Corrupted file was not preserved.")

        db.close()


if __name__ == "__main__":
    unittest.main()

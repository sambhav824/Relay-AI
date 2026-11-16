"""Unit tests for Relay AI Embedding Recovery Manager (Milestone 10.4, Phase 2).

Tests interrupted job recovery, application restart resumption, cache reuse,
repeated failure tracking, retry exhaustion guards, and DocumentManager integration.
"""

from __future__ import annotations

import json
import sqlite3
from unittest import TestCase
from unittest.mock import MagicMock

from src.context.db.schema import CORE_TABLES_DDL, FTS5_TABLES_DDL, TRIGGERS_DDL
from src.context.documents.manager import DocumentManager
from src.context.embeddings.cache import EmbeddingCache
from src.context.embeddings.pipeline import EmbeddingPipeline
from src.context.embeddings.recovery import EmbeddingRecoveryManager
from src.context.embeddings.retry import RetryConfig
from src.providers.embeddings import MockEmbeddingProvider
from src.providers.exceptions import ProviderConnectionError


class TestEmbeddingRecovery(TestCase):
    """Unit test suite for EmbeddingRecoveryManager."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(CORE_TABLES_DDL)
        self.conn.executescript(FTS5_TABLES_DDL)
        self.conn.executescript(TRIGGERS_DDL)

        self.cache = EmbeddingCache(self.conn)
        self.retry_config = RetryConfig(max_retries=1, base_delay=0.01, jitter_ratio=0.0, sleep_fn=lambda _: None)
        self.pipeline = EmbeddingPipeline(cache=self.cache, retry_config=self.retry_config)
        self.provider = MockEmbeddingProvider(dimension=768)

        self.recovery = EmbeddingRecoveryManager(
            connection=self.conn,
            pipeline=self.pipeline,
            provider=self.provider,
            max_retries=3,
        )

        # Seed sample project and document
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO projects (id, name) VALUES ('p1', 'Test Project')")
            self.conn.execute(
                """
                INSERT INTO documents (id, project_id, file_name, file_path, file_hash, file_size_bytes, mime_type, status)
                VALUES ('doc_1', 'p1', 'report.txt', '/tmp/report.txt', 'hash_doc_1', 1024, 'text/plain', 'INDEXED')
                """
            )

    def tearDown(self):
        self.conn.close()

    def _insert_chunk(self, chunk_id: str, doc_id: str, idx: int, content: str, meta: dict = None):
        meta_json = json.dumps(meta or {}, separators=(",", ":"))
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content, metadata_json)
                VALUES (?, ?, 'p1', ?, ?, ?)
                """,
                (chunk_id, doc_id, idx, content, meta_json),
            )

    def test_interrupted_job_recovery(self):
        """Simulate an interrupted ingestion where 3 chunks exist but 2 were never embedded."""
        self._insert_chunk("c1", "doc_1", 0, "Chunk 1 content")
        self._insert_chunk("c2", "doc_1", 1, "Chunk 2 content")
        self._insert_chunk("c3", "doc_1", 2, "Chunk 3 content")

        # Suppose c1 was already embedded and in cache before the app crashed/interrupted
        self.pipeline.embed_chunks([{"id": "c1", "content": "Chunk 1 content", "index": 0}], self.provider)

        # Verify get_unembedded_chunks detects only c2 and c3
        unembedded = self.recovery.get_unembedded_chunks(document_id="doc_1")
        self.assertEqual(len(unembedded), 2)
        self.assertEqual(unembedded[0].chunk_id, "c2")
        self.assertEqual(unembedded[1].chunk_id, "c3")

        # Execute recovery
        result = self.recovery.recover_document("doc_1")
        self.assertEqual(result.total_chunks, 2)
        self.assertEqual(len(result.successful), 2)
        self.assertTrue(result.is_complete_success)

        # Now all 3 must be recognized as embedded
        remaining = self.recovery.get_unembedded_chunks(document_id="doc_1")
        self.assertEqual(len(remaining), 0)

    def test_repeated_failure_and_retry_exhaustion(self):
        """Verify chunks that fail repeatedly increment retry count and are skipped after max_retries."""
        self._insert_chunk("c_fail", "doc_1", 0, "Flaky content")

        failing_provider = MockEmbeddingProvider()
        # Mock persistent connection failure across all retries
        failing_provider.embed_documents = MagicMock(
            side_effect=ProviderConnectionError("Host down", provider_id="mock")
        )
        fail_recovery = EmbeddingRecoveryManager(
            connection=self.conn,
            pipeline=self.pipeline,
            provider=failing_provider,
            max_retries=2,
        )

        # Attempt 1: fails
        res1 = fail_recovery.recover_document("doc_1")
        self.assertEqual(len(res1.failed), 1)

        row = self.conn.execute("SELECT metadata_json FROM document_chunks WHERE id = 'c_fail'").fetchone()
        meta = json.loads(row[0])
        self.assertEqual(meta.get("embedding_status"), "FAILED")
        self.assertEqual(meta.get("embedding_retry_count"), 1)
        self.assertIn("Host down", meta.get("embedding_error", ""))

        # Attempt 2: fails again
        res2 = fail_recovery.recover_document("doc_1")
        self.assertEqual(len(res2.failed), 1)

        row = self.conn.execute("SELECT metadata_json FROM document_chunks WHERE id = 'c_fail'").fetchone()
        meta = json.loads(row[0])
        self.assertEqual(meta.get("embedding_retry_count"), 2)

        # Attempt 3: retry count (2) >= max_retries (2). Chunk must be skipped to avoid infinite loops!
        unembedded = fail_recovery.get_unembedded_chunks(document_id="doc_1", force=False)
        self.assertEqual(len(unembedded), 0)

        # If user forces recovery and provider is healthy, chunk is re-evaluated and succeeds
        healthy_provider = MockEmbeddingProvider()
        fail_recovery.provider = healthy_provider
        res_forced = fail_recovery.recover_document("doc_1", force=True)
        self.assertEqual(len(res_forced.successful), 1)

        row = self.conn.execute("SELECT metadata_json FROM document_chunks WHERE id = 'c_fail'").fetchone()
        meta = json.loads(row[0])
        self.assertIn(meta.get("embedding_status"), ("COMPLETED", "INDEXED"))
        self.assertNotIn("embedding_error", meta)

    def test_cache_reuse_during_recovery(self):
        """Verify that recovering documents reuses cached embeddings with zero provider calls."""
        self._insert_chunk("c_reuse", "doc_1", 0, "Reusable content")

        # Pre-seed cache
        self.pipeline.embed_chunks([{"id": "c_reuse", "content": "Reusable content", "index": 0}], self.provider)

        spy_embed = MagicMock(wraps=self.provider.embed_documents)
        self.provider.embed_documents = spy_embed

        unembedded = self.recovery.get_unembedded_chunks(document_id="doc_1")
        # Since it is in the cache, it was auto-synced to INDEXED and requires zero API calls
        self.assertEqual(len(unembedded), 0)
        self.assertEqual(spy_embed.call_count, 0)

    def test_document_manager_ingestion_with_failing_provider(self):
        """Verify that DocumentManager gracefully completes ingestion even when embedding provider is offline."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "notes.txt"
            file_path.write_text("Relay AI universal context engine testing notes.\nSecond paragraph.", encoding="utf-8")

            # Provider that crashes persistently across all attempts
            crash_provider = MockEmbeddingProvider()
            crash_provider.embed_documents = MagicMock(
                side_effect=ProviderConnectionError("Offline daemon", provider_id="mock")
            )

            doc_mgr = DocumentManager(
                connection=self.conn,
                user_id="user_123",
                embedding_pipeline=self.pipeline,
                embedding_provider=crash_provider,
            )

            # Ingestion must succeed without throwing an unhandled exception!
            result = doc_mgr.ingest(file_path, project_id="p1")
            self.assertEqual(result.file_name, "notes.txt")
            self.assertEqual(result.status, "INDEXED")
            self.assertGreater(result.chunk_count, 0)

            # Chunks are in SQLite database and in FTS5 table
            fts_count = self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            self.assertEqual(fts_count, result.chunk_count)

            # Chunks are detected as unembedded by RecoveryManager
            unembedded = self.recovery.get_unembedded_chunks(document_id=result.document_id)
            self.assertEqual(len(unembedded), result.chunk_count)

            # Later when provider is healthy, recovery succeeds
            healthy_recovery = EmbeddingRecoveryManager(
                connection=self.conn,
                pipeline=self.pipeline,
                provider=self.provider,
            )
            recovery_res = healthy_recovery.recover_document(result.document_id)
            self.assertTrue(recovery_res.is_complete_success)
            self.assertEqual(len(recovery_res.successful), result.chunk_count)

    def test_crash_interruption_and_restart_resumption(self):
        """Simulate a crash midway through multi-batch processing, verify state persistence across restarts."""
        import tempfile
        from pathlib import Path
        from src.context.db.connection import initialize_database

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "relay_restart_test.db"

            # 1. Initialize DB and seed 6 chunks across 2 batches of 3
            ctx_db = initialize_database(db_path)
            conn1 = ctx_db.get_connection()
            with conn1:
                conn1.execute("INSERT OR IGNORE INTO projects (id, name) VALUES ('p_restart', 'Restart Project')")
                conn1.execute(
                    """
                    INSERT INTO documents (id, project_id, file_name, file_path, file_hash, file_size_bytes, mime_type, status)
                    VALUES ('doc_crash', 'p_restart', 'big.txt', '/tmp/big.txt', 'hash_big', 2048, 'text/plain', 'INDEXED')
                    """
                )
                for i in range(6):
                    meta = json.dumps({"owner_user_id": "u1", "embedding_status": "PENDING", "embedding_retry_count": 0})
                    conn1.execute(
                        """
                        INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content, metadata_json)
                        VALUES (?, 'doc_crash', 'p_restart', ?, ?, ?)
                        """,
                        (f"c_rst_{i}", i, f"Paragraph text chunk {i}", meta),
                    )

            # Verify all 6 are initially marked PENDING in SQLite
            rows = conn1.execute("SELECT metadata_json FROM document_chunks WHERE document_id = 'doc_crash'").fetchall()
            for r in rows:
                meta = json.loads(r[0])
                self.assertEqual(meta.get("embedding_status"), "PENDING")

            # 2. Setup pipeline with batch_size=3 that succeeds on batch 1 but simulates fatal crash on batch 2
            cache1 = EmbeddingCache(conn1)
            pipeline1 = EmbeddingPipeline(cache=cache1, batch_size=3)

            call_count = 0
            provider1 = MockEmbeddingProvider()

            def embed_with_crash_on_batch_2(texts):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return [[0.1] * 768 for _ in texts]
                # Batch 2 encounters fatal exception / crash
                raise RuntimeError("Simulated process crash / power failure during batch 2")

            provider1.embed_documents = embed_with_crash_on_batch_2

            rec1 = EmbeddingRecoveryManager(conn1, pipeline1, provider1)

            # Execute recovery - batch 1 completes and writes to SQLite; batch 2 fails gracefully
            res1 = rec1.recover_document("doc_crash")
            self.assertEqual(len(res1.successful), 3)
            self.assertEqual(len(res1.failed), 3)

            # 3. Verify on disk that batch 1 chunks are COMPLETED, batch 2 chunks are FAILED with retry count 1
            for i in range(3):
                r = conn1.execute("SELECT metadata_json FROM document_chunks WHERE id = ?", (f"c_rst_{i}",)).fetchone()
                meta = json.loads(r[0])
                self.assertEqual(meta.get("embedding_status"), "COMPLETED")
            for i in range(3, 6):
                r = conn1.execute("SELECT metadata_json FROM document_chunks WHERE id = ?", (f"c_rst_{i}",)).fetchone()
                meta = json.loads(r[0])
                self.assertEqual(meta.get("embedding_status"), "FAILED")
                self.assertEqual(meta.get("embedding_retry_count"), 1)
                self.assertIn("Simulated process crash", meta.get("embedding_error", ""))

            # 4. SIMULATE COMPLETE RESTART: close conn1 and ctx_db completely
            conn1.close()
            ctx_db.close()

            # 5. Open fresh connection to the same SQLite database
            conn2 = sqlite3.connect(str(db_path))
            try:
                cache2 = EmbeddingCache(conn2)
                pipeline2 = EmbeddingPipeline(cache=cache2, batch_size=3)
                healthy_provider = MockEmbeddingProvider()
                spy_healthy = MagicMock(wraps=healthy_provider.embed_documents)
                healthy_provider.embed_documents = spy_healthy

                rec2 = EmbeddingRecoveryManager(conn2, pipeline2, healthy_provider)

                # Inspect unembedded chunks: must ONLY find the 3 remaining failed chunks!
                unembedded = rec2.get_unembedded_chunks(document_id="doc_crash")
                self.assertEqual(len(unembedded), 3)
                self.assertEqual([u.chunk_id for u in unembedded], ["c_rst_3", "c_rst_4", "c_rst_5"])

                # 6. Resume recovery after restart
                resume_res = rec2.recover_document("doc_crash")
                self.assertTrue(resume_res.is_complete_success)
                self.assertEqual(len(resume_res.successful), 3)

                # Provider must only have been called ONCE for the 3 missing chunks (0 re-processing of completed chunks!)
                self.assertEqual(spy_healthy.call_count, 1)
                self.assertEqual(len(spy_healthy.call_args[0][0]), 3)

                # 7. Final database verification: 100% of chunks are COMPLETED
                rows_final = conn2.execute("SELECT metadata_json FROM document_chunks WHERE document_id = 'doc_crash'").fetchall()
                self.assertEqual(len(rows_final), 6)
                for r in rows_final:
                    meta = json.loads(r[0])
                    self.assertEqual(meta.get("embedding_status"), "COMPLETED")

                # No remaining unembedded chunks
                self.assertEqual(len(rec2.get_unembedded_chunks(document_id="doc_crash")), 0)
            finally:
                conn2.close()


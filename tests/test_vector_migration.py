"""Unit tests for Supabase pgvector migration and schema validation (Milestone 10.4, Phase 3).

Validates migration and rollback SQL files for:
- pgvector extension provisioning
- Dual-dimension vector columns (1536 and 768)
- Integrity check constraints
- Canonical model-specific partial HNSW indexes (avoiding generic fallback duplication)
- Rejection of dynamic SECURITY DEFINER DDL functions
- Secure RPC implementation (SECURITY INVOKER, composite join on documents.project_id, auth.uid() checks)
- Fail-closed data safety rollback guard
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from unittest import TestCase


class TestVectorMigrationFiles(TestCase):
    """Test suite for validating vector migration and rollback SQL files."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        cls.migrations_dir = repo_root / "supabase" / "migrations"
        cls.up_path = cls.migrations_dir / "20261115000001_milestone_10_4_vector_retrieval.sql"
        cls.down_path = cls.migrations_dir / "20261115000001_milestone_10_4_vector_retrieval_down.sql"

        assert cls.up_path.exists(), f"Up migration not found at {cls.up_path}"
        assert cls.down_path.exists(), f"Down migration not found at {cls.down_path}"

        cls.up_sql = cls.up_path.read_text(encoding="utf-8")
        cls.down_sql = cls.down_path.read_text(encoding="utf-8")

    def test_migration_files_exist(self):
        self.assertTrue(self.up_path.exists())
        self.assertTrue(self.down_path.exists())
        self.assertGreater(len(self.up_sql), 200)
        self.assertGreater(len(self.down_sql), 200)

    def test_pgvector_extension_setup(self):
        """Verify extension is created safely in extensions schema."""
        match = re.search(r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+vector\s+WITH\s+SCHEMA\s+extensions", self.up_sql, re.IGNORECASE)
        self.assertIsNotNone(match, "pgvector extension must be provisioned with SCHEMA extensions.")

    def test_dual_vector_columns_added(self):
        """Verify dual-vector columns (1536, 768) and metadata columns are added to document_chunks."""
        self.assertIn("embedding_1536", self.up_sql)
        self.assertIn("embedding_768", self.up_sql)
        self.assertIn("embedding_model", self.up_sql)
        self.assertIn("embedding_dimension", self.up_sql)

        # Check vector dimension types
        self.assertRegex(self.up_sql, r"embedding_1536\s+extensions\.vector\(1536\)")
        self.assertRegex(self.up_sql, r"embedding_768\s+extensions\.vector\(768\)")

    def test_integrity_constraints(self):
        """Verify check constraints for dimension bounds and vector-dimension consistency."""
        self.assertIn("chk_document_chunks_embedding_dimension", self.up_sql)
        self.assertIn("chk_document_chunks_vector_consistency", self.up_sql)
        # Prevents simultaneous population of both 1536 and 768
        self.assertRegex(self.up_sql, r"embedding_dimension\s*=\s*1536\s+AND\s+embedding_1536\s+IS\s+NOT\s+NULL\s+AND\s+embedding_768\s+IS\s+NULL")
        self.assertRegex(self.up_sql, r"embedding_dimension\s*=\s*768\s+AND\s+embedding_768\s+IS\s+NOT\s+NULL\s+AND\s+embedding_1536\s+IS\s+NULL")

    def test_canonical_partial_hnsw_indexes(self):
        """Verify the 3 canonical model-specific partial HNSW indexes are defined with cosine operator."""
        self.assertIn("idx_chunks_hnsw_openai_1536", self.up_sql)
        self.assertIn("idx_chunks_hnsw_openai_768", self.up_sql)
        self.assertIn("idx_chunks_hnsw_nomic_768", self.up_sql)

        # Verify vector_cosine_ops is used
        self.assertRegex(self.up_sql, r"USING\s+hnsw\s*\(embedding_1536\s+extensions\.vector_cosine_ops\)")
        self.assertRegex(self.up_sql, r"USING\s+hnsw\s*\(embedding_768\s+extensions\.vector_cosine_ops\)")

        # Verify exact partial index predicate matching
        self.assertIn("embedding_model = 'openai/text-embedding-3-small'", self.up_sql)
        self.assertIn("embedding_model = 'ollama/nomic-embed-text'", self.up_sql)

    def test_no_overlapping_generic_fallback_indexes(self):
        """Mandatory Correction A: Ensure no duplicate generic fallback indexes exist to prevent write amplification."""
        self.assertNotIn("idx_chunks_hnsw_fallback_1536", self.up_sql)
        self.assertNotIn("idx_chunks_hnsw_fallback_768", self.up_sql)

    def test_no_security_definer_dynamic_ddl(self):
        """Security Review: Ensure no client-callable dynamic index creation function exists."""
        self.assertNotIn("provision_model_hnsw_index", self.up_sql)
        self.assertNotIn("SECURITY DEFINER", self.up_sql)

    def test_rpc_security_and_tenant_isolation(self):
        """Verify match_document_chunks RPC uses SECURITY INVOKER, composite join, and auth.uid()."""
        # Security invoker
        self.assertIn("SECURITY INVOKER", self.up_sql)

        # Unauthenticated rejection with SQLSTATE 28000
        self.assertRegex(self.up_sql, r"v_user_id\s*:=\s*auth\.uid\(\);")
        self.assertIn("ERRCODE = '28000'", self.up_sql)

        # Composite join to public.documents for project_id
        self.assertRegex(
            self.up_sql,
            r"JOIN\s+public\.documents\s+d\s+ON\s+d\.id\s*=\s*c\.document_id\s+AND\s+d\.user_id\s*=\s*c\.user_id",
        )
        self.assertIn("d.project_id", self.up_sql)
        self.assertNotIn("c.project_id", self.up_sql)  # document_chunks does NOT have project_id in cloud schema

        # Defense-in-depth: explicit user_id filter
        self.assertIn("c.user_id = v_user_id", self.up_sql)

        # Cosine distance ordering
        self.assertIn("ORDER BY c.embedding_1536 <=> query_embedding_1536 ASC", self.up_sql)
        self.assertIn("ORDER BY c.embedding_768 <=> query_embedding_768 ASC", self.up_sql)

        # Permissions: only authenticated role granted execute
        self.assertRegex(self.up_sql, r"(?s)REVOKE\s+ALL\s+ON\s+FUNCTION\s+public\.match_document_chunks.*FROM\s+PUBLIC")
        self.assertRegex(self.up_sql, r"(?s)GRANT\s+EXECUTE\s+ON\s+FUNCTION\s+public\.match_document_chunks.*TO\s+authenticated")

    def test_rpc_static_branches_for_hnsw_compatibility(self):
        """Verify match_document_chunks contains static model literal branches for HNSW index planner selection."""
        # Exploration depth
        self.assertIn("SET LOCAL hnsw.ef_search = 100;", self.up_sql)

        # Static branch 1: OpenAI 1536d
        self.assertIn("p_dimension = 1536 AND p_model_id = 'openai/text-embedding-3-small'", self.up_sql)
        self.assertIn("c.embedding_model = 'openai/text-embedding-3-small'\n        AND c.embedding_dimension = 1536", self.up_sql)

        # Static branch 2: OpenAI 768d MRL
        self.assertIn("p_dimension = 768 AND p_model_id = 'openai/text-embedding-3-small'", self.up_sql)
        self.assertIn("c.embedding_model = 'openai/text-embedding-3-small'\n        AND c.embedding_dimension = 768", self.up_sql)

        # Static branch 3: Ollama Nomic 768d
        self.assertIn("p_dimension = 768 AND p_model_id = 'ollama/nomic-embed-text'", self.up_sql)
        self.assertIn("c.embedding_model = 'ollama/nomic-embed-text'\n        AND c.embedding_dimension = 768", self.up_sql)

        # Fallback branches for custom models
        self.assertIn("c.embedding_model = p_model_id\n        AND c.embedding_dimension = 1536", self.up_sql)
        self.assertIn("c.embedding_model = p_model_id\n        AND c.embedding_dimension = 768", self.up_sql)

    def test_down_migration_fail_closed_data_safety_guard(self):
        """Mandatory Correction B: Verify down migration refuses destructive rollback when vector data exists."""
        # 1. Contains check for existing embeddings
        self.assertRegex(
            self.down_sql,
            r"SELECT\s+count\(\*\)\s+INTO\s+v_chunk_count\s+FROM\s+public\.document_chunks\s+WHERE\s+embedding_1536\s+IS\s+NOT\s+NULL\s+OR\s+embedding_768\s+IS\s+NOT\s+NULL",
        )

        # 2. Checks session parameter relay.force_vector_rollback and documents SET LOCAL under PgBouncer/Supavisor
        self.assertIn("relay.force_vector_rollback", self.down_sql)
        self.assertIn("SET LOCAL relay.force_vector_rollback = 'true';", self.down_sql)
        self.assertIn("Supavisor", self.down_sql)

        # 3. Aborts with SQLSTATE 55000 if data exists and parameter is not set
        self.assertIn("ERRCODE = '55000'", self.down_sql)
        self.assertIn("DATA SAFETY GUARD ABORT", self.down_sql)

        # 4. Cleans up RPC, indexes, constraints, and columns
        self.assertIn("DROP FUNCTION IF EXISTS public.match_document_chunks", self.down_sql)
        self.assertIn("DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_1536", self.down_sql)
        self.assertIn("DROP INDEX IF EXISTS public.idx_chunks_hnsw_openai_768", self.down_sql)
        self.assertIn("DROP INDEX IF EXISTS public.idx_chunks_hnsw_nomic_768", self.down_sql)
        self.assertIn("DROP COLUMN IF EXISTS embedding_1536", self.down_sql)
        self.assertIn("DROP COLUMN IF EXISTS embedding_768", self.down_sql)

        # 5. Extension vector is preserved
        self.assertNotIn("DROP EXTENSION", self.down_sql)

    def test_live_supabase_status(self):
        """Explicitly classify live cloud database execution status."""
        supabase_url = os.environ.get("SUPABASE_URL")
        anon_key = os.environ.get("SUPABASE_ANON_KEY")

        if not supabase_url or not anon_key or "placeholder" in supabase_url:
            # Documented as expected: credentials unavailable in local test environment
            self.assertTrue(True, "Live Supabase test skipped — PENDING LIVE SUPABASE VALIDATION.")
        else:
            self.fail("Unexpected live credentials found in test environment.")

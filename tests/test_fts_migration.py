"""Unit tests for Supabase Full-Text Search migration and schema validation (Milestone 10.4, Phase 4).

Validates:
- Migration and rollback SQL file existence
- SQL function definition and return table signature
- SECURITY INVOKER enforcement (rejects SECURITY DEFINER)
- Tenant isolation and auth.uid() check with SQLSTATE 28000
- Safe query construction via websearch_to_tsquery and plainto_tsquery
- Ranking via ts_rank_cd
- Composite join on documents table (project_id & user_id)
- Permission grants (REVOKE FROM PUBLIC, GRANT TO authenticated)
- Rollback safety (drops RPC without deleting tables or data)
"""

from __future__ import annotations

from pathlib import Path
import re
from unittest import TestCase


class TestFTSMigrationFiles(TestCase):
    """Test suite for validating FTS migration and rollback SQL files."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        cls.migrations_dir = repo_root / "supabase" / "migrations"
        cls.up_path = cls.migrations_dir / "20261116000001_milestone_10_4_fts_retrieval.sql"
        cls.down_path = cls.migrations_dir / "20261116000001_milestone_10_4_fts_retrieval_down.sql"

        assert cls.up_path.exists(), f"Up migration not found at {cls.up_path}"
        assert cls.down_path.exists(), f"Down migration not found at {cls.down_path}"

        cls.up_sql = cls.up_path.read_text(encoding="utf-8")
        cls.down_sql = cls.down_path.read_text(encoding="utf-8")

    def test_migration_files_exist(self):
        """Verify migration files exist and have non-empty content."""
        self.assertTrue(self.up_path.exists())
        self.assertTrue(self.down_path.exists())
        self.assertGreater(len(self.up_sql), 200)
        self.assertGreater(len(self.down_sql), 50)

    def test_rpc_function_signature(self):
        """Verify search_document_chunks_fts signature and return types."""
        self.assertIn("CREATE OR REPLACE FUNCTION public.search_document_chunks_fts", self.up_sql)
        self.assertIn("p_query text", self.up_sql)
        self.assertIn("match_count integer DEFAULT 20", self.up_sql)
        self.assertIn("filter_project_id uuid DEFAULT NULL", self.up_sql)
        self.assertIn("filter_document_id uuid DEFAULT NULL", self.up_sql)

        # Return table columns
        self.assertIn("score double precision", self.up_sql)
        self.assertIn("project_id uuid", self.up_sql)
        self.assertIn("chunk_index integer", self.up_sql)
        self.assertIn("content text", self.up_sql)
        self.assertIn("metadata jsonb", self.up_sql)

    def test_security_invoker_and_no_security_definer(self):
        """Verify RPC uses SECURITY INVOKER and does NOT introduce SECURITY DEFINER."""
        self.assertIn("SECURITY INVOKER", self.up_sql)
        self.assertNotIn("SECURITY DEFINER", self.up_sql)

    def test_authentication_and_tenant_isolation(self):
        """Verify unauthenticated callers are rejected with SQLSTATE 28000 and user_id is enforced."""
        self.assertRegex(self.up_sql, r"v_user_id\s*:=\s*auth\.uid\(\);")
        self.assertIn("ERRCODE = '28000'", self.up_sql)
        self.assertIn("c.user_id = v_user_id", self.up_sql)

    def test_safe_tsquery_and_ranking(self):
        """Verify query is parsed safely with websearch_to_tsquery and plainto_tsquery, ranked with ts_rank_cd."""
        self.assertIn("websearch_to_tsquery('english', p_query)", self.up_sql)
        self.assertIn("plainto_tsquery('english', p_query)", self.up_sql)
        self.assertIn("ts_rank_cd(c.tsv_content, v_tsquery)", self.up_sql)
        self.assertIn("c.tsv_content @@ v_tsquery", self.up_sql)

    def test_composite_join_for_project_filtering(self):
        """Verify composite join to documents for project filtering and cross-user prevention."""
        self.assertRegex(
            self.up_sql,
            r"JOIN\s+public\.documents\s+d\s+ON\s+d\.id\s*=\s*c\.document_id\s+AND\s+d\.user_id\s*=\s*c\.user_id",
        )
        self.assertIn("d.project_id = filter_project_id", self.up_sql)
        self.assertIn("c.document_id = filter_document_id", self.up_sql)

    def test_permission_grants(self):
        """Verify public execution is revoked and authenticated execution is granted."""
        self.assertRegex(
            self.up_sql,
            r"(?s)REVOKE\s+ALL\s+ON\s+FUNCTION\s+public\.search_document_chunks_fts.*FROM\s+PUBLIC",
        )
        self.assertRegex(
            self.up_sql,
            r"(?s)GRANT\s+EXECUTE\s+ON\s+FUNCTION\s+public\.search_document_chunks_fts.*TO\s+authenticated",
        )

    def test_down_migration_drops_function(self):
        """Verify down migration drops the RPC function cleanly without touching tables."""
        self.assertIn("DROP FUNCTION IF EXISTS public.search_document_chunks_fts", self.down_sql)
        self.assertNotIn("DROP TABLE", self.down_sql)
        self.assertNotIn("DROP COLUMN", self.down_sql)

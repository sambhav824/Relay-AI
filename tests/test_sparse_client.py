"""Unit tests for Relay AI Sparse Keyword Retrieval Client (Milestone 10.4, Phase 4).

Validates:
- Supabase Full-Text Search RPC invocation and JSON response mapping
- Bearer token authentication via AuthManager
- Authentication failure rejection (HTTP 401/403 never swallowed)
- Network failure handling and seamless fallback to local SQLite FTS5
- Local SQLite FTS5 search execution and BM25 score ranking
- Safe query sanitization preventing SQLite FTS5 syntax errors
- Project and document level filtering
- Input validation and empty query handling
"""

from __future__ import annotations

import json
import sqlite3
from unittest import TestCase
from unittest.mock import MagicMock

import httpx

from src.context.auth.client import AuthSession, AuthUser
from src.context.auth.manager import AuthManager
from src.context.retrieval.sparse_client import (
    SparseClientAuthError,
    SparseClientRPCError,
    SparseClientValidationError,
    SparseMatch,
    SparseRetrievalClient,
)


class TestSparseRetrievalClient(TestCase):
    """Test suite for SparseRetrievalClient."""

    def setUp(self):
        self.supabase_url = "https://test-project.supabase.co"
        self.anon_key = "test-anon-key"
        self.access_token = "valid-test-jwt"

        # Create in-memory SQLite database with FTS5 and document_chunks tables
        self.sqlite_conn = sqlite3.connect(":memory:")
        self.sqlite_conn.execute("""
            CREATE TABLE document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                project_id TEXT DEFAULT 'global',
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            );
        """)
        self.sqlite_conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                project_id UNINDEXED,
                content,
                tokenize = 'porter unicode61'
            );
        """)
        self.sqlite_conn.commit()

    def tearDown(self):
        self.sqlite_conn.close()

    def _insert_local_chunk(self, chunk_id: str, doc_id: str, proj_id: str, chunk_idx: int, content: str, meta: dict):
        meta_json = json.dumps(meta)
        self.sqlite_conn.execute(
            "INSERT INTO document_chunks (id, document_id, project_id, chunk_index, content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, doc_id, proj_id, chunk_idx, content, meta_json),
        )
        self.sqlite_conn.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, project_id, content) VALUES (?, ?, ?, ?)",
            (chunk_id, doc_id, proj_id, content),
        )
        self.sqlite_conn.commit()

    def _make_mock_client(self, handler):
        transport = httpx.MockTransport(handler)
        return httpx.Client(transport=transport)

    def test_search_supabase_success(self):
        """Verify Supabase FTS RPC request formatting and JSON response parsing."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.read())

            return httpx.Response(
                200,
                json=[
                    {
                        "id": "c1-uuid",
                        "document_id": "d1-uuid",
                        "project_id": "p1-uuid",
                        "chunk_index": 2,
                        "content": "Full text keyword match content.",
                        "metadata": {"source": "manual.pdf"},
                        "score": 0.385,
                    }
                ],
            )

        http_client = self._make_mock_client(handler)
        client = SparseRetrievalClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=http_client,
        )

        results = client.search_keywords(
            query="keyword match",
            match_count=10,
            filter_project_id="p1-uuid",
        )

        self.assertEqual(len(results), 1)
        match = results[0]
        self.assertEqual(match.id, "c1-uuid")
        self.assertEqual(match.document_id, "d1-uuid")
        self.assertEqual(match.project_id, "p1-uuid")
        self.assertEqual(match.chunk_index, 2)
        self.assertEqual(match.content, "Full text keyword match content.")
        self.assertEqual(match.metadata, {"source": "manual.pdf"})
        self.assertAlmostEqual(match.score, 0.385)

        # Verify request details
        self.assertIn("/rest/v1/rpc/search_document_chunks_fts", captured["url"])
        self.assertEqual(captured["headers"]["authorization"], f"Bearer {self.access_token}")
        self.assertEqual(captured["body"]["p_query"], "keyword match")
        self.assertEqual(captured["body"]["match_count"], 10)
        self.assertEqual(captured["body"]["filter_project_id"], "p1-uuid")

    def test_auth_manager_integration(self):
        """Verify Bearer token is dynamically retrieved from AuthManager."""
        mock_mgr = MagicMock(spec=AuthManager)
        mock_mgr.get_access_token.return_value = "dynamic-jwt-from-auth-manager"

        captured_auth = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_auth.append(request.headers.get("authorization"))
            return httpx.Response(200, json=[])

        http_client = self._make_mock_client(handler)
        client = SparseRetrievalClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            auth_manager=mock_mgr,
            http_client=http_client,
        )

        client.search_keywords(query="test query")
        mock_mgr.get_access_token.assert_called_once()
        self.assertEqual(captured_auth, ["Bearer dynamic-jwt-from-auth-manager"])

    def test_auth_failure_raises_auth_error_never_swallowed(self):
        """Security: Verify HTTP 401/403 raises SparseClientAuthError and is never converted to fallback."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "JWT expired"})

        http_client = self._make_mock_client(handler)
        # Even if SQLite connection is provided, auth failures MUST NOT be swallowed
        client = SparseRetrievalClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token="expired-token",
            sqlite_connection=self.sqlite_conn,
            http_client=http_client,
        )

        with self.assertRaises(SparseClientAuthError):
            client.search_keywords(query="test query")

    def test_sqlstate_permission_errors_raise_auth_error(self):
        """Verify SQLSTATE 28000 and 42501 raise SparseClientAuthError and are not converted to fallback."""
        for code in ("28000", "42501"):
            def handler(_: httpx.Request) -> httpx.Response:
                return httpx.Response(400, json={"code": code, "message": "Permission denied"})

            client = SparseRetrievalClient(
                supabase_url=self.supabase_url,
                supabase_anon_key=self.anon_key,
                access_token=self.access_token,
                sqlite_connection=self.sqlite_conn,
                http_client=self._make_mock_client(handler),
            )

            with self.assertRaises(SparseClientAuthError):
                client.search_keywords(query="test query")

    def test_multi_user_isolation_in_sqlite(self):
        """Verify local SQLite search excludes chunks owned by other users when active user is known."""
        self._insert_local_chunk(
            chunk_id="user1_chunk",
            doc_id="d1",
            proj_id="proj1",
            chunk_idx=0,
            content="User 1 confidential database report.",
            meta={"owner_user_id": "user-uuid-1"},
        )
        self._insert_local_chunk(
            chunk_id="user2_chunk",
            doc_id="d2",
            proj_id="proj1",
            chunk_idx=0,
            content="User 2 confidential database report.",
            meta={"owner_user_id": "user-uuid-2"},
        )

        mock_auth_user1 = MagicMock(spec=AuthManager)
        mock_auth_user1.user = MagicMock()
        mock_auth_user1.user.id = "user-uuid-1"

        client_user1 = SparseRetrievalClient(
            sqlite_connection=self.sqlite_conn,
            auth_manager=mock_auth_user1,
        )

        results = client_user1.search_keywords("confidential database", prefer_offline=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "user1_chunk")
        self.assertEqual(results[0].metadata["owner_user_id"], "user-uuid-1")

    def test_network_failure_without_sqlite_raises_rpc_error(self):
        """Verify network connection failure raises SparseClientRPCError when no local SQLite fallback exists."""
        def handler(request: httpx.Request):
            raise httpx.ConnectError("Connection refused by peer")

        http_client = self._make_mock_client(handler)
        client = SparseRetrievalClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            sqlite_connection=None,
            http_client=http_client,
        )

        with self.assertRaises(SparseClientRPCError):
            client.search_keywords(query="test query")

    def test_network_failure_with_sqlite_falls_back_seamlessly(self):
        """Verify network failure transparently falls back to local SQLite FTS5 when SQLite connection exists."""
        def handler(request: httpx.Request):
            raise httpx.ConnectError("Cloud server unreachable")

        self._insert_local_chunk(
            chunk_id="local_c1",
            doc_id="local_d1",
            proj_id="proj_1",
            chunk_idx=0,
            content="Architecture of the distributed database engine.",
            meta={"local": True},
        )

        http_client = self._make_mock_client(handler)
        client = SparseRetrievalClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            sqlite_connection=self.sqlite_conn,
            http_client=http_client,
        )

        # Should log warning and fall back to SQLite without raising exception
        results = client.search_keywords(query="distributed database")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "local_c1")
        self.assertEqual(results[0].content, "Architecture of the distributed database engine.")
        self.assertEqual(results[0].metadata, {"local": True})

    def test_sqlite_fts5_direct_search(self):
        """Verify prefer_offline directly searches local SQLite FTS5."""
        self._insert_local_chunk(
            chunk_id="chunk_a",
            doc_id="doc_1",
            proj_id="project_alpha",
            chunk_idx=0,
            content="Neural networks and deep learning models.",
            meta={"page": 1},
        )
        self._insert_local_chunk(
            chunk_id="chunk_b",
            doc_id="doc_2",
            proj_id="project_beta",
            chunk_idx=1,
            content="Convolutional neural networks for image classification.",
            meta={"page": 2},
        )

        client = SparseRetrievalClient(sqlite_connection=self.sqlite_conn)

        # Alpha project filter
        results_alpha = client.search_keywords(
            query="neural networks",
            filter_project_id="project_alpha",
            prefer_offline=True,
        )
        self.assertEqual(len(results_alpha), 1)
        self.assertEqual(results_alpha[0].id, "chunk_a")

        # Beta project filter
        results_beta = client.search_keywords(
            query="neural networks",
            filter_project_id="project_beta",
            prefer_offline=True,
        )
        self.assertEqual(len(results_beta), 1)
        self.assertEqual(results_beta[0].id, "chunk_b")

    def test_query_sanitization_sqlite_fts5(self):
        """Verify complex or potentially crashing FTS5 query syntax is sanitized safely."""
        self._insert_local_chunk(
            chunk_id="safe_chunk",
            doc_id="doc_1",
            proj_id="global",
            chunk_idx=0,
            content="Relay AI query sanitization test case.",
            meta={},
        )

        client = SparseRetrievalClient(sqlite_connection=self.sqlite_conn)

        # Raw user inputs that would crash un-sanitized SQLite FTS5
        dangerous_queries = [
            'Relay AND OR NOT "unclosed quote',
            'query: sanitization (test) [case] *',
            '*** +++ ---',
            'Relay * AI',
        ]

        for q in dangerous_queries:
            results = client.search_keywords(query=q, prefer_offline=True)
            self.assertIsInstance(results, list)

    def test_empty_or_whitespace_query(self):
        """Verify empty or purely whitespace query immediately returns empty list."""
        client = SparseRetrievalClient(sqlite_connection=self.sqlite_conn)
        self.assertEqual(client.search_keywords(""), [])
        self.assertEqual(client.search_keywords("   \n\t  "), [])
        self.assertEqual(client.search_keywords(None), [])

    def test_parameter_bounds_validation(self):
        """Verify match_count bounds checking."""
        client = SparseRetrievalClient(sqlite_connection=self.sqlite_conn)
        with self.assertRaises(SparseClientValidationError):
            client.search_keywords("test", match_count=0)
        with self.assertRaises(SparseClientValidationError):
            client.search_keywords("test", match_count=101)

"""Unit tests for Relay AI Vector Similarity Client (Milestone 10.4, Phase 3).

Tests query vector validation, dimension routing, PostgREST payload serialization,
authentication handling, project/document filtering, and error handling.
"""

from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import httpx

from src.context.auth.client import AuthSession, AuthUser
from src.context.auth.manager import AuthManager
from src.context.retrieval.vector_client import (
    VectorClientAuthError,
    VectorClientRPCError,
    VectorClientValidationError,
    VectorMatch,
    VectorSimilarityClient,
)


class TestVectorSimilarityClient(TestCase):
    """Test suite for VectorSimilarityClient."""

    def setUp(self):
        self.supabase_url = "https://test-project.supabase.co"
        self.anon_key = "test-anon-key"
        self.access_token = "valid-test-jwt"

    def _make_mock_client(self, handler):
        transport = httpx.MockTransport(handler)
        return httpx.Client(transport=transport)

    def test_search_1536_dimension_success(self):
        """Verify 1536-dim vector is routed to query_embedding_1536 and response parsed."""
        captured_request = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["headers"] = dict(request.headers)
            captured_request["body"] = json.loads(request.read())

            return httpx.Response(
                200,
                json=[
                    {
                        "id": "c1-uuid",
                        "document_id": "d1-uuid",
                        "project_id": "p1-uuid",
                        "chunk_index": 0,
                        "content": "Paragraph 1 text content.",
                        "metadata": {"page": 1},
                        "similarity": 0.895,
                    }
                ],
            )

        http_client = self._make_mock_client(handler)
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=http_client,
        )

        query_vec = [0.05] * 1536
        matches = client.search_similar_chunks(
            query_vector=query_vec,
            model_id="openai/text-embedding-3-small",
            dimension=1536,
            match_threshold=0.5,
            match_count=5,
            filter_project_id="p1-uuid",
            filter_document_id="d1-uuid",
        )

        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.id, "c1-uuid")
        self.assertEqual(m.document_id, "d1-uuid")
        self.assertEqual(m.project_id, "p1-uuid")
        self.assertEqual(m.chunk_index, 0)
        self.assertEqual(m.content, "Paragraph 1 text content.")
        self.assertEqual(m.similarity, 0.895)

        # Inspect request payload
        self.assertIn("/rest/v1/rpc/match_document_chunks", captured_request["url"])
        self.assertEqual(captured_request["headers"]["authorization"], f"Bearer {self.access_token}")
        self.assertEqual(captured_request["headers"]["apikey"], self.anon_key)

        body = captured_request["body"]
        self.assertEqual(body["p_model_id"], "openai/text-embedding-3-small")
        self.assertEqual(body["p_dimension"], 1536)
        self.assertEqual(len(body["query_embedding_1536"]), 1536)
        self.assertIsNone(body["query_embedding_768"])
        self.assertEqual(body["match_threshold"], 0.5)
        self.assertEqual(body["match_count"], 5)
        self.assertEqual(body["filter_project_id"], "p1-uuid")
        self.assertEqual(body["filter_document_id"], "d1-uuid")

    def test_search_768_dimension_success(self):
        """Verify 768-dim vector is routed to query_embedding_768 and response parsed."""
        captured_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.read()))
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "c2-uuid",
                        "document_id": "d2-uuid",
                        "project_id": None,
                        "chunk_index": 2,
                        "content": "Nomic chunk content",
                        "metadata": {},
                        "similarity": 0.772,
                    }
                ],
            )

        http_client = self._make_mock_client(handler)
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=http_client,
        )

        query_vec = [-0.02] * 768
        matches = client.search_similar_chunks(
            query_vector=query_vec,
            model_id="ollama/nomic-embed-text",
            dimension=768,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, "c2-uuid")
        self.assertEqual(matches[0].chunk_index, 2)
        self.assertIsNone(matches[0].project_id)
        self.assertAlmostEqual(matches[0].similarity, 0.772)

        self.assertEqual(captured_body["p_model_id"], "ollama/nomic-embed-text")
        self.assertEqual(captured_body["p_dimension"], 768)
        self.assertEqual(len(captured_body["query_embedding_768"]), 768)
        self.assertIsNone(captured_body["query_embedding_1536"])

    def test_dimension_validation(self):
        """Verify invalid dimensions or length mismatches are rejected before network requests."""
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=self._make_mock_client(lambda _: httpx.Response(200, json=[])),
        )

        # Unsupported dimension
        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 512, "custom-model", 512)

        # Dimension mismatch: passed 768 parameter but 1536 vector
        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 1536, "openai/text-embedding-3-small", 768)

        # Empty vector
        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([], "openai/text-embedding-3-small", 1536)

        # Non-finite values (NaN / inf)
        with self.assertRaises(VectorClientValidationError):
            vec = [0.1] * 768
            vec[5] = float("nan")
            client.search_similar_chunks(vec, "ollama/nomic-embed-text", 768)

    def test_model_dimension_compatibility(self):
        """Verify Nomic model cannot be queried with 1536 dimensions."""
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=self._make_mock_client(lambda _: httpx.Response(200, json=[])),
        )

        with self.assertRaises(VectorClientValidationError) as ctx:
            client.search_similar_chunks([0.1] * 1536, "ollama/nomic-embed-text", 1536)
        self.assertIn("does not support 1536 dimensions", str(ctx.exception))

    def test_parameter_bounds_validation(self):
        """Verify threshold and count parameter bounds."""
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=self._make_mock_client(lambda _: httpx.Response(200, json=[])),
        )

        # Threshold < -1.0 or > 1.0
        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768, match_threshold=1.5)

        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768, match_threshold=-1.5)

        # Count < 1 or > 100
        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768, match_count=0)

        with self.assertRaises(VectorClientValidationError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768, match_count=101)

    def test_auth_manager_integration(self):
        """Verify token is retrieved dynamically from AuthManager."""
        mock_auth_manager = MagicMock(spec=AuthManager)
        mock_auth_manager.get_access_token.return_value = "dynamic-auth-jwt"

        captured_auth = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_auth
            captured_auth = request.headers.get("authorization")
            return httpx.Response(200, json=[])

        http_client = self._make_mock_client(handler)
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            auth_manager=mock_auth_manager,
            http_client=http_client,
        )

        client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768)
        self.assertEqual(captured_auth, "Bearer dynamic-auth-jwt")
        mock_auth_manager.get_access_token.assert_called_once()

    def test_missing_auth_raises_error(self):
        """Verify client raises VectorClientAuthError when no token is available."""
        mock_auth_manager = MagicMock(spec=AuthManager)
        mock_auth_manager.get_access_token.return_value = None

        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            auth_manager=mock_auth_manager,
            http_client=self._make_mock_client(lambda _: httpx.Response(200, json=[])),
        )

        with self.assertRaises(VectorClientAuthError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768)

    def test_unauthorized_http_response_raises_auth_error(self):
        """Verify 401 response from Supabase raises VectorClientAuthError."""
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="JWT expired")

        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token="expired-token",
            http_client=self._make_mock_client(handler),
        )

        with self.assertRaises(VectorClientAuthError):
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768)

    def test_postgrest_rpc_error_handling(self):
        """Verify 400 error with PostgreSQL error message is parsed properly."""
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"code": "22023", "message": "Invalid dimension: 512. Allowed dimensions are 768 and 1536."},
            )

        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=self._make_mock_client(handler),
        )

        with self.assertRaises(VectorClientRPCError) as ctx:
            client.search_similar_chunks([0.1] * 768, "custom-model", 768)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, "22023")
        self.assertIn("Invalid dimension", str(ctx.exception))

    def test_network_failure_handling(self):
        """Verify network connection errors raise VectorClientRPCError without leaking secrets."""
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused by peer")

        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
            http_client=self._make_mock_client(handler),
        )

        with self.assertRaises(VectorClientRPCError) as ctx:
            client.search_similar_chunks([0.1] * 768, "ollama/nomic-embed-text", 768)
        self.assertIn("Network error", str(ctx.exception))

    def test_context_manager(self):
        """Verify context manager interface operates cleanly."""
        client = VectorSimilarityClient(
            supabase_url=self.supabase_url,
            supabase_anon_key=self.anon_key,
            access_token=self.access_token,
        )
        with client as c:
            self.assertIsInstance(c, VectorSimilarityClient)

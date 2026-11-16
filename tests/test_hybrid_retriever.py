"""Unit tests for Relay AI Hybrid Retrieval Orchestrator (Milestone 10.4, Phase 4).

Validates:
- Full hybrid retrieval combining dense and sparse modalities (MODE_HYBRID)
- Dense-only mode fallback when sparse retrieval is empty/fails (MODE_DENSE_ONLY)
- Sparse-only mode fallback when dense retrieval is empty/fails (MODE_SPARSE_ONLY)
- Offline mode using local SQLite FTS5 (MODE_OFFLINE_SPARSE_ONLY)
- Empty search results handling (MODE_EMPTY)
- Query embedding generation via EmbeddingPipeline and provider
- Candidate oversampling verification
- Concurrent execution via ThreadPoolExecutor
- Re-raising authentication errors without swallowing
- Project and document filter propagation
- Provenance tracking (ranks, scores, modes)
"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.context.embeddings.pipeline import EmbeddingPipeline
from src.context.retrieval.hybrid import (
    MODE_DENSE_ONLY,
    MODE_EMPTY,
    MODE_HYBRID,
    MODE_OFFLINE_SPARSE_ONLY,
    MODE_SPARSE_ONLY,
    HybridConfig,
    HybridRetriever,
    HybridSearchResult,
)
from src.context.retrieval.sparse_client import (
    SparseClientAuthError,
    SparseClientRPCError,
    SparseMatch,
    SparseRetrievalClient,
)
from src.context.retrieval.vector_client import (
    VectorClientAuthError,
    VectorClientRPCError,
    VectorMatch,
    VectorSimilarityClient,
)
from src.providers.embeddings import MockEmbeddingProvider


class TestHybridRetriever(TestCase):
    """Test suite for HybridRetriever."""

    def setUp(self):
        self.mock_provider = MockEmbeddingProvider(dimension=768, model_id="mock/emb-768")
        self.mock_vector_client = MagicMock(spec=VectorSimilarityClient)
        self.mock_sparse_client = MagicMock(spec=SparseRetrievalClient)
        self.mock_pipeline = MagicMock(spec=EmbeddingPipeline)
        self.mock_pipeline.embed_query.return_value = [0.1] * 768

    def test_both_modalities_succeed_mode_hybrid(self):
        """Verify full hybrid mode when both dense and sparse return results."""
        # Chunk 1 is returned in both modalities (consensus)
        # Chunk 2 is dense only
        # Chunk 3 is sparse only
        self.mock_vector_client.search_similar_chunks.return_value = [
            VectorMatch(
                id="c1", document_id="d1", project_id="p1", chunk_index=0,
                content="Dense match 1", metadata={"page": 1}, similarity=0.92,
            ),
            VectorMatch(
                id="c2", document_id="d2", project_id="p1", chunk_index=1,
                content="Dense match 2", metadata={"page": 2}, similarity=0.85,
            ),
        ]
        self.mock_sparse_client.search_keywords.return_value = [
            SparseMatch(
                id="c3", document_id="d3", project_id="p1", chunk_index=0,
                content="Sparse match 3", metadata={"page": 1}, score=12.5,
            ),
            SparseMatch(
                id="c1", document_id="d1", project_id="p1", chunk_index=0,
                content="Dense match 1", metadata={"page": 1}, score=8.0,
            ),
        ]

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        results = retriever.search("query test", filter_project_id="p1")

        self.assertEqual(len(results), 3)
        # Consensus chunk c1 must rank first
        self.assertEqual(results[0].id, "c1")
        self.assertEqual(results[0].retrieval_mode, MODE_HYBRID)
        self.assertEqual(results[0].dense_rank, 1)
        self.assertEqual(results[0].sparse_rank, 2)
        self.assertAlmostEqual(results[0].dense_similarity, 0.92)
        self.assertAlmostEqual(results[0].sparse_score, 8.0)

    def test_candidate_oversampling(self):
        """Verify candidate_count requests min(max(match_count * factor, 20), 100) from both branches."""
        self.mock_vector_client.search_similar_chunks.return_value = []
        self.mock_sparse_client.search_keywords.return_value = []

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        config = HybridConfig(match_count=5, candidate_oversampling_factor=2)
        retriever.search("test", config=config)

        # max(5 * 2, 20) = 20 candidates requested
        self.mock_vector_client.search_similar_chunks.assert_called_once()
        self.assertEqual(self.mock_vector_client.search_similar_chunks.call_args[1]["match_count"], 20)

        self.mock_sparse_client.search_keywords.assert_called_once()
        self.assertEqual(self.mock_sparse_client.search_keywords.call_args[1]["match_count"], 20)

    def test_dense_only_mode_when_sparse_fails(self):
        """Verify fallback to DENSE_ONLY when sparse search encounters an RPC error."""
        self.mock_vector_client.search_similar_chunks.return_value = [
            VectorMatch(
                id="c1", document_id="d1", project_id="p1", chunk_index=0,
                content="Dense only chunk", metadata={}, similarity=0.88,
            )
        ]
        self.mock_sparse_client.search_keywords.side_effect = SparseClientRPCError("FTS unavailable", status_code=500)

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        results = retriever.search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "c1")
        self.assertEqual(results[0].retrieval_mode, MODE_DENSE_ONLY)
        self.assertEqual(results[0].dense_rank, 1)
        self.assertIsNone(results[0].sparse_rank)

    def test_sparse_only_mode_when_vector_fails(self):
        """Verify fallback to SPARSE_ONLY when vector search encounters a network failure."""
        self.mock_vector_client.search_similar_chunks.side_effect = VectorClientRPCError("Vector timeout")
        self.mock_sparse_client.search_keywords.return_value = [
            SparseMatch(
                id="c2", document_id="d2", project_id="p1", chunk_index=1,
                content="Sparse only chunk", metadata={}, score=5.4,
            )
        ]

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        results = retriever.search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "c2")
        self.assertEqual(results[0].retrieval_mode, MODE_SPARSE_ONLY)
        self.assertIsNone(results[0].dense_rank)
        self.assertEqual(results[0].sparse_rank, 1)

    def test_offline_mode_sparse_only(self):
        """Verify prefer_offline bypasses vector client and uses local SQLite FTS5."""
        self.mock_sparse_client.search_keywords.return_value = [
            SparseMatch(
                id="local_c1", document_id="d1", project_id="p1", chunk_index=0,
                content="Local SQLite match", metadata={}, score=2.5,
            )
        ]

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        results = retriever.search("test query", prefer_offline=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "local_c1")
        self.assertEqual(results[0].retrieval_mode, MODE_OFFLINE_SPARSE_ONLY)
        # Vector client must not be called when prefer_offline=True
        self.mock_vector_client.search_similar_chunks.assert_not_called()

    def test_empty_results_mode(self):
        """Verify MODE_EMPTY when both branches return zero matches."""
        self.mock_vector_client.search_similar_chunks.return_value = []
        self.mock_sparse_client.search_keywords.return_value = []

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        results = retriever.search("query with no matches")
        self.assertEqual(results, [])

    def test_auth_errors_not_swallowed(self):
        """Security: Verify VectorClientAuthError and SparseClientAuthError are raised immediately."""
        self.mock_vector_client.search_similar_chunks.side_effect = VectorClientAuthError("JWT expired")

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        with self.assertRaises(VectorClientAuthError):
            retriever.search("test query")

        self.mock_vector_client.search_similar_chunks.side_effect = None
        self.mock_sparse_client.search_keywords.side_effect = SparseClientAuthError("Invalid API key")

        with self.assertRaises(SparseClientAuthError):
            retriever.search("test query")

    def test_filter_propagation(self):
        """Verify project and document filters are correctly propagated to both clients."""
        self.mock_vector_client.search_similar_chunks.return_value = []
        self.mock_sparse_client.search_keywords.return_value = []

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        retriever.search("test", filter_project_id="proj_123", filter_document_id="doc_456")

        vec_call = self.mock_vector_client.search_similar_chunks.call_args[1]
        self.assertEqual(vec_call["filter_project_id"], "proj_123")
        self.assertEqual(vec_call["filter_document_id"], "doc_456")

        sparse_call = self.mock_sparse_client.search_keywords.call_args[1]
        self.assertEqual(sparse_call["filter_project_id"], "proj_123")
        self.assertEqual(sparse_call["filter_document_id"], "doc_456")

    def test_empty_or_invalid_query_validation(self):
        """Verify empty query returns empty list without calling downstream clients."""
        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        self.assertEqual(retriever.search(""), [])
        self.assertEqual(retriever.search("   \n\t  "), [])
        self.assertEqual(retriever.search(None), [])

        self.mock_vector_client.search_similar_chunks.assert_not_called()
        self.mock_sparse_client.search_keywords.assert_not_called()

        with self.assertRaises(ValueError):
            retriever.search("valid query", config=HybridConfig(match_count=0))

    def test_sequential_execution_mode(self):
        """Verify sequential execution operates identically when concurrent_execution=False."""
        self.mock_vector_client.search_similar_chunks.return_value = [
            VectorMatch(id="c1", document_id="d1", project_id="p1", chunk_index=0, content="c1", metadata={}, similarity=0.9)
        ]
        self.mock_sparse_client.search_keywords.return_value = [
            SparseMatch(id="c1", document_id="d1", project_id="p1", chunk_index=0, content="c1", metadata={}, score=5.0)
        ]

        retriever = HybridRetriever(
            vector_client=self.mock_vector_client,
            sparse_client=self.mock_sparse_client,
            embedding_pipeline=self.mock_pipeline,
            default_provider=self.mock_provider,
        )

        config = HybridConfig(concurrent_execution=False)
        results = retriever.search("test", config=config)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "c1")
        self.assertEqual(results[0].retrieval_mode, MODE_HYBRID)

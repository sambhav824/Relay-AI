"""Relay AI Universal Context Engine — Hybrid Retrieval Orchestrator.

Combines dense vector similarity search and sparse full-text keyword search
using Reciprocal Rank Fusion (RRF), concurrent execution, candidate oversampling,
and multi-tier graceful fallback.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from src.context.embeddings.pipeline import EmbeddingPipeline
from src.context.retrieval.rrf import FusedResult, ReciprocalRankFusion
from src.context.retrieval.sparse_client import (
    SparseClientAuthError,
    SparseClientError,
    SparseClientValidationError,
    SparseMatch,
    SparseRetrievalClient,
)
from src.context.retrieval.vector_client import (
    VectorClientAuthError,
    VectorClientError,
    VectorClientValidationError,
    VectorMatch,
    VectorSimilarityClient,
)
from src.providers.embeddings import EmbeddingProvider

logger = logging.getLogger("RelayAI.HybridRetriever")


# Explicit Retrieval Modes
MODE_HYBRID = "HYBRID"
MODE_DENSE_ONLY = "DENSE_ONLY"
MODE_SPARSE_ONLY = "SPARSE_ONLY"
MODE_OFFLINE_SPARSE_ONLY = "OFFLINE_SPARSE_ONLY"
MODE_EMPTY = "EMPTY"


@dataclass
class HybridConfig:
    """Configuration parameters for hybrid search execution and rank fusion."""

    k: int = 60
    weight_dense: float = 0.6
    weight_sparse: float = 0.4
    dense_threshold: float = 0.0  # Cosine similarity floor
    match_count: int = 10
    candidate_oversampling_factor: int = 2
    timeout_seconds: float = 10.0
    concurrent_execution: bool = True


@dataclass
class HybridSearchResult:
    """Final merged search result with full provenance and retrieval telemetry."""

    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_similarity: Optional[float] = None
    sparse_score: Optional[float] = None
    retrieval_mode: str = MODE_HYBRID


class HybridRetriever:
    """Coordinates dense vector and sparse keyword search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_client: Optional[VectorSimilarityClient] = None,
        sparse_client: Optional[SparseRetrievalClient] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        default_provider: Optional[EmbeddingProvider] = None,
        default_config: Optional[HybridConfig] = None,
    ) -> None:
        self.vector_client = vector_client
        self.sparse_client = sparse_client
        self.embedding_pipeline = embedding_pipeline
        self.default_provider = default_provider
        self.config = default_config or HybridConfig()

    def search(
        self,
        query: str,
        *,
        filter_project_id: Optional[str] = None,
        filter_document_id: Optional[str] = None,
        provider: Optional[EmbeddingProvider] = None,
        config: Optional[HybridConfig] = None,
        prefer_offline: bool = False,
    ) -> List[HybridSearchResult]:
        """Execute hybrid retrieval combining dense vector similarity and sparse keyword search.

        Args:
            query: User search text query.
            filter_project_id: Optional project UUID filter.
            filter_document_id: Optional document UUID filter.
            provider: Optional override for embedding provider.
            config: Optional override for hybrid configuration.
            prefer_offline: If True, operates locally using SQLite FTS5.

        Returns:
            List of HybridSearchResult instances ordered by relevance.

        Raises:
            VectorClientAuthError: If vector search explicitly fails authentication.
            SparseClientAuthError: If sparse search explicitly fails authentication.
            ValueError: If input arguments fail validation.
        """
        cfg = config or self.config
        active_provider = provider or self.default_provider

        # 1. Query Validation
        if query is None or not str(query).strip():
            return []

        cleaned_query = str(query).strip()

        if cfg.match_count < 1:
            raise ValueError(f"match_count must be at least 1, got {cfg.match_count}.")

        # 2. Determine Candidate Request Limit (Oversampling)
        candidate_count = min(
            max(cfg.match_count * cfg.candidate_oversampling_factor, 20),
            100,
        )

        dense_matches: List[VectorMatch] = []
        sparse_matches: List[SparseMatch] = []
        dense_error: Optional[Exception] = None
        sparse_error: Optional[Exception] = None

        # 3. Dense Retrieval Task
        def _run_dense() -> List[VectorMatch]:
            if prefer_offline:
                return []
            if self.vector_client is None or active_provider is None:
                return []

            # Generate query embedding
            if self.embedding_pipeline is not None:
                query_vec = self.embedding_pipeline.embed_query(cleaned_query, active_provider)
            else:
                query_vec = active_provider.embed_query(cleaned_query)

            # Search pgvector
            return self.vector_client.search_similar_chunks(
                query_vector=query_vec,
                model_id=active_provider.model_id,
                dimension=active_provider.dimension,
                match_threshold=cfg.dense_threshold,
                match_count=candidate_count,
                filter_project_id=filter_project_id,
                filter_document_id=filter_document_id,
            )

        # 4. Sparse Retrieval Task
        def _run_sparse() -> List[SparseMatch]:
            if self.sparse_client is None:
                return []
            return self.sparse_client.search_keywords(
                query=cleaned_query,
                match_count=candidate_count,
                filter_project_id=filter_project_id,
                filter_document_id=filter_document_id,
                prefer_offline=prefer_offline,
            )

        # 5. Execute Retrievers (Concurrently or Sequentially)
        if cfg.concurrent_execution and not prefer_offline and self.vector_client is not None and self.sparse_client is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(_run_dense)
                sparse_future = executor.submit(_run_sparse)
                deadline = time.monotonic() + cfg.timeout_seconds

                try:
                    dense_wait = max(0.0, deadline - time.monotonic())
                    dense_matches = dense_future.result(timeout=dense_wait)
                except Exception as exc:
                    dense_error = exc
                    # Re-raise authentication errors immediately; do not convert to fallback
                    if isinstance(exc, (VectorClientAuthError, VectorClientValidationError)):
                        sparse_future.cancel()
                        raise

                try:
                    sparse_wait = max(0.0, deadline - time.monotonic())
                    sparse_matches = sparse_future.result(timeout=sparse_wait)
                except Exception as exc:
                    sparse_error = exc
                    if isinstance(exc, (SparseClientAuthError, SparseClientValidationError)):
                        dense_future.cancel()
                        raise
        else:
            # Sequential execution
            try:
                dense_matches = _run_dense()
            except Exception as exc:
                dense_error = exc
                if isinstance(exc, (VectorClientAuthError, VectorClientValidationError)):
                    raise

            try:
                sparse_matches = _run_sparse()
            except Exception as exc:
                sparse_error = exc
                if isinstance(exc, (SparseClientAuthError, SparseClientValidationError)):
                    raise

        # 6. Evaluate Result Modality & Mode
        dense_available = bool(dense_matches) and (dense_error is None)
        sparse_available = bool(sparse_matches) and (sparse_error is None)

        if prefer_offline:
            mode = MODE_OFFLINE_SPARSE_ONLY if sparse_available else MODE_EMPTY
        elif dense_available and sparse_available:
            mode = MODE_HYBRID
        elif dense_available and not sparse_available:
            mode = MODE_DENSE_ONLY
        elif sparse_available and not dense_available:
            mode = MODE_SPARSE_ONLY
        else:
            mode = MODE_EMPTY

        # If both branches encountered errors and no matches exist, re-raise first error
        if not dense_available and not sparse_available:
            if dense_error and sparse_error:
                logger.error("Both dense and sparse retrieval failed: Dense=%s, Sparse=%s", dense_error, sparse_error)
            return []

        # 7. Merge Rankings using Reciprocal Rank Fusion
        rrf = ReciprocalRankFusion(
            k=cfg.k,
            weight_dense=cfg.weight_dense if dense_available else 0.0,
            weight_sparse=cfg.weight_sparse if sparse_available else 0.0,
        )

        fused = rrf.fuse(
            dense_results=dense_matches,
            sparse_results=sparse_matches,
            limit=cfg.match_count,
        )

        # 8. Transform to HybridSearchResult
        results: List[HybridSearchResult] = []
        for item in fused:
            results.append(
                HybridSearchResult(
                    id=item.id,
                    document_id=item.document_id or "",
                    project_id=item.project_id,
                    chunk_index=item.chunk_index,
                    content=item.content,
                    metadata=item.metadata,
                    rrf_score=item.rrf_score,
                    dense_rank=item.dense_rank,
                    sparse_rank=item.sparse_rank,
                    dense_similarity=item.dense_score,
                    sparse_score=item.sparse_score,
                    retrieval_mode=mode,
                )
            )

        return results

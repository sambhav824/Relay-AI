"""Relay AI Universal Context Engine — Vector & Hybrid Retrieval Package."""

from __future__ import annotations

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
from src.context.retrieval.rrf import (
    FusedResult,
    ReciprocalRankFusion,
)
from src.context.retrieval.sparse_client import (
    SparseClientAuthError,
    SparseClientError,
    SparseClientRPCError,
    SparseClientValidationError,
    SparseMatch,
    SparseRetrievalClient,
)
from src.context.retrieval.vector_client import (
    VectorClientAuthError,
    VectorClientError,
    VectorClientRPCError,
    VectorClientValidationError,
    VectorMatch,
    VectorSimilarityClient,
)

__all__ = [
    # Vector retrieval
    "VectorSimilarityClient",
    "VectorMatch",
    "VectorClientError",
    "VectorClientAuthError",
    "VectorClientValidationError",
    "VectorClientRPCError",
    # Sparse retrieval
    "SparseRetrievalClient",
    "SparseMatch",
    "SparseClientError",
    "SparseClientAuthError",
    "SparseClientValidationError",
    "SparseClientRPCError",
    # Rank fusion
    "ReciprocalRankFusion",
    "FusedResult",
    # Hybrid retrieval
    "HybridRetriever",
    "HybridConfig",
    "HybridSearchResult",
    "MODE_HYBRID",
    "MODE_DENSE_ONLY",
    "MODE_SPARSE_ONLY",
    "MODE_OFFLINE_SPARSE_ONLY",
    "MODE_EMPTY",
]

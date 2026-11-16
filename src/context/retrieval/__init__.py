"""Relay AI Universal Context Engine — Vector & Hybrid Retrieval Package."""

from __future__ import annotations

from src.context.retrieval.vector_client import (
    VectorClientAuthError,
    VectorClientError,
    VectorClientRPCError,
    VectorClientValidationError,
    VectorMatch,
    VectorSimilarityClient,
)

__all__ = [
    "VectorSimilarityClient",
    "VectorMatch",
    "VectorClientError",
    "VectorClientAuthError",
    "VectorClientValidationError",
    "VectorClientRPCError",
]

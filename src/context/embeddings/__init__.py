"""Relay AI Universal Context Engine — Vector Embeddings & Cache Package."""

from __future__ import annotations

from src.context.embeddings.cache import (
    EmbeddingCache,
    EmbeddingCacheRecord,
    compute_cache_key,
    deserialize_embedding,
    serialize_embedding,
)
from src.context.embeddings.pipeline import (
    ChunkItem,
    EmbeddedChunk,
    EmbeddingPipeline,
    EmbeddingResult,
    FailedChunk,
    PipelineProgress,
)
from src.context.embeddings.rate_limiter import RateLimitCancelledError, RateLimiter
from src.context.embeddings.recovery import (
    EmbeddingRecoveryManager,
    update_chunk_embedding_metadata,
)
from src.context.embeddings.retry import (
    ProviderAuthError,
    ProviderNotFoundError,
    RetryAfterExceededError,
    RetryConfig,
    extract_retry_after,
)

__all__ = [
    "EmbeddingCache",
    "EmbeddingCacheRecord",
    "compute_cache_key",
    "serialize_embedding",
    "deserialize_embedding",
    "ChunkItem",
    "EmbeddedChunk",
    "FailedChunk",
    "PipelineProgress",
    "EmbeddingResult",
    "EmbeddingPipeline",
    "RateLimiter",
    "RateLimitCancelledError",
    "RetryConfig",
    "RetryAfterExceededError",
    "extract_retry_after",
    "ProviderAuthError",
    "ProviderNotFoundError",
    "EmbeddingRecoveryManager",
    "update_chunk_embedding_metadata",
]

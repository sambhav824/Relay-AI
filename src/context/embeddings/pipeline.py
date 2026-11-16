"""Relay AI Universal Context Engine — Batch Embedding Pipeline.

Coordinates cache-first embedding generation, batch chunking, rate limiting,
exponential backoff with jitter, order preservation, and progress tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import math
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.context.embeddings.cache import EmbeddingCache
from src.context.embeddings.rate_limiter import RateLimitCancelledError, RateLimiter
from src.context.embeddings.retry import RetryAfterExceededError, RetryConfig
from src.providers.embeddings import (
    EmbeddingProvider,
    normalize_vector,
    validate_vector,
)

logger = logging.getLogger("RelayAI.EmbeddingPipeline")


@dataclass
class ChunkItem:
    """Standardized internal representation of an input text chunk."""

    chunk_id: str
    content: str
    index: int = 0
    document_id: Optional[str] = None
    page_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """Represents a chunk that has been successfully embedded."""

    chunk_id: str
    content: str
    index: int
    embedding: List[float]
    from_cache: bool
    document_id: Optional[str] = None
    page_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailedChunk:
    """Represents a chunk whose embedding generation failed."""

    chunk_id: str
    content: str
    index: int
    error: str
    retry_count: int = 0
    document_id: Optional[str] = None
    page_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineProgress:
    """Live progress telemetry for the embedding pipeline."""

    total_chunks: int = 0
    cached_chunks: int = 0
    pending_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    current_batch: int = 0
    total_batches: int = 0
    retry_count: int = 0
    cancelled: bool = False

    @property
    def completion_percentage(self) -> float:
        if self.total_chunks == 0:
            return 100.0
        done = self.successful_chunks + self.failed_chunks
        return min(100.0, (done / self.total_chunks) * 100.0)


@dataclass
class EmbeddingResult:
    """Structured result returned by the embedding pipeline."""

    successful: List[EmbeddedChunk] = field(default_factory=list)
    failed: List[FailedChunk] = field(default_factory=list)
    total_chunks: int = 0
    cached_count: int = 0
    generated_count: int = 0
    cancelled: bool = False
    error: Optional[str] = None

    @property
    def is_partial(self) -> bool:
        return bool(self.successful and self.failed)

    @property
    def is_success(self) -> bool:
        return bool(self.successful and not self.failed and not self.cancelled)

    @property
    def is_complete_success(self) -> bool:
        return not self.failed and not self.cancelled



def _normalize_chunk_input(chunk: Any, default_index: int = 0) -> ChunkItem:
    """Convert input types (ChunkItem, dict, DocumentChunk, str) to ChunkItem."""
    if isinstance(chunk, ChunkItem):
        return chunk

    # Object with attributes (e.g. DocumentChunk)
    if hasattr(chunk, "content"):
        chunk_id = getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None) or str(uuid.uuid4())
        idx = getattr(chunk, "index", default_index)
        doc_id = getattr(chunk, "document_id", None)
        page_num = getattr(chunk, "page_number", 1)
        meta = getattr(chunk, "metadata", {}) or {}
        return ChunkItem(
            chunk_id=str(chunk_id),
            content=chunk.content,
            index=idx,
            document_id=doc_id,
            page_number=page_num,
            metadata=meta,
        )

    # Dictionary
    if isinstance(chunk, dict):
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or str(uuid.uuid4())
        content = chunk.get("content", "")
        idx = chunk.get("index", chunk.get("chunk_index", default_index))
        doc_id = chunk.get("document_id")
        page_num = chunk.get("page_number", 1)
        meta = chunk.get("metadata", {}) or {}
        return ChunkItem(
            chunk_id=str(chunk_id),
            content=content,
            index=idx,
            document_id=doc_id,
            page_number=page_num,
            metadata=meta,
        )

    # Raw string
    if isinstance(chunk, str):
        return ChunkItem(
            chunk_id=str(uuid.uuid4()),
            content=chunk,
            index=default_index,
        )

    raise TypeError(f"Unsupported chunk type: {type(chunk)}")


class EmbeddingPipeline:
    """Coordinates batching, cache-first lookup, rate limiting, and backoff retries."""

    def __init__(
        self,
        cache: Optional[EmbeddingCache] = None,
        retry_config: Optional[RetryConfig] = None,
        rate_limiters: Optional[Dict[str, RateLimiter]] = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.cache = cache
        self.retry_config = retry_config or RetryConfig()
        self.batch_size = batch_size
        self._rate_limiters = rate_limiters or {
            "openai": RateLimiter(requests_per_minute=300),
            "ollama": RateLimiter(requests_per_minute=120),
            "mock": RateLimiter(requests_per_minute=10000),
        }
        self._cancellation_event = threading.Event()
        self._is_cancelled = False
        self._current_progress = PipelineProgress()

    def cancel(self) -> None:
        """Signal the pipeline to abort pending batches safely."""
        self._is_cancelled = True
        self._cancellation_event.set()
        self._current_progress.cancelled = True
        logger.info("Embedding pipeline cancellation requested.")

    def reset_cancellation(self) -> None:
        """Reset cancellation flag before starting a new run."""
        self._is_cancelled = False
        self._cancellation_event.clear()
        self._current_progress.cancelled = False

    def get_progress(self) -> PipelineProgress:
        """Return a snapshot of current pipeline progress."""
        return PipelineProgress(
            total_chunks=self._current_progress.total_chunks,
            cached_chunks=self._current_progress.cached_chunks,
            pending_chunks=self._current_progress.pending_chunks,
            successful_chunks=self._current_progress.successful_chunks,
            failed_chunks=self._current_progress.failed_chunks,
            current_batch=self._current_progress.current_batch,
            total_batches=self._current_progress.total_batches,
            retry_count=self._current_progress.retry_count,
            cancelled=self._current_progress.cancelled,
        )

    def _get_rate_limiter(self, provider_id: str) -> Optional[RateLimiter]:
        return self._rate_limiters.get(provider_id.lower())

    def embed_query(
        self,
        text: str,
        provider: EmbeddingProvider,
        preprocessing_version: str = "v1",
        normalize: bool = True,
    ) -> List[float]:
        """Generate an embedding for a single query text with cache-first lookup."""
        cleaned = text.strip() if text else ""
        if not cleaned:
            raise ValueError("Query text cannot be empty or purely whitespace.")

        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        # 1. Check cache
        if self.cache is not None:
            cached = self.cache.get(
                content_hash=content_hash,
                provider_id=provider.provider_id,
                model_id=provider.model_id,
                model_version=provider.model_version,
                dimension=provider.dimension,
                preprocessing_version=preprocessing_version,
                normalize=normalize,
            )
            if cached is not None:
                return cached

        # 2. Rate limit
        limiter = self._get_rate_limiter(provider.provider_id)
        if limiter is not None:
            limiter.acquire(cancellation_event=self._cancellation_event)

        # 3. Call provider with retry loop
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                vector = provider.embed_query(cleaned)
                validated = validate_vector(vector, expected_dimension=provider.dimension)
                res = normalize_vector(validated) if normalize else validated

                # Store in cache
                if self.cache is not None:
                    self.cache.set(
                        content_hash=content_hash,
                        provider_id=provider.provider_id,
                        model_id=provider.model_id,
                        model_version=provider.model_version,
                        dimension=provider.dimension,
                        embedding=res,
                        preprocessing_version=preprocessing_version,
                        normalize=normalize,
                    )
                return res

            except Exception as exc:
                last_exc = exc
                if not self.retry_config.is_retryable(exc) or attempt >= self.retry_config.max_retries:
                    raise

                try:
                    delay = self.retry_config.compute_delay(attempt, exc=exc)
                except RetryAfterExceededError:
                    raise

                logger.warning(
                    "Query embedding attempt %d failed: %s. Retrying in %.3fs...",
                    attempt + 1,
                    exc,
                    delay,
                )
                self.retry_config.sleep_fn(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected termination of query embedding retry loop.")

    def embed_chunks(
        self,
        chunks: Sequence[Any],
        provider: EmbeddingProvider,
        preprocessing_version: str = "v1",
        normalize: bool = True,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
        on_batch_complete: Optional[Callable[[List[EmbeddedChunk], List[FailedChunk]], None]] = None,
    ) -> EmbeddingResult:
        """Batch-embed document chunks, using the cache for hits and generating only for misses.

        Preserves the exact original order of chunks in the returned result.
        Calls `on_batch_complete(batch_successful, batch_failed)` incrementally.
        """
        self.reset_cancellation()

        if not chunks:
            return EmbeddingResult(total_chunks=0)

        normalized_chunks = [_normalize_chunk_input(c, default_index=i) for i, c in enumerate(chunks)]
        total_chunks = len(normalized_chunks)

        # Pre-allocate ordered output slot for every chunk to preserve original ordering
        ordered_results: List[Optional[EmbeddedChunk]] = [None] * total_chunks
        failed_chunks: List[FailedChunk] = []

        cache_misses: List[tuple[int, ChunkItem, str]] = []  # (orig_idx, chunk, content_hash)
        cached_count = 0

        # Step 1: Cache-first lookup (does not make network calls)
        initial_successful_cached: List[EmbeddedChunk] = []
        for i, chunk in enumerate(normalized_chunks):
            content_cleaned = chunk.content.strip()
            if not content_cleaned:
                fc = FailedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    index=chunk.index,
                    error="Chunk content is empty or purely whitespace.",
                    retry_count=0,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
                failed_chunks.append(fc)
                continue

            content_hash = hashlib.sha256(content_cleaned.encode("utf-8")).hexdigest()

            cached_vector: Optional[List[float]] = None
            if self.cache is not None:
                cached_vector = self.cache.get(
                    content_hash=content_hash,
                    provider_id=provider.provider_id,
                    model_id=provider.model_id,
                    model_version=provider.model_version,
                    dimension=provider.dimension,
                    preprocessing_version=preprocessing_version,
                    normalize=normalize,
                )

            if cached_vector is not None:
                ec = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    index=chunk.index,
                    embedding=cached_vector,
                    from_cache=True,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
                ordered_results[i] = ec
                initial_successful_cached.append(ec)
                cached_count += 1
            else:
                cache_misses.append((i, chunk, content_hash))

        # Notify initial cached chunks if callback provided
        if on_batch_complete and initial_successful_cached:
            try:
                on_batch_complete(initial_successful_cached, [])
            except Exception as cb_exc:
                logger.warning("on_batch_complete callback raised during cache evaluation: %s", cb_exc)

        # Setup initial progress state
        total_misses = len(cache_misses)
        num_batches = math.ceil(total_misses / self.batch_size) if total_misses > 0 else 0

        self._current_progress = PipelineProgress(
            total_chunks=total_chunks,
            cached_chunks=cached_count,
            pending_chunks=total_misses,
            successful_chunks=cached_count,
            failed_chunks=len(failed_chunks),
            current_batch=0,
            total_batches=num_batches,
            retry_count=0,
            cancelled=False,
        )

        if progress_callback:
            progress_callback(self.get_progress())

        # If everything was cached or empty, return immediately
        if not cache_misses:
            successful = [c for c in ordered_results if c is not None]
            return EmbeddingResult(
                successful=successful,
                failed=failed_chunks,
                total_chunks=total_chunks,
                cached_count=cached_count,
                generated_count=0,
                cancelled=False,
            )

        # Step 2: Batch processing for cache misses
        limiter = self._get_rate_limiter(provider.provider_id)

        for batch_num in range(num_batches):
            if self._is_cancelled:
                logger.info("Halting batch processing due to cancellation.")
                cancelled_slice = cache_misses[batch_num * self.batch_size :]
                batch_cancelled: List[FailedChunk] = []
                for orig_idx, chunk, _ in cancelled_slice:
                    fc = FailedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        index=chunk.index,
                        error="Embedding job was cancelled by user.",
                        retry_count=0,
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        metadata=chunk.metadata,
                    )
                    failed_chunks.append(fc)
                    batch_cancelled.append(fc)
                if on_batch_complete and batch_cancelled:
                    try:
                        on_batch_complete([], batch_cancelled)
                    except Exception as cb_exc:
                        logger.warning("on_batch_complete callback raised: %s", cb_exc)
                break

            self._current_progress.current_batch = batch_num + 1
            batch_slice = cache_misses[batch_num * self.batch_size : (batch_num + 1) * self.batch_size]
            batch_texts = [item[1].content.strip() for item in batch_slice]

            # Apply rate limiting before dispatching HTTP request
            if limiter is not None:
                try:
                    limiter.acquire(cancellation_event=self._cancellation_event)
                except RateLimitCancelledError:
                    self._is_cancelled = True
                    break

            # Execute batch embedding with retry loop
            batch_success = False
            last_batch_error = ""
            retries_used = 0
            batch_successful_chunks: List[EmbeddedChunk] = []
            batch_failed_chunks: List[FailedChunk] = []

            for attempt in range(self.retry_config.max_retries + 1):
                if self._is_cancelled:
                    break

                try:
                    generated_vectors = provider.embed_documents(batch_texts)
                    if len(generated_vectors) != len(batch_slice):
                        raise ValueError(
                            f"Provider returned {len(generated_vectors)} vectors for {len(batch_slice)} texts."
                        )

                    # Store successful embeddings into cache and output slots
                    for (orig_idx, chunk, content_hash), raw_vec in zip(batch_slice, generated_vectors):
                        validated = validate_vector(raw_vec, expected_dimension=provider.dimension)
                        norm_vec = normalize_vector(validated) if normalize else validated

                        if self.cache is not None:
                            self.cache.set(
                                content_hash=content_hash,
                                provider_id=provider.provider_id,
                                model_id=provider.model_id,
                                model_version=provider.model_version,
                                dimension=provider.dimension,
                                embedding=norm_vec,
                                preprocessing_version=preprocessing_version,
                                normalize=normalize,
                            )

                        ec = EmbeddedChunk(
                            chunk_id=chunk.chunk_id,
                            content=chunk.content,
                            index=chunk.index,
                            embedding=norm_vec,
                            from_cache=False,
                            document_id=chunk.document_id,
                            page_number=chunk.page_number,
                            metadata=chunk.metadata,
                        )
                        ordered_results[orig_idx] = ec
                        batch_successful_chunks.append(ec)
                        self._current_progress.successful_chunks += 1
                        self._current_progress.pending_chunks -= 1

                    batch_success = True
                    break

                except Exception as exc:
                    last_batch_error = str(exc)
                    retries_used = attempt + 1

                    if not self.retry_config.is_retryable(exc) or attempt >= self.retry_config.max_retries:
                        logger.error(
                            "Batch %d/%d failed permanently: %s",
                            batch_num + 1,
                            num_batches,
                            exc,
                        )
                        break

                    self._current_progress.retry_count += 1
                    try:
                        delay = self.retry_config.compute_delay(attempt, exc=exc)
                    except RetryAfterExceededError as ra_err:
                        logger.warning(
                            "Batch %d/%d deferred: %s",
                            batch_num + 1,
                            num_batches,
                            ra_err,
                        )
                        last_batch_error = str(ra_err)
                        retries_used = 1
                        break

                    logger.warning(
                        "Batch %d/%d attempt %d failed: %s. Retrying in %.3fs...",
                        batch_num + 1,
                        num_batches,
                        attempt + 1,
                        exc,
                        delay,
                    )
                    self.retry_config.sleep_fn(delay)

            if not batch_success:
                # Record failure for all chunks in this batch
                for orig_idx, chunk, _ in batch_slice:
                    fc = FailedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        index=chunk.index,
                        error=last_batch_error or "Batch embedding failed.",
                        retry_count=retries_used,
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        metadata=chunk.metadata,
                    )
                    failed_chunks.append(fc)
                    batch_failed_chunks.append(fc)
                    self._current_progress.failed_chunks += 1
                    self._current_progress.pending_chunks -= 1

            if on_batch_complete:
                try:
                    on_batch_complete(batch_successful_chunks, batch_failed_chunks)
                except Exception as cb_exc:
                    logger.warning("on_batch_complete callback raised: %s", cb_exc)

            if progress_callback:
                progress_callback(self.get_progress())

        # Collect successful chunks in original order
        successful_ordered = [c for c in ordered_results if c is not None]

        return EmbeddingResult(
            successful=successful_ordered,
            failed=failed_chunks,
            total_chunks=total_chunks,
            cached_count=cached_count,
            generated_count=len(successful_ordered) - cached_count,
            cancelled=self._is_cancelled,
            error=failed_chunks[0].error if failed_chunks else None,
        )

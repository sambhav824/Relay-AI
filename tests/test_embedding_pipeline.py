"""Unit tests for Relay AI Embedding Pipeline (Milestone 10.4, Phase 2).

Tests batching (size 32), original chunk ordering, cache-first avoidance of API calls,
partial cache hits, retry/backoff on 429/5xx, Retry-After compliance, non-retryable auth errors,
rate limiting, progress tracking, and cancellation.
"""

from __future__ import annotations

import math
import sqlite3
from unittest import TestCase
from unittest.mock import MagicMock

from src.context.embeddings.cache import EmbeddingCache
from src.context.embeddings.pipeline import (
    ChunkItem,
    EmbeddingPipeline,
    PipelineProgress,
)
from src.context.embeddings.rate_limiter import RateLimitCancelledError, RateLimiter
from src.context.embeddings.retry import (
    ProviderAuthError,
    ProviderNotFoundError,
    RetryAfterExceededError,
    RetryConfig,
    extract_retry_after,
    parse_retry_after_header,
)
from src.providers.embeddings import (
    MockEmbeddingProvider,
)
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
import datetime
import threading
import time



class TestEmbeddingPipeline(TestCase):
    """Unit test suite for EmbeddingPipeline."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.cache = EmbeddingCache(self.conn)
        self.sleep_calls: list[float] = []

        def mock_sleep(duration: float) -> None:
            self.sleep_calls.append(duration)

        self.mock_sleep = mock_sleep

        self.retry_config = RetryConfig(
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
            jitter_ratio=0.0,  # deterministic for testing
            sleep_fn=self.mock_sleep,
        )

        self.pipeline = EmbeddingPipeline(
            cache=self.cache,
            retry_config=self.retry_config,
            batch_size=32,
        )

    def tearDown(self):
        self.conn.close()

    def test_empty_input_handling(self):
        provider = MockEmbeddingProvider()
        result = self.pipeline.embed_chunks([], provider)
        self.assertEqual(result.total_chunks, 0)
        self.assertEqual(len(result.successful), 0)
        self.assertEqual(len(result.failed), 0)
        self.assertTrue(result.is_complete_success)

    def test_batch_size_32_and_ordering(self):
        """Verify 70 chunks are sliced into batches of 32, 32, 6, and order is strictly preserved."""
        provider = MockEmbeddingProvider(dimension=768)
        spy_embed = MagicMock(wraps=provider.embed_documents)
        provider.embed_documents = spy_embed

        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"Document paragraph text {i}", index=i)
            for i in range(70)
        ]

        result = self.pipeline.embed_chunks(chunks, provider)
        self.assertEqual(result.total_chunks, 70)
        self.assertEqual(len(result.successful), 70)
        self.assertEqual(len(result.failed), 0)
        self.assertTrue(result.is_complete_success)

        # Provider must be called 3 times (ceil(70 / 32) == 3)
        self.assertEqual(spy_embed.call_count, 3)
        self.assertEqual(len(spy_embed.call_args_list[0][0][0]), 32)
        self.assertEqual(len(spy_embed.call_args_list[1][0][0]), 32)
        self.assertEqual(len(spy_embed.call_args_list[2][0][0]), 6)

        # Verify output order matches exact original indices
        for i, sc in enumerate(result.successful):
            self.assertEqual(sc.chunk_id, f"c_{i}")
            self.assertEqual(sc.index, i)
            self.assertEqual(sc.content, f"Document paragraph text {i}")
            self.assertFalse(sc.from_cache)
            self.assertEqual(len(sc.embedding), 768)

    def test_cache_first_no_api_calls_for_hits(self):
        """Verify that when chunks already reside in SQLite cache, zero provider calls are made."""
        provider = MockEmbeddingProvider(dimension=768)
        spy_embed = MagicMock(wraps=provider.embed_documents)
        provider.embed_documents = spy_embed

        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"Cached content text {i}", index=i)
            for i in range(5)
        ]

        # First run: populates the cache
        res1 = self.pipeline.embed_chunks(chunks, provider)
        self.assertEqual(spy_embed.call_count, 1)
        self.assertEqual(res1.cached_count, 0)
        self.assertEqual(res1.generated_count, 5)

        spy_embed.reset_mock()

        # Second run: must hit cache for 100% of chunks
        res2 = self.pipeline.embed_chunks(chunks, provider)
        self.assertEqual(spy_embed.call_count, 0)  # Zero provider network calls!
        self.assertEqual(res2.cached_count, 5)
        self.assertEqual(res2.generated_count, 0)
        self.assertEqual(len(res2.successful), 5)
        for sc in res2.successful:
            self.assertTrue(sc.from_cache)

    def test_partial_cache_hits_and_order_preservation(self):
        """Verify mixture of cached and uncached chunks: only misses go to API, original order preserved."""
        provider = MockEmbeddingProvider(dimension=768)
        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"Chunk payload {i}", index=i)
            for i in range(10)
        ]

        # Pre-seed chunks 1, 3, 7 into cache
        pre_seeded = [chunks[1], chunks[3], chunks[7]]
        self.pipeline.embed_chunks(pre_seeded, provider)

        spy_embed = MagicMock(wraps=provider.embed_documents)
        provider.embed_documents = spy_embed

        # Now embed all 10 chunks
        result = self.pipeline.embed_chunks(chunks, provider)
        self.assertEqual(result.total_chunks, 10)
        self.assertEqual(result.cached_count, 3)
        self.assertEqual(result.generated_count, 7)

        # Provider must only receive the 7 misses
        self.assertEqual(spy_embed.call_count, 1)
        sent_texts = spy_embed.call_args_list[0][0][0]
        self.assertEqual(len(sent_texts), 7)
        self.assertNotIn("Chunk payload 1", sent_texts)
        self.assertNotIn("Chunk payload 3", sent_texts)
        self.assertNotIn("Chunk payload 7", sent_texts)

        # All 10 must be returned in exact 0..9 order
        self.assertEqual(len(result.successful), 10)
        for i, sc in enumerate(result.successful):
            self.assertEqual(sc.index, i)
            self.assertEqual(sc.chunk_id, f"c_{i}")
            if i in (1, 3, 7):
                self.assertTrue(sc.from_cache)
            else:
                self.assertFalse(sc.from_cache)

    def test_retry_on_http_429_rate_limit(self):
        """Verify transient 429 rate limit is retried with backoff delay."""
        provider = MockEmbeddingProvider(dimension=768)

        call_count = 0
        original_embed = provider.embed_documents

        def flaky_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderRateLimitError("429 Too Many Requests: Rate limit reached", provider_id="openai")
            return original_embed(texts)

        provider.embed_documents = flaky_embed

        chunks = [ChunkItem(chunk_id="c_1", content="Text to embed", index=0)]
        result = self.pipeline.embed_chunks(chunks, provider)

        self.assertEqual(call_count, 2)
        self.assertEqual(len(result.successful), 1)
        self.assertEqual(len(result.failed), 0)
        # Sleep was called once with base_delay (1.0s)
        self.assertEqual(len(self.sleep_calls), 1)
        self.assertEqual(self.sleep_calls[0], 1.0)

    def test_respect_retry_after(self):
        """Verify Retry-After header/attribute on exception overrides exponential backoff."""
        provider = MockEmbeddingProvider(dimension=768)

        call_count = 0
        original_embed = provider.embed_documents

        def retry_after_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exc = ProviderRateLimitError("Rate limit reached. Please try again in 5.5s", provider_id="openai")
                raise exc
            return original_embed(texts)

        provider.embed_documents = retry_after_embed

        chunks = [ChunkItem(chunk_id="c_1", content="Text with retry after", index=0)]
        result = self.pipeline.embed_chunks(chunks, provider)

        self.assertEqual(call_count, 2)
        self.assertEqual(len(result.successful), 1)
        self.assertEqual(len(self.sleep_calls), 1)
        self.assertEqual(self.sleep_calls[0], 5.5)

    def test_retry_on_transient_5xx_errors(self):
        """Verify 502/503/504 errors are retried."""
        provider = MockEmbeddingProvider(dimension=768)

        call_count = 0
        original_embed = provider.embed_documents

        def error_502_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ProviderError("HTTP 502 Bad Gateway: Upstream failure", provider_id="openai")
            return original_embed(texts)

        provider.embed_documents = error_502_embed

        chunks = [ChunkItem(chunk_id="c_1", content="502 test text", index=0)]
        result = self.pipeline.embed_chunks(chunks, provider)

        self.assertEqual(call_count, 3)
        self.assertEqual(len(result.successful), 1)
        self.assertEqual(len(self.sleep_calls), 2)  # Retried twice

    def test_auth_errors_not_retried(self):
        """Verify 401 Authentication errors fail immediately without wasteful retries."""
        provider = MockEmbeddingProvider(dimension=768)

        call_count = 0

        def auth_error_embed(texts):
            nonlocal call_count
            call_count += 1
            raise ProviderAuthenticationError("401 Unauthorized: Invalid API Key", provider_id="openai")

        provider.embed_documents = auth_error_embed

        chunks = [ChunkItem(chunk_id="c_1", content="Auth test text", index=0)]
        result = self.pipeline.embed_chunks(chunks, provider)

        # Call count is exactly 1 (no retries!)
        self.assertEqual(call_count, 1)
        self.assertEqual(len(result.successful), 0)
        self.assertEqual(len(result.failed), 1)
        self.assertIn("Invalid API Key", result.failed[0].error)
        self.assertEqual(len(self.sleep_calls), 0)

    def test_rate_limiting_throttles_requests(self):
        """Verify RateLimiter throttles requests exceeding configured RPM."""
        simulated_time = 100.0

        def mock_time():
            nonlocal simulated_time
            return simulated_time

        sleep_amounts: list[float] = []

        def mock_limiter_sleep(seconds: float):
            nonlocal simulated_time
            sleep_amounts.append(seconds)
            simulated_time += seconds

        # Rate limiter configured with 2 requests per minute
        limiter = RateLimiter(
            requests_per_minute=2,
            time_fn=mock_time,
            sleep_fn=mock_limiter_sleep,
        )

        pipeline = EmbeddingPipeline(
            cache=self.cache,
            retry_config=self.retry_config,
            rate_limiters={"mock": limiter},
            batch_size=1,
        )
        provider = MockEmbeddingProvider()

        # 3 chunks with batch_size=1 will make 3 requests
        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"RL Text {i}", index=i)
            for i in range(3)
        ]
        res = pipeline.embed_chunks(chunks, provider)
        self.assertEqual(len(res.successful), 3)

        # Request 1: immediate. Request 2: immediate. Request 3: must throttle!
        self.assertEqual(len(sleep_amounts), 1)
        self.assertAlmostEqual(sleep_amounts[0], 60.0)

    def test_progress_tracking(self):
        """Verify progress updates after batches."""
        provider = MockEmbeddingProvider(dimension=768)
        progress_snapshots: list[PipelineProgress] = []

        def on_progress(p: PipelineProgress):
            progress_snapshots.append(p)

        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"Progress chunk {i}", index=i)
            for i in range(35)
        ]

        result = self.pipeline.embed_chunks(
            chunks,
            provider,
            progress_callback=on_progress,
        )
        self.assertEqual(result.total_chunks, 35)
        self.assertGreater(len(progress_snapshots), 1)

        final = progress_snapshots[-1]
        self.assertEqual(final.successful_chunks, 35)
        self.assertEqual(final.failed_chunks, 0)
        self.assertEqual(final.completion_percentage, 100.0)

    def test_cancellation(self):
        """Verify that cancel() stops pending batches and returns cancelled status."""
        provider = MockEmbeddingProvider(dimension=768)

        # 64 chunks = 2 batches of 32
        chunks = [
            ChunkItem(chunk_id=f"c_{i}", content=f"Cancel test chunk {i}", index=i)
            for i in range(64)
        ]

        def on_progress(p: PipelineProgress):
            if p.current_batch == 1:
                # Cancel after batch 1 finishes
                self.pipeline.cancel()

        result = self.pipeline.embed_chunks(
            chunks,
            provider,
            progress_callback=on_progress,
        )
        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.successful), 32)
        self.assertEqual(len(result.failed), 32)
        self.assertIn("cancelled", result.failed[0].error.lower())

    def test_embed_query_cache_and_provider(self):
        """Verify embed_query checks cache first and caches newly generated query vector."""
        provider = MockEmbeddingProvider(dimension=768)
        spy_query = MagicMock(wraps=provider.embed_query)
        provider.embed_query = spy_query

        # First query: calls provider
        v1 = self.pipeline.embed_query("Search question", provider)
        self.assertEqual(len(v1), 768)
        self.assertEqual(spy_query.call_count, 1)

        # Second query: hits cache
        v2 = self.pipeline.embed_query("Search question", provider)
        self.assertEqual(len(v1), len(v2))
        for x1, x2 in zip(v1, v2):
            self.assertAlmostEqual(x1, x2, places=5)
        self.assertEqual(spy_query.call_count, 1)  # No second provider call!

        # Empty query raises ValueError
        with self.assertRaises(ValueError):
            self.pipeline.embed_query("", provider)

    def test_retry_after_numeric_seconds(self):
        """Verify numeric Retry-After is parsed and backoff never retries earlier than requested."""
        err = ProviderRateLimitError("Rate limit reached. Retry-After: 4.5", provider_id="openai")
        extracted = extract_retry_after(err)
        self.assertEqual(extracted, 4.5)

        # compute_delay with jitter should never return less than 4.5
        cfg = RetryConfig(max_delay=30.0, jitter_ratio=0.2)
        delay = cfg.compute_delay(0, exc=err)
        self.assertGreaterEqual(delay, 4.5)

    def test_retry_after_http_date(self):
        """Verify RFC 7231 / RFC 2822 HTTP-date string is parsed accurately."""
        # Simulated now: 2026-11-16 10:00:00 UTC (timestamp: 1794823200.0)
        simulated_now = 1794823200.0
        # Target header: 15 seconds in future
        date_header = "Mon, 16 Nov 2026 10:00:15 GMT"
        parsed = parse_retry_after_header(date_header, now_fn=lambda: simulated_now)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed, 15.0, places=1)

    def test_retry_after_exceeds_max_delay_defers_without_early_retry(self):
        """Verify that when Retry-After exceeds max_delay, job is deferred without early retry."""
        long_wait_err = ProviderRateLimitError("Rate limited. Retry-After: 120", provider_id="openai")

        cfg = RetryConfig(max_delay=10.0, sleep_fn=self.mock_sleep)
        with self.assertRaises(RetryAfterExceededError) as ctx:
            cfg.compute_delay(0, exc=long_wait_err)

        self.assertEqual(ctx.exception.retry_after, 120.0)
        self.assertEqual(ctx.exception.max_delay, 10.0)

        # Pipeline test: when provider raises RetryAfter > max_delay, batch is deferred without sleeping
        provider = MockEmbeddingProvider()
        provider.embed_documents = MagicMock(side_effect=long_wait_err)
        chunks = [ChunkItem(chunk_id="c_defer", content="Deferred chunk content", index=0)]

        res = self.pipeline.embed_chunks(chunks, provider)
        self.assertEqual(len(res.failed), 1)
        self.assertIn("exceeds maximum permitted synchronous wait", res.failed[0].error)
        # MUST NOT have retried early!
        self.assertEqual(len(self.sleep_calls), 0)

    def test_exception_compatibility_auth_and_not_found_fail_fast(self):
        """Verify auth and model errors (and their aliases) fail fast even if messages contain 'connection'."""
        cfg = RetryConfig()

        # Canonical and alias classes
        self.assertFalse(cfg.is_retryable(ProviderAuthenticationError("Invalid API key", provider_id="openai")))
        self.assertFalse(cfg.is_retryable(ProviderAuthError("Unauthorized", provider_id="openai")))
        self.assertFalse(cfg.is_retryable(ModelNotFoundError("Model xyz not found", provider_id="ollama")))
        self.assertFalse(cfg.is_retryable(ProviderNotFoundError("Missing model", provider_id="ollama")))
        self.assertFalse(cfg.is_retryable(ValueError("Invalid dimension")))

        # Error containing connection keyword but also 401/unauthorized must NOT be retryable
        tricky_err = ProviderError("401 Unauthorized: Connection closed due to invalid api key")
        self.assertFalse(cfg.is_retryable(tricky_err))

        # Transient errors MUST be retryable
        self.assertTrue(cfg.is_retryable(ProviderRateLimitError("Quota limit", provider_id="openai")))
        self.assertTrue(cfg.is_retryable(ProviderTimeoutError("Gateway timeout", provider_id="openai")))
        self.assertTrue(cfg.is_retryable(ProviderConnectionError("Connection reset", provider_id="openai")))

    def test_rate_limiter_multithreaded_concurrency(self):
        """Verify RateLimiter under multi-threaded load: lock released during sleep, RPM never exceeded."""
        rpm_limit = 10
        virtual_clock = 1000.0
        clock_lock = threading.Lock()

        def thread_safe_time():
            with clock_lock:
                return virtual_clock

        def thread_safe_sleep(sec):
            nonlocal virtual_clock
            with clock_lock:
                virtual_clock += sec

        limiter = RateLimiter(
            requests_per_minute=rpm_limit,
            time_fn=thread_safe_time,
            sleep_fn=thread_safe_sleep,
        )

        threads: list[threading.Thread] = []
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(4):
                    limiter.acquire()
            except Exception as e:
                errors.append(e)

        # Launch 5 concurrent threads (5 * 4 = 20 acquisitions across RPM 10)
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        # Lock was not held deadlocked, virtual clock advanced, and RPM constraint held
        self.assertLessEqual(limiter.current_load, rpm_limit)

    def test_rate_limiter_cancellation(self):
        """Verify RateLimiter can be cancelled via cancellation event."""
        cancel_evt = threading.Event()
        cancel_evt.set()  # Cancel immediately

        limiter = RateLimiter(requests_per_minute=10)
        with self.assertRaises(RateLimitCancelledError):
            limiter.acquire(cancellation_event=cancel_evt)


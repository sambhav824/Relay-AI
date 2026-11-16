"""Unit tests for Relay AI Embedding Providers (Milestone 10.4, Phase 1).

Tests embedding provider abstractions, OpenAI configurable dimensions,
Ollama dimension validation, mock providers, vector validation, and error mappings.
"""

from __future__ import annotations

import math
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.providers.embeddings import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    normalize_vector,
    validate_vector,
)
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class TestVectorValidationAndNormalization(TestCase):
    """Tests for vector validation and L2 normalization helpers."""

    def test_validate_vector_valid(self):
        vec = [0.1, 0.2, 0.3]
        res = validate_vector(vec, expected_dimension=3)
        self.assertEqual(res, [0.1, 0.2, 0.3])

    def test_validate_vector_empty_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_vector([], expected_dimension=3)
        self.assertIn("cannot be empty", str(ctx.exception).lower())

    def test_validate_vector_dimension_mismatch_rejected(self):
        # Must never silently truncate or pad
        with self.assertRaises(ValueError) as ctx:
            validate_vector([0.1, 0.2], expected_dimension=3)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            validate_vector([0.1, 0.2, 0.3, 0.4], expected_dimension=3)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_validate_vector_non_finite_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_vector([0.1, float("nan"), 0.3], expected_dimension=3)
        self.assertIn("non-finite", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            validate_vector([0.1, float("inf"), 0.3], expected_dimension=3)
        self.assertIn("non-finite", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            validate_vector([0.1, float("-inf"), 0.3], expected_dimension=3)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_normalize_vector_unit_length(self):
        raw = [3.0, 4.0]
        norm = normalize_vector(raw)
        self.assertAlmostEqual(norm[0], 0.6)
        self.assertAlmostEqual(norm[1], 0.8)
        mag = math.sqrt(sum(x * x for x in norm))
        self.assertAlmostEqual(mag, 1.0, places=6)

    def test_normalize_vector_zero_magnitude_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_vector([0.0, 0.0, 0.0])
        self.assertIn("zero-magnitude", str(ctx.exception).lower())


class TestMockEmbeddingProvider(TestCase):
    """Tests for MockEmbeddingProvider."""

    def test_mock_provider_properties(self):
        provider = MockEmbeddingProvider(
            model_id="test-model",
            model_version="v2.0",
            dimension=768,
        )
        self.assertEqual(provider.provider_id, "mock")
        self.assertEqual(provider.model_id, "test-model")
        self.assertEqual(provider.model_version, "v2.0")
        self.assertEqual(provider.dimension, 768)

    def test_mock_provider_deterministic_query(self):
        provider = MockEmbeddingProvider(dimension=768)
        v1 = provider.embed_query("Hello world")
        v2 = provider.embed_query("Hello world")
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 768)

        # Unit magnitude
        mag = math.sqrt(sum(x * x for x in v1))
        self.assertAlmostEqual(mag, 1.0, places=5)

    def test_mock_provider_different_inputs(self):
        provider = MockEmbeddingProvider(dimension=768)
        v1 = provider.embed_query("Alpha")
        v2 = provider.embed_query("Beta")
        self.assertNotEqual(v1, v2)

    def test_mock_provider_embed_documents(self):
        provider = MockEmbeddingProvider(dimension=1536)
        docs = ["Document 1", "Document 2", "Document 3"]
        vectors = provider.embed_documents(docs)
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            self.assertEqual(len(v), 1536)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)

    def test_mock_provider_empty_input_rejected(self):
        provider = MockEmbeddingProvider()
        with self.assertRaises(ValueError):
            provider.embed_query("")
        with self.assertRaises(ValueError):
            provider.embed_query("   ")
        with self.assertRaises(ValueError):
            provider.embed_documents(["Valid", "  "])

    def test_mock_provider_fail_injection(self):
        provider = MockEmbeddingProvider(fail_next=ProviderConnectionError("Injected offline", provider_id="mock"))
        with self.assertRaises(ProviderConnectionError):
            provider.embed_query("Test")
        # Next call succeeds
        v = provider.embed_query("Test")
        self.assertEqual(len(v), 768)

    def test_mock_provider_corrupt_dimension_rejection(self):
        provider = MockEmbeddingProvider(dimension=768, corrupt_next_dim=512)
        with self.assertRaises(ValueError) as ctx:
            provider.embed_query("Test")
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_mock_provider_non_finite_rejection(self):
        provider = MockEmbeddingProvider(dimension=768, non_finite_next=True)
        with self.assertRaises(ValueError) as ctx:
            provider.embed_query("Test")
        self.assertIn("non-finite", str(ctx.exception).lower())


class TestOpenAIEmbeddingProvider(TestCase):
    """Tests for OpenAIEmbeddingProvider."""

    def test_initialization_dimensions(self):
        p1536 = OpenAIEmbeddingProvider(api_key="sk-test", dimension=1536)
        self.assertEqual(p1536.dimension, 1536)
        self.assertEqual(p1536.provider_id, "openai")

        p768 = OpenAIEmbeddingProvider(api_key="sk-test", dimension=768)
        self.assertEqual(p768.dimension, 768)

        with self.assertRaises(ValueError):
            OpenAIEmbeddingProvider(api_key="sk-test", dimension=0)

        with self.assertRaises(ValueError):
            OpenAIEmbeddingProvider(api_key="sk-test", dimension=-10)

    def test_api_key_not_exposed(self):
        p = OpenAIEmbeddingProvider(api_key="secret-key-123")
        # Ensure repr and str don't print the secret key
        self.assertNotIn("secret-key-123", repr(p))
        self.assertNotIn("secret-key-123", str(p))

    def test_configurable_dimension_call(self):
        mock_client = MagicMock()
        mock_item_1 = MagicMock()
        mock_item_1.index = 0
        mock_item_1.embedding = [0.1] * 768

        mock_item_2 = MagicMock()
        mock_item_2.index = 1
        mock_item_2.embedding = [0.2] * 768

        mock_resp = MagicMock()
        mock_resp.data = [mock_item_1, mock_item_2]
        mock_client.embeddings.create.return_value = mock_resp

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model_id="text-embedding-3-small",
            dimension=768,
            client=mock_client,
        )

        vectors = provider.embed_documents(["First", "Second"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)

        # Check call arguments passed to client
        mock_client.embeddings.create.assert_called_once_with(
            input=["First", "Second"],
            model="text-embedding-3-small",
            dimensions=768,
        )

    def test_openai_dimension_mismatch_rejected(self):
        # If API returns wrong dimension, provider must reject rather than pad/truncate
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.index = 0
        mock_item.embedding = [0.1] * 512  # returns 512 when 768 expected

        mock_resp = MagicMock()
        mock_resp.data = [mock_item]
        mock_client.embeddings.create.return_value = mock_resp

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            dimension=768,
            client=mock_client,
        )
        with self.assertRaises(ValueError) as ctx:
            provider.embed_query("Test")
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_openai_error_mappings(self):
        mock_client = MagicMock()
        provider = OpenAIEmbeddingProvider(api_key="sk-test", client=mock_client)

        # 401 Auth error
        mock_client.embeddings.create.side_effect = Exception("401 Unauthorized: Invalid API key")
        with self.assertRaises(ProviderAuthenticationError):
            provider.embed_query("Test")

        # 429 Rate limit
        mock_client.embeddings.create.side_effect = Exception("RateLimitError 429 quota exceeded")
        with self.assertRaises(ProviderRateLimitError):
            provider.embed_query("Test")

        # Timeout
        mock_client.embeddings.create.side_effect = Exception("APITimeoutError connection timed out")
        with self.assertRaises(ProviderTimeoutError):
            provider.embed_query("Test")

        # 404 Model not found
        mock_client.embeddings.create.side_effect = Exception("NotFoundError: model does not exist")
        with self.assertRaises(ModelNotFoundError):
            provider.embed_query("Test")

        # General error
        mock_client.embeddings.create.side_effect = Exception("Generic internal error")
        with self.assertRaises(ProviderError):
            provider.embed_query("Test")


class TestOllamaEmbeddingProvider(TestCase):
    """Tests for OllamaEmbeddingProvider."""

    def test_initialization(self):
        p = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model_id="nomic-embed-text",
            dimension=768,
        )
        self.assertEqual(p.provider_id, "ollama")
        self.assertEqual(p.model_id, "nomic-embed-text")
        self.assertEqual(p.dimension, 768)

    def test_ollama_batch_embed_success(self):
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [[0.5] * 768, [0.3] * 768]
        }
        mock_http.post.return_value = mock_resp

        provider = OllamaEmbeddingProvider(
            dimension=768,
            http_client=mock_http,
        )
        vectors = provider.embed_documents(["Doc 1", "Doc 2"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)
        self.assertEqual(len(vectors[1]), 768)

    def test_ollama_dimension_validation_rejection(self):
        # Ollama returns 1024 or 512 dimensions when 768 expected: must reject!
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [[0.5] * 1024]
        }
        mock_http.post.return_value = mock_resp

        provider = OllamaEmbeddingProvider(
            dimension=768,
            http_client=mock_http,
        )
        with self.assertRaises(ValueError) as ctx:
            provider.embed_documents(["Test"])
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_ollama_error_handling(self):
        mock_http = MagicMock()
        provider = OllamaEmbeddingProvider(http_client=mock_http)

        # 404 Model not found (in single endpoint fallback)
        batch_fail_resp = MagicMock(status_code=400)
        single_404_resp = MagicMock(status_code=404)
        mock_http.post.side_effect = [batch_fail_resp, single_404_resp]

        with self.assertRaises(ModelNotFoundError):
            provider.embed_query("Test")

        # Connection error
        mock_http.post.side_effect = Exception("ConnectError: connection refused")
        with self.assertRaises(ProviderConnectionError):
            provider.embed_query("Test")

        # Timeout
        mock_http.post.side_effect = Exception("TimeoutException: read timeout")
        with self.assertRaises(ProviderTimeoutError):
            provider.embed_query("Test")

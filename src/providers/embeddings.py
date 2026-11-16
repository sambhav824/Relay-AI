"""Relay AI — Embedding Provider Abstraction Layer.

Defines the abstract base class and concrete implementations for generating
dense vector embeddings via OpenAI, Ollama, and deterministic mock providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Sequence

from src.core.security import CredentialStore
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger("RelayAI.Embeddings")


def validate_vector(vector: Sequence[float], expected_dimension: Optional[int] = None) -> List[float]:
    """Validate that vector is non-empty, matches expected dimension, and contains only finite floats.

    Never silently truncates or pads vectors.
    """
    if not vector:
        raise ValueError("Embedding vector cannot be empty.")

    vec_list = [float(x) for x in vector]

    if expected_dimension is not None and len(vec_list) != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, got {len(vec_list)}. "
            "Vectors must never be silently truncated or padded."
        )

    for i, val in enumerate(vec_list):
        if not math.isfinite(val):
            raise ValueError(f"Vector contains non-finite value ({val}) at index {i}.")

    return vec_list


def normalize_vector(vector: Sequence[float], eps: float = 1e-12) -> List[float]:
    """Compute L2 unit normalization on vector so that ||v||_2 == 1.0."""
    validated = validate_vector(vector)
    norm = math.sqrt(sum(x * x for x in validated))
    if norm < eps:
        raise ValueError("Cannot normalize a zero-magnitude vector.")
    return [x / norm for x in validated]


class EmbeddingProvider(ABC):
    """Abstract Base Class for all Relay AI vector embedding providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g. 'openai', 'ollama', 'mock')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Embedding model identifier (e.g. 'text-embedding-3-small', 'nomic-embed-text')."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model revision, version tag, or 'default'."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality (e.g. 1536, 768)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate unit-normalized embedding vector for a single query text."""
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate unit-normalized embedding vectors for a sequence of texts."""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embedding provider using OpenAI's API.

    Supports configurable dimensions for Matryoshka models (e.g. text-embedding-3-small).
    API key is retrieved securely from Windows Credential Manager or optional override.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "text-embedding-3-small",
        model_version: str = "default",
        dimension: int = 1536,
        credential_store: Optional[CredentialStore] = None,
        client: Optional[Any] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}.")
        if dimension not in (768, 1536) and not model_id.startswith("mock"):
            logger.warning(
                "Non-standard dimension %d specified for OpenAI model %s. "
                "Ensure target database column supports this dimension.",
                dimension,
                model_id,
            )

        self._model_id = model_id
        self._model_version = model_version
        self._dimension = dimension
        self._timeout = timeout
        self._base_url = base_url
        self._manual_api_key = api_key
        self.credential_store = credential_store or CredentialStore()
        self._injected_client = client

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_api_key(self) -> Optional[str]:
        if self._manual_api_key:
            return self._manual_api_key.strip()
        key = self.credential_store.get_api_key(self.provider_id)
        return key.strip() if key else None

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client

        api_key = self._get_api_key()
        if not api_key:
            raise ProviderAuthenticationError(
                "OpenAI API key not found. Please configure your key in Settings or Credential Manager.",
                provider_id=self.provider_id,
            )

        try:
            import openai
            kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return openai.OpenAI(**kwargs)
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize OpenAI client: {exc}",
                provider_id=self.provider_id,
            ) from exc

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text else ""
        if not cleaned:
            raise ValueError("Input text cannot be empty or purely whitespace.")
        results = self.embed_documents([cleaned])
        return results[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts: List[str] = []
        for i, t in enumerate(texts):
            c = t.strip() if t else ""
            if not c:
                raise ValueError(f"Document text at index {i} cannot be empty or purely whitespace.")
            cleaned_texts.append(c)

        client = self._get_client()

        call_kwargs: Dict[str, Any] = {
            "input": cleaned_texts,
            "model": self._model_id,
        }
        # text-embedding-3 models support native Matryoshka dimension reduction
        if "text-embedding-3" in self._model_id:
            call_kwargs["dimensions"] = self._dimension

        try:
            response = client.embeddings.create(**call_kwargs)
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"

            if "Authentication" in err_text or "401" in err_text:
                raise ProviderAuthenticationError("Invalid or missing OpenAI API key.", provider_id=self.provider_id) from exc
            if "RateLimit" in err_text or "429" in err_text:
                raise ProviderRateLimitError("OpenAI rate limit or quota exceeded.", provider_id=self.provider_id) from exc
            if "Timeout" in err_text or "timed out" in err_text.lower():
                raise ProviderTimeoutError(f"OpenAI embedding request timed out: {exc}", provider_id=self.provider_id) from exc
            if "NotFound" in err_text or "404" in err_text:
                raise ModelNotFoundError(f"OpenAI embedding model '{self._model_id}' not found.", provider_id=self.provider_id) from exc
            if "Connect" in err_text or "Network" in err_text:
                raise ProviderConnectionError(f"OpenAI connection error: {exc}", provider_id=self.provider_id) from exc
            raise ProviderError(f"OpenAI embedding generation failed: {exc}", provider_id=self.provider_id) from exc

        # Ensure ordered matching by response index
        items = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        vectors: List[List[float]] = []

        for item in items:
            raw_vec = item.embedding
            validated = validate_vector(raw_vec, expected_dimension=self._dimension)
            normalized = normalize_vector(validated)
            vectors.append(normalized)

        return vectors


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using Ollama's HTTP API.

    Validates actual output dimensionality against expected dimension.
    Never silently pads or truncates.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "nomic-embed-text",
        model_version: str = "default",
        dimension: int = 768,
        http_client: Optional[Any] = None,
        timeout: float = 30.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}.")

        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._model_version = model_version
        self._dimension = dimension
        self._timeout = timeout
        self._injected_client = http_client

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        import httpx
        return httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text else ""
        if not cleaned:
            raise ValueError("Input text cannot be empty or purely whitespace.")
        results = self.embed_documents([cleaned])
        return results[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts: List[str] = []
        for i, t in enumerate(texts):
            c = t.strip() if t else ""
            if not c:
                raise ValueError(f"Document text at index {i} cannot be empty or purely whitespace.")
            cleaned_texts.append(c)

        client = self._get_client()
        vectors: List[List[float]] = []

        try:
            # Try batch /api/embed endpoint first (Ollama 0.1.34+)
            response = client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model_id, "input": cleaned_texts},
            )

            if response.status_code == 200:
                data = response.json()
                raw_embeddings = data.get("embeddings", [])
                if len(raw_embeddings) != len(cleaned_texts):
                    raise ProviderError(
                        f"Ollama returned {len(raw_embeddings)} embeddings for {len(cleaned_texts)} inputs.",
                        provider_id=self.provider_id,
                    )
                for raw_vec in raw_embeddings:
                    validated = validate_vector(raw_vec, expected_dimension=self._dimension)
                    vectors.append(normalize_vector(validated))
                return vectors

            # Fallback to single-prompt /api/embeddings endpoint
            for text in cleaned_texts:
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model_id, "prompt": text},
                )
                if resp.status_code == 404:
                    raise ModelNotFoundError(
                        f"Ollama model '{self._model_id}' not found. Please run 'ollama pull {self._model_id}'.",
                        provider_id=self.provider_id,
                    )
                if resp.status_code != 200:
                    raise ProviderError(
                        f"Ollama embedding request failed with HTTP {resp.status_code}: {resp.text}",
                        provider_id=self.provider_id,
                    )

                data = resp.json()
                raw_vec = data.get("embedding", [])
                validated = validate_vector(raw_vec, expected_dimension=self._dimension)
                vectors.append(normalize_vector(validated))

            return vectors

        except (ProviderError, ValueError):
            raise
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            if "Timeout" in err_text or "timed out" in err_text.lower():
                raise ProviderTimeoutError(f"Ollama request timed out: {exc}", provider_id=self.provider_id) from exc
            if "Connect" in err_text or "Network" in err_text or "connection refused" in err_text.lower():
                raise ProviderConnectionError(
                    f"Failed to connect to Ollama at {self._base_url}. Ensure the Ollama daemon is running.",
                    provider_id=self.provider_id,
                ) from exc
            raise ProviderError(f"Ollama embedding generation failed: {exc}", provider_id=self.provider_id) from exc


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, self-contained embedding provider for testing.

    Generates reproducible unit-normalized vectors without any external network calls.
    Supports deliberate failure injection for resilience testing.
    """

    def __init__(
        self,
        model_id: str = "mock-embed-v1",
        model_version: str = "v1.0",
        dimension: int = 768,
        provider_id: str = "mock",
        fail_next: Optional[Exception] = None,
        corrupt_next_dim: Optional[int] = None,
        non_finite_next: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}.")

        self._provider_id = provider_id
        self._model_id = model_id
        self._model_version = model_version
        self._dimension = dimension
        self.fail_next = fail_next
        self.corrupt_next_dim = corrupt_next_dim
        self.non_finite_next = non_finite_next

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str, target_dim: int) -> List[float]:
        """Generate a deterministic unit-normalized pseudo-vector seeded by text."""
        # Mix text, model_id, and model_version into digest
        seed_content = f"{self._model_id}:{self._model_version}:{text}"
        digest = hashlib.sha256(seed_content.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big")

        raw: List[float] = []
        for i in range(target_dim):
            # Deterministic linear congruential pseudo-random calculation
            seed_int = (seed_int * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            val = ((seed_int >> 32) / 0xFFFFFFFF) - 0.5
            raw.append(val)

        if self.non_finite_next:
            self.non_finite_next = False
            raw[0] = float("nan")

        return raw

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text else ""
        if not cleaned:
            raise ValueError("Input text cannot be empty or purely whitespace.")
        results = self.embed_documents([cleaned])
        return results[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self.fail_next:
            exc = self.fail_next
            self.fail_next = None
            raise exc

        if not texts:
            return []

        vectors: List[List[float]] = []
        target_dim = self.corrupt_next_dim if self.corrupt_next_dim is not None else self._dimension
        self.corrupt_next_dim = None

        for i, text in enumerate(texts):
            cleaned = text.strip() if text else ""
            if not cleaned:
                raise ValueError(f"Document text at index {i} cannot be empty or purely whitespace.")

            raw = self._generate_vector(cleaned, target_dim)
            validated = validate_vector(raw, expected_dimension=self._dimension)
            normalized = normalize_vector(validated)
            vectors.append(normalized)

        return vectors

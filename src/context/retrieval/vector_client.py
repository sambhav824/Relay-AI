"""Relay AI Universal Context Engine — Vector Similarity Client.

Interacts with Supabase PostgREST RPC `match_document_chunks` using authenticated
user credentials, strict dimension validation, and safe error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import httpx

from src.context.auth.manager import AuthManager
from src.providers.embeddings import validate_vector

logger = logging.getLogger("RelayAI.VectorClient")


class VectorClientError(Exception):
    """Base exception for vector retrieval client operations."""
    pass


class VectorClientAuthError(VectorClientError):
    """Raised when authentication is missing, invalid, or expired."""
    pass


class VectorClientValidationError(VectorClientError):
    """Raised when query vector or search parameters fail validation."""
    pass


class VectorClientRPCError(VectorClientError):
    """Raised when Supabase PostgREST RPC returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class VectorMatch:
    """Represents a matching document chunk returned by similarity search."""

    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    similarity: float


class VectorSimilarityClient:
    """Client for executing vector similarity search against Supabase pgvector."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        auth_manager: Optional[AuthManager] = None,
        access_token: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ) -> None:
        url = supabase_url or os.environ.get("SUPABASE_URL", "")
        key = supabase_anon_key or os.environ.get("SUPABASE_ANON_KEY", "")

        self.supabase_url = url.rstrip("/")
        self.anon_key = key
        self.auth_manager = auth_manager
        self.access_token = access_token
        self.timeout = timeout
        self._external_client = http_client is not None
        self._client = http_client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        """Close underlying HTTP client if internally owned."""
        if not self._external_client:
            self._client.close()

    def __enter__(self) -> VectorSimilarityClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get_active_token(self) -> str:
        """Retrieve current access token from AuthManager or static override."""
        if self.auth_manager is not None:
            token = self.auth_manager.get_access_token()
            if token:
                return token

        if self.access_token:
            return self.access_token

        raise VectorClientAuthError("Authentication required: no active user session or access token available.")

    def search_similar_chunks(
        self,
        query_vector: Sequence[float],
        model_id: str,
        dimension: int,
        match_threshold: float = 0.3,
        match_count: int = 10,
        filter_project_id: Optional[str] = None,
        filter_document_id: Optional[str] = None,
    ) -> List[VectorMatch]:
        """Query Supabase pgvector for document chunks similar to query_vector.

        Args:
            query_vector: Float sequence of dimensions 768 or 1536.
            model_id: Identifier of model used to generate query vector.
            dimension: Dimensionality (768 or 1536).
            match_threshold: Minimum cosine similarity (-1.0 to 1.0).
            match_count: Maximum number of matches to return (1 to 100).
            filter_project_id: Optional project UUID filter.
            filter_document_id: Optional document UUID filter.

        Returns:
            List of VectorMatch instances ordered by similarity descending.
        """
        # 1. Parameter Validation
        if not model_id or not model_id.strip():
            raise VectorClientValidationError("model_id must not be empty.")

        if dimension not in (768, 1536):
            raise VectorClientValidationError(f"Invalid dimension {dimension}. Supported dimensions are 768 and 1536.")

        if model_id == "ollama/nomic-embed-text" and dimension == 1536:
            raise VectorClientValidationError(
                f"Model '{model_id}' does not support 1536 dimensions. Nomic natively outputs 768 dimensions."
            )

        if match_threshold < -1.0 or match_threshold > 1.0:
            raise VectorClientValidationError(
                f"match_threshold must be between -1.0 and 1.0, got {match_threshold}."
            )

        if match_count < 1 or match_count > 100:
            raise VectorClientValidationError(
                f"match_count must be between 1 and 100, got {match_count}."
            )

        # 2. Vector Validation (checks length, empty, and non-finite values)
        try:
            validated_vec = validate_vector(query_vector, expected_dimension=dimension)
        except ValueError as exc:
            raise VectorClientValidationError(f"Invalid query vector: {exc}") from exc

        # 3. Authentication
        token = self._get_active_token()

        if not self.supabase_url:
            raise VectorClientValidationError("Supabase URL is not configured.")

        # 4. Construct RPC payload
        endpoint = f"{self.supabase_url}/rest/v1/rpc/match_document_chunks"
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "p_model_id": model_id.strip(),
            "p_dimension": dimension,
            "query_embedding_1536": validated_vec if dimension == 1536 else None,
            "query_embedding_768": validated_vec if dimension == 768 else None,
            "match_threshold": float(match_threshold),
            "match_count": int(match_count),
            "filter_project_id": str(filter_project_id) if filter_project_id else None,
            "filter_document_id": str(filter_document_id) if filter_document_id else None,
        }

        # 5. Dispatch Request
        try:
            response = self._client.post(endpoint, json=payload, headers=headers)
        except httpx.RequestError as req_err:
            logger.error("Vector RPC network error: %s", req_err)
            raise VectorClientRPCError(f"Network error during vector similarity search: {req_err}") from req_err

        # 6. Response Evaluation
        if response.status_code == 401 or response.status_code == 403:
            raise VectorClientAuthError(
                f"Vector RPC authorization failed (HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.is_error:
            error_msg = response.text[:500]
            code = None
            try:
                err_json = response.json()
                error_msg = err_json.get("message") or err_json.get("hint") or error_msg
                code = err_json.get("code")
            except Exception:
                pass

            # Map PostgreSQL authorization SQLSTATE codes to VectorClientAuthError
            if code in ("28000", "42501"):
                raise VectorClientAuthError(
                    f"Vector RPC authorization failed (code {code}): {error_msg}"
                )

            raise VectorClientRPCError(
                f"Vector RPC error (HTTP {response.status_code}): {error_msg}",
                status_code=response.status_code,
                code=code,
            )

        try:
            results_data = response.json()
        except Exception as json_err:
            raise VectorClientRPCError(f"Failed to parse vector RPC JSON response: {json_err}") from json_err

        if not isinstance(results_data, list):
            raise VectorClientRPCError(f"Expected list of matches from RPC, got: {type(results_data)}")

        # 7. Convert to structured VectorMatch list
        matches: List[VectorMatch] = []
        for row in results_data:
            matches.append(
                VectorMatch(
                    id=row.get("id", ""),
                    document_id=row.get("document_id", ""),
                    project_id=row.get("project_id"),
                    chunk_index=int(row.get("chunk_index", 0)),
                    content=row.get("content", ""),
                    metadata=row.get("metadata", {}) or {},
                    similarity=float(row.get("similarity", 0.0)),
                )
            )

        return matches

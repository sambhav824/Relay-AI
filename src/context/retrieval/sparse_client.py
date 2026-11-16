"""Relay AI Universal Context Engine — Sparse Keyword Retrieval Client.

Executes full-text keyword retrieval using either Supabase PostgreSQL Full-Text Search
(via RPC `search_document_chunks_fts`) or local SQLite FTS5 (`chunks_fts`) for offline
and fallback scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

import httpx

from src.context.auth.manager import AuthManager

logger = logging.getLogger("RelayAI.SparseClient")


class SparseClientError(Exception):
    """Base exception for sparse full-text retrieval errors."""
    pass


class SparseClientAuthError(SparseClientError):
    """Raised when authentication is missing, invalid, or expired."""
    pass


class SparseClientValidationError(SparseClientError):
    """Raised when search parameters or queries fail validation."""
    pass


class SparseClientRPCError(SparseClientError):
    """Raised when Supabase PostgREST RPC returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class SparseMatch:
    """Represents a matching document chunk returned by full-text search."""

    id: str
    document_id: str
    project_id: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    score: float


class SparseRetrievalClient:
    """Client for executing sparse full-text search across Supabase and local SQLite."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        auth_manager: Optional[AuthManager] = None,
        access_token: Optional[str] = None,
        sqlite_connection: Optional[sqlite3.Connection] = None,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 10.0,
    ) -> None:
        url = supabase_url or os.environ.get("SUPABASE_URL", "")
        key = supabase_anon_key or os.environ.get("SUPABASE_ANON_KEY", "")

        self.supabase_url = url.rstrip("/")
        self.anon_key = key
        self.auth_manager = auth_manager
        self.access_token = access_token
        self.sqlite_conn = sqlite_connection
        self.timeout = timeout
        self._external_client = http_client is not None
        self._client = http_client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        """Close underlying HTTP client if internally owned."""
        if not self._external_client:
            self._client.close()

    def __enter__(self) -> SparseRetrievalClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get_active_user_id(self) -> Optional[str]:
        """Retrieve current authenticated user UUID from AuthManager if active."""
        if self.auth_manager is not None and self.auth_manager.user:
            return self.auth_manager.user.id
        return None

    def _get_active_token(self) -> Optional[str]:
        """Retrieve current access token from AuthManager or static override."""
        if self.auth_manager is not None:
            token = self.auth_manager.get_access_token()
            if token:
                return token
        if self.access_token:
            return self.access_token
        return None

    @staticmethod
    def _sanitize_fts5_query(raw_query: str) -> str:
        """Sanitize raw user query into safe tokenized phrase terms for SQLite FTS5."""
        tokens = re.findall(r"\b\w+\b", raw_query, flags=re.UNICODE)
        if not tokens:
            return ""
        # Wrap each alphanumeric token in double quotes to prevent FTS5 operator collisions
        return " ".join(f'"{t}"' for t in tokens)

    def search_keywords(
        self,
        query: str,
        match_count: int = 20,
        filter_project_id: Optional[str] = None,
        filter_document_id: Optional[str] = None,
        prefer_offline: bool = False,
    ) -> List[SparseMatch]:
        """Execute full-text keyword retrieval.

        Args:
            query: User search text query.
            match_count: Maximum number of matches to return (1 to 100).
            filter_project_id: Optional project UUID filter.
            filter_document_id: Optional document UUID filter.
            prefer_offline: If True, bypass cloud and query local SQLite FTS5 directly.

        Returns:
            List of SparseMatch ordered by score descending.

        Raises:
            SparseClientValidationError: If inputs are invalid.
            SparseClientAuthError: If cloud authentication explicitly fails (never swallowed).
            SparseClientRPCError: If cloud search fails and no local database is available.
        """
        # 1. Parameter Validation
        if query is None or not str(query).strip():
            return []

        cleaned_query = str(query).strip()

        if match_count < 1 or match_count > 100:
            raise SparseClientValidationError(f"match_count must be between 1 and 100, got {match_count}.")

        # 2. Local-only mode requested
        if prefer_offline:
            if self.sqlite_conn is None:
                raise SparseClientValidationError("prefer_offline requested but no SQLite connection was provided.")
            return self._search_sqlite_fts(
                query=cleaned_query,
                match_count=match_count,
                filter_project_id=filter_project_id,
                filter_document_id=filter_document_id,
            )

        # 3. Cloud Mode Attempt (with network fallback to SQLite)
        can_attempt_cloud = bool(self.supabase_url and not self.supabase_url.startswith("placeholder"))
        if can_attempt_cloud:
            try:
                return self._search_supabase_fts(
                    query=cleaned_query,
                    match_count=match_count,
                    filter_project_id=filter_project_id,
                    filter_document_id=filter_document_id,
                )
            except (SparseClientAuthError, SparseClientValidationError):
                # Security boundary: NEVER swallow authentication or validation errors
                raise
            except (SparseClientRPCError, httpx.RequestError) as cloud_err:
                # Transient network or RPC failure
                if self.sqlite_conn is not None:
                    logger.warning("Cloud FTS unavailable (%s). Falling back to local SQLite FTS5.", cloud_err)
                    return self._search_sqlite_fts(
                        query=cleaned_query,
                        match_count=match_count,
                        filter_project_id=filter_project_id,
                        filter_document_id=filter_document_id,
                    )
                raise

        # 4. Fallback to Local SQLite FTS5 when cloud URL is not configured
        if self.sqlite_conn is not None:
            return self._search_sqlite_fts(
                query=cleaned_query,
                match_count=match_count,
                filter_project_id=filter_project_id,
                filter_document_id=filter_document_id,
            )

        # Neither cloud configured nor local SQLite available
        raise SparseClientValidationError(
            "Sparse search unavailable: Supabase URL is not configured and no local SQLite connection provided."
        )

    def _search_supabase_fts(
        self,
        query: str,
        match_count: int,
        filter_project_id: Optional[str],
        filter_document_id: Optional[str],
    ) -> List[SparseMatch]:
        """Dispatch query to Supabase PostgREST RPC `search_document_chunks_fts`."""
        token = self._get_active_token()
        if not token:
            raise SparseClientAuthError("Authentication required: no active user session or access token available.")

        endpoint = f"{self.supabase_url}/rest/v1/rpc/search_document_chunks_fts"
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "p_query": query,
            "match_count": int(match_count),
            "filter_project_id": str(filter_project_id) if filter_project_id else None,
            "filter_document_id": str(filter_document_id) if filter_document_id else None,
        }

        try:
            response = self._client.post(endpoint, json=payload, headers=headers)
        except httpx.RequestError as req_err:
            logger.error("Supabase FTS RPC network error: %s", req_err)
            raise SparseClientRPCError(f"Network error during full-text search: {req_err}") from req_err

        if response.status_code in (401, 403):
            raise SparseClientAuthError(
                f"Full-text search authorization failed (HTTP {response.status_code}): {response.text[:300]}"
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

            # Map PostgreSQL authorization SQLSTATE codes to SparseClientAuthError
            if code in ("28000", "42501") or response.status_code in (401, 403):
                raise SparseClientAuthError(
                    f"Full-text search authorization failed (HTTP {response.status_code}, code {code}): {error_msg}"
                )

            raise SparseClientRPCError(
                f"Full-text search RPC error (HTTP {response.status_code}): {error_msg}",
                status_code=response.status_code,
                code=code,
            )

        try:
            results_data = response.json()
        except Exception as json_err:
            raise SparseClientRPCError(f"Failed to parse FTS RPC response JSON: {json_err}") from json_err

        if not isinstance(results_data, list):
            raise SparseClientRPCError(f"Expected list of matches from FTS RPC, got: {type(results_data)}")

        matches: List[SparseMatch] = []
        for row in results_data:
            matches.append(
                SparseMatch(
                    id=str(row.get("id", "")),
                    document_id=str(row.get("document_id", "")),
                    project_id=str(row["project_id"]) if row.get("project_id") else None,
                    chunk_index=int(row.get("chunk_index", 0)),
                    content=str(row.get("content", "")),
                    metadata=dict(row.get("metadata") or {}),
                    score=float(row.get("score", 0.0)),
                )
            )

        return matches

    def _search_sqlite_fts(
        self,
        query: str,
        match_count: int,
        filter_project_id: Optional[str],
        filter_document_id: Optional[str],
    ) -> List[SparseMatch]:
        """Execute BM25 search on local SQLite FTS5 table `chunks_fts`."""
        if self.sqlite_conn is None:
            return []

        fts_query = self._sanitize_fts5_query(query)
        if not fts_query:
            return []

        proj_id_clean = filter_project_id.strip() if filter_project_id and filter_project_id.strip() else None
        doc_id_clean = filter_document_id.strip() if filter_document_id and filter_document_id.strip() else None
        active_user_id = self._get_active_user_id()

        sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                c.project_id,
                COALESCE(dc.chunk_index, 0) AS chunk_index,
                c.content,
                COALESCE(dc.metadata_json, '{}') AS metadata_json,
                c.rank AS bm25_rank
            FROM chunks_fts c
            LEFT JOIN document_chunks dc ON dc.id = c.chunk_id
            WHERE chunks_fts MATCH ?
              AND (? IS NULL OR c.project_id = ?)
              AND (? IS NULL OR c.document_id = ?)
            ORDER BY c.rank ASC
            LIMIT ?
        """
        # Oversample by 2x if active user filter is present to compensate for post-filtering
        oversample_limit = match_count * 2 if active_user_id else match_count
        params = (
            fts_query,
            proj_id_clean,
            proj_id_clean,
            doc_id_clean,
            doc_id_clean,
            oversample_limit,
        )

        try:
            cursor = self.sqlite_conn.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as op_err:
            logger.warning("Local SQLite FTS5 query execution failed: %s", op_err)
            return []

        matches: List[SparseMatch] = []
        for row in rows:
            cid, doc_id, proj_id, chunk_idx, content, meta_raw, bm25_rank = row
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {}

            # Multi-profile isolation: if user is authenticated and chunk has owner_user_id,
            # enforce that chunk belongs to caller
            chunk_owner = meta.get("owner_user_id")
            if active_user_id and chunk_owner and chunk_owner != active_user_id:
                continue

            # In SQLite FTS5, rank is more negative for better matches.
            # Invert rank (-bm25_rank) so higher numerical score indicates better relevance.
            normalized_score = -float(bm25_rank) if bm25_rank is not None else 0.0

            matches.append(
                SparseMatch(
                    id=str(cid),
                    document_id=str(doc_id) if doc_id else "",
                    project_id=str(proj_id) if proj_id else None,
                    chunk_index=int(chunk_idx),
                    content=str(content or ""),
                    metadata=meta,
                    score=normalized_score,
                )
            )
            if len(matches) >= match_count:
                break

        return matches

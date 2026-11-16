"""Relay AI Universal Context Engine — SQLite Vector Embedding Cache.

Implements high-performance, collision-free local caching for vector embeddings
using IEEE-754 32-bit little-endian binary BLOB serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import math
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("RelayAI.EmbeddingCache")

EMBEDDING_CACHE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    preprocessing_version TEXT NOT NULL,
    normalize INTEGER NOT NULL DEFAULT 1,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_emb_cache_lookup 
ON embedding_cache(provider_id, model_id, model_version, dimension, content_hash);
"""


@dataclass(frozen=True)
class EmbeddingCacheRecord:
    """Full representation of a cached vector embedding."""
    cache_key: str
    content_hash: str
    provider_id: str
    model_id: str
    model_version: str
    dimension: int
    preprocessing_version: str
    normalize: bool
    embedding: List[float]
    created_at: Optional[str] = None


def compute_cache_key(
    content_hash: str,
    provider_id: str,
    model_id: str,
    model_version: str,
    dimension: int,
    preprocessing_version: str = "v1",
    normalize: bool = True,
) -> str:
    """Compute a collision-free deterministic SHA-256 cache key.

    Prevents collisions between:
    - Same content with different dimensions (e.g. 1536 vs 768).
    - Same content across different models or providers.
    - Updated model versions or revisions.
    - Different preprocessing or normalization configurations.
    """
    key_str = (
        f"{provider_id.strip()}:{model_id.strip()}:{model_version.strip()}:"
        f"{dimension}:{preprocessing_version.strip()}:{int(normalize)}:{content_hash.strip()}"
    )
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def serialize_embedding(vector: Sequence[float], dimension: int) -> bytes:
    """Serialize a float vector into a compact IEEE-754 little-endian binary float32 BLOB.

    Ensures exact dimension match and finite float values.
    """
    if len(vector) != dimension:
        raise ValueError(
            f"Embedding vector dimension mismatch: expected {dimension}, got {len(vector)}. "
            "Vectors must never be silently truncated or padded."
        )

    for i, x in enumerate(vector):
        if not math.isfinite(x):
            raise ValueError(f"Cannot serialize vector containing non-finite value ({x}) at index {i}.")

    return struct.pack(f"<{dimension}f", *vector)


def deserialize_embedding(blob: bytes, expected_dimension: int) -> List[float]:
    """Deserialize an IEEE-754 little-endian float32 BLOB back into a list of floats.

    Validates byte length and rejects corrupted or non-finite records.
    """
    expected_bytes = expected_dimension * 4
    if len(blob) != expected_bytes:
        raise ValueError(
            f"Corrupted cache BLOB: expected {expected_bytes} bytes for dimension {expected_dimension}, "
            f"got {len(blob)} bytes."
        )

    floats = list(struct.unpack(f"<{expected_dimension}f", blob))
    for i, x in enumerate(floats):
        if not math.isfinite(x):
            raise ValueError(f"Corrupted cache BLOB contains non-finite value ({x}) at index {i}.")

    return floats


class EmbeddingCache:
    """SQLite-backed persistent vector embedding cache.

    Provides atomic, transactional access and ensures corrupted entries
    are detected and purged automatically without corrupting search.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection
        self.ensure_table()

    def ensure_table(self) -> None:
        """Create the embedding_cache table and index if they do not exist."""
        with self.conn:
            self.conn.executescript(EMBEDDING_CACHE_TABLE_DDL)

    def get(
        self,
        content_hash: str,
        provider_id: str,
        model_id: str,
        model_version: str,
        dimension: int,
        preprocessing_version: str = "v1",
        normalize: bool = True,
    ) -> Optional[List[float]]:
        """Retrieve a cached vector embedding matching the exact configuration.

        Returns:
            Unit-normalized float list if cache hit; None if cache miss or corrupted.
        """
        cache_key = compute_cache_key(
            content_hash=content_hash,
            provider_id=provider_id,
            model_id=model_id,
            model_version=model_version,
            dimension=dimension,
            preprocessing_version=preprocessing_version,
            normalize=normalize,
        )

        cursor = self.conn.execute(
            "SELECT embedding, dimension FROM embedding_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        blob, stored_dim = row[0], row[1]
        if stored_dim != dimension:
            logger.warning(
                "Cache entry %s has stored dimension %d but requested %d. Evicting.",
                cache_key,
                stored_dim,
                dimension,
            )
            self.delete(cache_key)
            return None

        try:
            return deserialize_embedding(blob, expected_dimension=dimension)
        except ValueError as exc:
            logger.warning("Purging corrupted embedding cache entry %s: %s", cache_key, exc)
            self.delete(cache_key)
            return None

    def get_record(
        self,
        content_hash: str,
        provider_id: str,
        model_id: str,
        model_version: str,
        dimension: int,
        preprocessing_version: str = "v1",
        normalize: bool = True,
    ) -> Optional[EmbeddingCacheRecord]:
        """Retrieve the full record including metadata."""
        cache_key = compute_cache_key(
            content_hash=content_hash,
            provider_id=provider_id,
            model_id=model_id,
            model_version=model_version,
            dimension=dimension,
            preprocessing_version=preprocessing_version,
            normalize=normalize,
        )

        cursor = self.conn.execute(
            """
            SELECT cache_key, content_hash, provider_id, model_id, model_version,
                   dimension, preprocessing_version, normalize, embedding, created_at
            FROM embedding_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        try:
            vector = deserialize_embedding(row[8], expected_dimension=row[5])
            return EmbeddingCacheRecord(
                cache_key=row[0],
                content_hash=row[1],
                provider_id=row[2],
                model_id=row[3],
                model_version=row[4],
                dimension=row[5],
                preprocessing_version=row[6],
                normalize=bool(row[7]),
                embedding=vector,
                created_at=row[9],
            )
        except ValueError as exc:
            logger.warning("Corrupted record %s: %s. Evicting.", cache_key, exc)
            self.delete(cache_key)
            return None

    def set(
        self,
        content_hash: str,
        provider_id: str,
        model_id: str,
        model_version: str,
        dimension: int,
        embedding: Sequence[float],
        preprocessing_version: str = "v1",
        normalize: bool = True,
    ) -> bool:
        """Store an embedding in the cache.

        Avoids duplicate writes using INSERT OR IGNORE.
        Returns:
            True if newly inserted; False if entry already exists.
        """
        blob = serialize_embedding(embedding, dimension=dimension)
        cache_key = compute_cache_key(
            content_hash=content_hash,
            provider_id=provider_id,
            model_id=model_id,
            model_version=model_version,
            dimension=dimension,
            preprocessing_version=preprocessing_version,
            normalize=normalize,
        )

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO embedding_cache
                (cache_key, content_hash, provider_id, model_id, model_version,
                 dimension, preprocessing_version, normalize, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    content_hash,
                    provider_id,
                    model_id,
                    model_version,
                    dimension,
                    preprocessing_version,
                    int(normalize),
                    blob,
                ),
            )
            return cursor.rowcount > 0

    def delete(self, cache_key: str) -> bool:
        """Delete a cache entry by cache_key."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            )
            return cursor.rowcount > 0

    def count(self, provider_id: Optional[str] = None, model_id: Optional[str] = None) -> int:
        """Return total count of cached embeddings, optionally filtered by provider and model."""
        query = "SELECT COUNT(*) FROM embedding_cache"
        params: List[Any] = []
        conditions: List[str] = []

        if provider_id:
            conditions.append("provider_id = ?")
            params.append(provider_id)
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self.conn.execute(query, tuple(params))
        return cursor.fetchone()[0]

    def clear(self) -> int:
        """Purge all entries from the cache."""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM embedding_cache")
            return cursor.rowcount

"""Relay AI Universal Context Engine — Embedding Recovery Manager.

Detects and recovers document chunks that lack embeddings or failed during
prior ingestion/interrupted jobs, with retry exhaustion guards and cache reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from src.context.embeddings.pipeline import (
    ChunkItem,
    EmbeddedChunk,
    EmbeddingPipeline,
    EmbeddingResult,
    FailedChunk,
)
from src.providers.embeddings import EmbeddingProvider

logger = logging.getLogger("RelayAI.EmbeddingRecovery")


def update_chunk_embedding_metadata(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    successful: Optional[Sequence[EmbeddedChunk]] = None,
    failed: Optional[Sequence[FailedChunk]] = None,
) -> None:
    """Persist embedding outcomes to SQLite document_chunks metadata_json.

    Sets 'embedding_status' to 'COMPLETED' on success or 'FAILED' on failure,
    updating provider metadata and incrementing retry counts.
    """
    if not successful and not failed:
        return

    with conn:
        # Update successful chunks
        if successful:
            for sc in successful:
                row = conn.execute(
                    "SELECT metadata_json FROM document_chunks WHERE id = ?",
                    (sc.chunk_id,),
                ).fetchone()
                if row:
                    meta = json.loads(row[0] or "{}")
                    meta["embedding_status"] = "COMPLETED"
                    meta["embedding_model"] = provider.model_id
                    meta["embedding_dimension"] = provider.dimension
                    meta.pop("embedding_error", None)
                    conn.execute(
                        "UPDATE document_chunks SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta, separators=(",", ":")), sc.chunk_id),
                    )

        # Update failed chunks
        if failed:
            for fc in failed:
                row = conn.execute(
                    "SELECT metadata_json FROM document_chunks WHERE id = ?",
                    (fc.chunk_id,),
                ).fetchone()
                if row:
                    meta = json.loads(row[0] or "{}")
                    meta["embedding_status"] = "FAILED"
                    meta["embedding_error"] = fc.error
                    meta["embedding_retry_count"] = meta.get("embedding_retry_count", 0) + 1
                    conn.execute(
                        "UPDATE document_chunks SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta, separators=(",", ":")), fc.chunk_id),
                    )


class EmbeddingRecoveryManager:
    """Manages recovery of unembedded or failed document chunks."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        pipeline: EmbeddingPipeline,
        provider: EmbeddingProvider,
        max_retries: int = 3,
    ) -> None:
        self.conn = connection
        self.pipeline = pipeline
        self.provider = provider
        self.max_retries = max_retries

    def get_unembedded_chunks(
        self,
        document_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> List[ChunkItem]:
        """Find chunks in SQLite that lack valid embeddings in the cache or failed previously.

        If a chunk was already completed for this provider/model/dimension and not forced,
        it is skipped to avoid redundant processing.
        If a chunk exists in the cache, its status is updated directly to COMPLETED in SQLite.
        """
        query = (
            "SELECT id, document_id, project_id, chunk_index, content, "
            "page_number, metadata_json FROM document_chunks"
        )
        conditions: List[str] = []
        params: List[Any] = []

        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY document_id, chunk_index ASC"

        cursor = self.conn.execute(query, tuple(params))
        rows = cursor.fetchall()

        unembedded: List[ChunkItem] = []
        chunks_to_mark_completed: List[tuple[str, Dict[str, Any]]] = []

        for row in rows:
            cid, doc_id, proj_id, c_idx, content, page_num, meta_str = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
            )
            meta: Dict[str, Any] = json.loads(meta_str or "{}")

            status = str(meta.get("embedding_status", "")).upper()
            model = meta.get("embedding_model")
            dimension = meta.get("embedding_dimension")

            # Completed chunks are never unnecessarily reprocessed unless force=True
            if not force and status in ("COMPLETED", "INDEXED"):
                if model == self.provider.model_id and dimension == self.provider.dimension:
                    continue

            retry_count = meta.get("embedding_retry_count", 0)
            if not force and retry_count >= self.max_retries:
                # Exhausted maximum allowed retries; skip to prevent infinite loops
                continue

            cleaned_content = content.strip() if content else ""
            if not cleaned_content:
                continue

            content_hash = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()

            # Check if embedding already exists in cache
            cached_vector = None
            if self.pipeline.cache is not None:
                cached_vector = self.pipeline.cache.get(
                    content_hash=content_hash,
                    provider_id=self.provider.provider_id,
                    model_id=self.provider.model_id,
                    model_version=self.provider.model_version,
                    dimension=self.provider.dimension,
                )

            if cached_vector is not None:
                # Already cached! If status in metadata is not COMPLETED/INDEXED, update SQLite
                if status not in ("COMPLETED", "INDEXED"):
                    meta["embedding_status"] = "COMPLETED"
                    meta["embedding_model"] = self.provider.model_id
                    meta["embedding_dimension"] = self.provider.dimension
                    meta.pop("embedding_error", None)
                    chunks_to_mark_completed.append((cid, meta))
            else:
                unembedded.append(
                    ChunkItem(
                        chunk_id=cid,
                        content=content,
                        index=c_idx,
                        document_id=doc_id,
                        page_number=page_num,
                        metadata=meta,
                    )
                )

                if limit and len(unembedded) >= limit:
                    break

        # Synchronize metadata for chunks that are already in cache
        if chunks_to_mark_completed:
            with self.conn:
                for cid, updated_meta in chunks_to_mark_completed:
                    self.conn.execute(
                        "UPDATE document_chunks SET metadata_json = ? WHERE id = ?",
                        (json.dumps(updated_meta, separators=(",", ":")), cid),
                    )

        return unembedded

    def recover_document(self, document_id: str, force: bool = False) -> EmbeddingResult:
        """Execute recovery for all pending or failed chunks belonging to a document."""
        chunks = self.get_unembedded_chunks(document_id=document_id, force=force)
        if not chunks:
            return EmbeddingResult(total_chunks=0)

        # Mark all pending chunks as PENDING before embedding begins
        with self.conn:
            for c in chunks:
                row = self.conn.execute(
                    "SELECT metadata_json FROM document_chunks WHERE id = ?",
                    (c.chunk_id,),
                ).fetchone()
                if row:
                    meta = json.loads(row[0] or "{}")
                    meta["embedding_status"] = "PENDING"
                    self.conn.execute(
                        "UPDATE document_chunks SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta, separators=(",", ":")), c.chunk_id),
                    )

        def on_batch(succ: List[EmbeddedChunk], fail: List[FailedChunk]) -> None:
            update_chunk_embedding_metadata(self.conn, self.provider, successful=succ, failed=fail)

        result = self.pipeline.embed_chunks(chunks, self.provider, on_batch_complete=on_batch)
        return result

    def recover_all(
        self,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> EmbeddingResult:
        """Execute recovery across all documents for the active provider."""
        chunks = self.get_unembedded_chunks(project_id=project_id, limit=limit, force=force)
        if not chunks:
            return EmbeddingResult(total_chunks=0)

        # Mark all pending chunks as PENDING before embedding begins
        with self.conn:
            for c in chunks:
                row = self.conn.execute(
                    "SELECT metadata_json FROM document_chunks WHERE id = ?",
                    (c.chunk_id,),
                ).fetchone()
                if row:
                    meta = json.loads(row[0] or "{}")
                    meta["embedding_status"] = "PENDING"
                    self.conn.execute(
                        "UPDATE document_chunks SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta, separators=(",", ":")), c.chunk_id),
                    )

        def on_batch(succ: List[EmbeddedChunk], fail: List[FailedChunk]) -> None:
            update_chunk_embedding_metadata(self.conn, self.provider, successful=succ, failed=fail)

        result = self.pipeline.embed_chunks(chunks, self.provider, on_batch_complete=on_batch)
        return result

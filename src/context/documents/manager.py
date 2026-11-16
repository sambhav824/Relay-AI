"""End-to-end local document ingestion and optional Supabase synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import mimetypes
from pathlib import Path
import sqlite3
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.context.documents.chunker import TextChunker
from src.context.documents.extractors import DocumentExtractor
from src.context.documents.storage import SupabaseDocumentStorage
from src.context.sync.queue import SyncQueueManager

if TYPE_CHECKING:
    from src.context.embeddings.pipeline import EmbeddingPipeline
    from src.providers.embeddings import EmbeddingProvider

logger = logging.getLogger("RelayAI.DocumentManager")


@dataclass
class DocumentIngestionResult:
    document_id: str
    file_name: str
    file_hash: str
    chunk_count: int
    page_count: int
    storage_path: Optional[str]
    duplicate: bool = False
    status: str = "INDEXED"


class DocumentManager:
    """Coordinates extraction, deduplication, local FTS indexing, and cloud upload.

    Pass the active user's ID explicitly. Cloud storage is optional; when configured,
    the original file is uploaded before local metadata is committed. The sync queue
    receives document and chunk records for later PostgREST synchronization.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        storage: Optional[SupabaseDocumentStorage] = None,
        sync_queue: Optional[SyncQueueManager] = None,
        extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[TextChunker] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        max_file_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if not user_id:
            raise ValueError("An authenticated user_id is required")
        self.conn = connection
        self.user_id = user_id
        self.storage = storage
        self.sync_queue = sync_queue
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or TextChunker()
        self.embedding_pipeline = embedding_pipeline
        self.embedding_provider = embedding_provider
        self.max_file_size_bytes = max_file_size_bytes

    def ingest(self, file_path: str | Path, *, project_id: str = "global") -> DocumentIngestionResult:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size <= 0:
            raise ValueError("Document is empty")
        if size > self.max_file_size_bytes:
            raise ValueError(f"Document exceeds the {self.max_file_size_bytes // (1024 * 1024)} MB limit")

        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        existing = self.conn.execute(
            "SELECT id, page_count, metadata_json FROM documents WHERE file_hash=? AND project_id=? LIMIT 1",
            (digest, project_id),
        ).fetchone()
        if existing:
            meta = json.loads(existing[2] or "{}")
            if meta.get("owner_user_id") == self.user_id:
                count = self.conn.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id=?", (existing[0],)).fetchone()[0]
                return DocumentIngestionResult(existing[0], path.name, digest, count, existing[1], meta.get("storage_path"), True)

        extracted = self.extractor.extract(path)
        chunks = self.chunker.chunk(
            extracted.text,
            pages=extracted.pages,
            metadata={"source_file": path.name, **extracted.metadata},
        )
        if not chunks:
            raise ValueError("No text chunks were produced from this document")

        document_id = str(uuid.uuid4())
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        storage_path = None
        if self.storage:
            storage_path = self.storage.build_object_path(self.user_id, document_id, path.name)
            self.storage.upload(storage_path, data, mime_type)

        metadata = {
            "owner_user_id": self.user_id,
            "storage_path": storage_path,
            "extractor": extracted.metadata,
        }
        inserted_chunk_items: List[tuple[str, Any]] = []
        try:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO documents
                       (id,project_id,file_name,file_path,file_hash,file_size_bytes,mime_type,page_count,status,metadata_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (document_id, project_id, path.name, storage_path or str(path.resolve()),
                     digest, size, mime_type, extracted.page_count, "INDEXED",
                     json.dumps(metadata, separators=(",", ":"))),
                )
                # Queue the parent document before chunks to preserve remote FK order.
                if self.sync_queue:
                    self.sync_queue.stage(
                        user_id=self.user_id, table_name="documents", record_id=document_id,
                        operation="INSERT",
                        payload={"project_id": None if project_id == "global" else project_id,
                                 "file_name": path.name, "storage_path": storage_path or "",
                                 "file_hash": digest, "mime_type": mime_type,
                                 "file_size_bytes": size, "status": "INDEXED",
                                 "metadata": metadata},
                    )
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    inserted_chunk_items.append((chunk_id, chunk))
                    chunk_metadata = {
                        **chunk.metadata,
                        "owner_user_id": self.user_id,
                        "embedding_status": "PENDING",
                        "embedding_retry_count": 0,
                    }
                    self.conn.execute(
                        """INSERT INTO document_chunks
                           (id,document_id,project_id,chunk_index,content,token_estimate,page_number,metadata_json)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (chunk_id, document_id, project_id, chunk.index, chunk.content,
                         chunk.token_estimate, chunk.page_number,
                         json.dumps(chunk_metadata, separators=(",", ":"))),
                    )
                    if self.sync_queue:
                        self.sync_queue.stage(
                            user_id=self.user_id, table_name="document_chunks", record_id=chunk_id,
                            operation="INSERT",
                            payload={"document_id": document_id, "chunk_index": chunk.index,
                                     "content": chunk.content, "metadata": chunk_metadata},
                        )
        except Exception:
            if self.storage and storage_path:
                try:
                    self.storage.delete(storage_path)
                except Exception:
                    pass
            raise

        # Generate embeddings outside the SQLite transaction if pipeline and provider are configured
        if self.embedding_pipeline and self.embedding_provider:
            try:
                from src.context.embeddings.pipeline import ChunkItem
                from src.context.embeddings.recovery import update_chunk_embedding_metadata
                items_to_embed = [
                    ChunkItem(
                        chunk_id=cid,
                        content=c.content,
                        index=c.index,
                        document_id=document_id,
                        page_number=c.page_number,
                        metadata={**c.metadata, "owner_user_id": self.user_id, "embedding_status": "PENDING", "embedding_retry_count": 0},
                    )
                    for cid, c in inserted_chunk_items
                ]

                def on_batch(succ: list[Any], fail: list[Any]) -> None:
                    update_chunk_embedding_metadata(self.conn, self.embedding_provider, successful=succ, failed=fail)

                self.embedding_pipeline.embed_chunks(
                    items_to_embed,
                    self.embedding_provider,
                    on_batch_complete=on_batch,
                )
            except Exception as exc:
                logger.warning(
                    "Embedding generation during ingestion failed for document %s: %s. Chunks remain available for keyword search.",
                    document_id,
                    exc,
                )

        return DocumentIngestionResult(document_id, path.name, digest, len(chunks),
                                       extracted.page_count, storage_path)

    def delete(self, document_id: str) -> bool:
        row = self.conn.execute(
            "SELECT metadata_json FROM documents WHERE id=?", (document_id,)
        ).fetchone()
        if row is None:
            return False
        metadata = json.loads(row[0] or "{}")
        if metadata.get("owner_user_id") != self.user_id:
            raise PermissionError("Document belongs to another user")
        storage_path = metadata.get("storage_path")
        if self.storage and storage_path:
            self.storage.delete(storage_path)
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        if self.sync_queue:
            self.sync_queue.stage(user_id=self.user_id, table_name="documents",
                                  record_id=document_id, operation="DELETE", payload={})
        return True

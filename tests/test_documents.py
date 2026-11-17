"""Tests for Relay AI document extraction, chunking, and ingestion."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.context.db.migrations import MigrationManager
from src.context.documents.chunker import TextChunker
from src.context.documents.extractors import DocumentExtractor, ExtractionError
from src.context.documents.manager import DocumentManager
from src.context.documents.storage import SupabaseDocumentStorage


class TestTextChunker(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        text = ("A paragraph with useful context. " * 80)
        chunks = TextChunker(chunk_size=300, overlap=50).chunk(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.content) <= 300 for c in chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(c.token_estimate > 0 for c in chunks))

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(TextChunker().chunk("  "), [])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, overlap=100)


class TestDocumentExtractor(unittest.TestCase):
    def test_extract_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("Relay AI context document", encoding="utf-8")
            result = DocumentExtractor().extract(path)
            self.assertEqual(result.text, "Relay AI context document")
            self.assertEqual(result.page_count, 1)

    def test_extract_docx(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.docx"
            xml_content = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Hello Relay AI DOCX</w:t></w:r></w:p></w:body>'
                '</w:document>'
            )
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("word/document.xml", xml_content.encode("utf-8"))
            result = DocumentExtractor().extract(path)
            self.assertEqual(result.text, "Hello Relay AI DOCX")
            self.assertEqual(result.page_count, 1)

    def test_unsupported_extension_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.exe"
            path.write_bytes(b"binary")
            with self.assertRaises(ExtractionError):
                DocumentExtractor().extract(path)


class TestStoragePaths(unittest.TestCase):
    def test_user_scoped_object_path(self):
        path = SupabaseDocumentStorage.build_object_path("user-123", "doc-456", "../report final.pdf")
        self.assertTrue(path.startswith("user-123/doc-456/"))
        self.assertNotIn("..", path.split("/")[2])


class TestDocumentManager(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        MigrationManager(self.conn).apply_pending_migrations()

    def tearDown(self):
        self.conn.close()

    def test_ingest_indexes_chunks_and_fts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.txt"
            path.write_text("This is a Relay AI document about persistent context and retrieval.", encoding="utf-8")
            manager = DocumentManager(self.conn, user_id="user-1")
            result = manager.ingest(path)
            self.assertEqual(result.status, "INDEXED")
            self.assertGreater(result.chunk_count, 0)
            doc = self.conn.execute("SELECT file_hash,status FROM documents WHERE id=?", (result.document_id,)).fetchone()
            self.assertEqual(doc[1], "INDEXED")
            fts = self.conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE document_id=?", (result.document_id,)).fetchone()[0]
            self.assertEqual(fts, result.chunk_count)

    def test_duplicate_ingest_is_idempotent_for_same_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "same.txt"
            path.write_text("Same content for duplicate detection.", encoding="utf-8")
            manager = DocumentManager(self.conn, user_id="user-1")
            first = manager.ingest(path)
            second = manager.ingest(path)
            self.assertTrue(second.duplicate)
            self.assertEqual(first.document_id, second.document_id)

    def test_delete_checks_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "owned.txt"
            path.write_text("Owned document.", encoding="utf-8")
            first = DocumentManager(self.conn, user_id="user-1").ingest(path)
            other = DocumentManager(self.conn, user_id="user-2")
            with self.assertRaises(PermissionError):
                other.delete(first.document_id)

    def test_list_documents_filtering_and_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "doc1.txt"
            p1.write_text("Doc 1 content for user 1.", encoding="utf-8")
            p2 = Path(tmp) / "doc2.txt"
            p2.write_text("Doc 2 content for user 2.", encoding="utf-8")

            mgr1 = DocumentManager(self.conn, user_id="user-1")
            mgr2 = DocumentManager(self.conn, user_id="user-2")

            mgr1.ingest(p1, project_id="p-alpha")
            mgr2.ingest(p2, project_id="p-alpha")

            # user-1 only lists user-1 docs
            docs_user1 = mgr1.list_documents()
            self.assertEqual(len(docs_user1), 1)
            self.assertEqual(docs_user1[0]["file_name"], "doc1.txt")
            self.assertEqual(docs_user1[0]["project_id"], "p-alpha")
            self.assertGreater(docs_user1[0]["chunk_count"], 0)

            # user-2 only lists user-2 docs
            docs_user2 = mgr2.list_documents()
            self.assertEqual(len(docs_user2), 1)
            self.assertEqual(docs_user2[0]["file_name"], "doc2.txt")

            # Project filtering
            self.assertEqual(len(mgr1.list_documents(project_id="p-alpha")), 1)
            self.assertEqual(len(mgr1.list_documents(project_id="p-nonexistent")), 0)


if __name__ == "__main__":
    unittest.main()

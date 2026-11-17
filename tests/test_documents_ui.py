"""Unit tests for Relay AI Documents View and Background Ingestion Worker.

Validates:
- DocumentsView construction and UI components
- Document listing population in QTableWidget
- Size formatting utility
- DocumentIngestionWorker background execution and signal emissions (progress, completed, failed)
- Document deletion confirmation and callback handling
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from src.context.db.connection import initialize_database
from src.context.documents.manager import DocumentIngestionResult, DocumentManager
from src.ui.documents_view import DocumentIngestionWorker, DocumentsView

app = QApplication.instance() or QApplication([])


class TestDocumentsUI(unittest.TestCase):
    """Test suite for DocumentsView and DocumentIngestionWorker."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_docs_ui.db"
        self.db = initialize_database(str(self.db_path))
        self.conn = self.db.get_connection()
        self.manager = DocumentManager(self.conn, user_id="test-ui-user")

    def tearDown(self):
        self.conn.close()
        self.tmp_dir.cleanup()

    def test_view_construction_and_empty_state(self):
        """Verify DocumentsView renders cleanly with 0 documents initially."""
        view = DocumentsView(document_manager=self.manager)
        self.assertEqual(view.table.rowCount(), 0)
        self.assertIn("0 document(s) indexed", view.count_label.text())

    def test_format_size_helper(self):
        """Verify _format_size converts bytes to B, KB, and MB accurately."""
        view = DocumentsView(document_manager=self.manager)
        self.assertEqual(view._format_size(512), "512 B")
        self.assertEqual(view._format_size(2048), "2.0 KB")
        self.assertEqual(view._format_size(5 * 1024 * 1024), "5.0 MB")

    def test_documents_table_population(self):
        """Verify documents table displays ingested files with chunk and page counts."""
        # Create a sample document file and ingest it
        sample_file = Path(self.tmp_dir.name) / "sample_guide.txt"
        sample_file.write_text("Relay AI architecture guide and context documentation.", encoding="utf-8")

        result = self.manager.ingest(sample_file, project_id="global")
        self.assertIsNotNone(result.document_id)

        view = DocumentsView(document_manager=self.manager)
        self.assertEqual(view.table.rowCount(), 1)

        # Check column values
        name_item = view.table.item(0, 0)
        self.assertEqual(name_item.text(), "sample_guide.txt")

        pages_item = view.table.item(0, 2)
        self.assertEqual(pages_item.text(), "1")

        chunks_item = view.table.item(0, 3)
        self.assertGreater(int(chunks_item.text()), 0)

        status_item = view.table.item(0, 4)
        self.assertEqual(status_item.text(), "INDEXED")

        # Check delete button widget exists in column 5
        del_btn = view.table.cellWidget(0, 5)
        self.assertIsNotNone(del_btn)
        self.assertEqual(del_btn.text(), "Delete")

    def test_ingestion_worker_successful_run(self):
        """Verify DocumentIngestionWorker executes in thread and emits progress and completed signals."""
        sample_file = Path(self.tmp_dir.name) / "worker_test.md"
        sample_file.write_text("# Worker Test\nDocument chunking and worker execution test.", encoding="utf-8")

        worker = DocumentIngestionWorker(self.manager, str(sample_file), project_id="global")

        progress_events = []
        completed_results = []
        failed_errors = []

        worker.progress.connect(lambda msg, pct: progress_events.append((msg, pct)))
        worker.completed.connect(lambda res: completed_results.append(res))
        worker.failed.connect(lambda err: failed_errors.append(err))

        worker.run()

        self.assertGreater(len(progress_events), 0)
        self.assertEqual(len(failed_errors), 0)
        self.assertEqual(len(completed_results), 1)

        res = completed_results[0]
        self.assertEqual(res.file_name, "worker_test.md")
        self.assertGreater(res.chunk_count, 0)

    def test_ingestion_worker_file_not_found(self):
        """Verify DocumentIngestionWorker emits failed signal when file does not exist."""
        worker = DocumentIngestionWorker(self.manager, "C:/non_existent_file_path.pdf")

        failed_errors = []
        worker.failed.connect(lambda err: failed_errors.append(err))

        worker.run()

        self.assertEqual(len(failed_errors), 1)
        self.assertIn("File not found", failed_errors[0])

    @patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes)
    def test_document_deletion_from_view(self, mock_msgbox):
        """Verify deleting a document from view removes it from SQLite and refreshes table."""
        sample_file = Path(self.tmp_dir.name) / "delete_me.txt"
        sample_file.write_text("Temporary file to be deleted.", encoding="utf-8")

        result = self.manager.ingest(sample_file)
        view = DocumentsView(document_manager=self.manager)
        self.assertEqual(view.table.rowCount(), 1)

        deleted_ids = []
        view.document_deleted.connect(lambda did: deleted_ids.append(did))

        view._on_delete_document(result.document_id, "delete_me.txt")

        # Table row count should now be 0
        self.assertEqual(view.table.rowCount(), 0)
        self.assertEqual(len(deleted_ids), 1)
        self.assertEqual(deleted_ids[0], result.document_id)

        # Confirm gone from SQLite
        self.assertEqual(len(self.manager.list_documents()), 0)


if __name__ == "__main__":
    unittest.main()

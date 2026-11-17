"""Relay AI Universal Context Engine — Documents & RAG Management View.

Provides a responsive PySide6 UI for document file selection, asynchronous background
ingestion with progress indicators, document listing, and deletion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.context.documents.manager import DocumentIngestionResult, DocumentManager

logger = logging.getLogger("RelayAI.DocumentsView")


class DocumentIngestionWorker(QThread):
    """Background worker thread executing document extraction and indexing without blocking the GUI."""

    progress = Signal(str, int)  # (status_message, percentage_0_to_100)
    completed = Signal(object)  # (DocumentIngestionResult)
    failed = Signal(str)  # (error_message)

    def __init__(
        self,
        manager: DocumentManager,
        file_path: str,
        project_id: str = "global",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self.file_path = file_path
        self.project_id = project_id

    def run(self) -> None:
        try:
            path = Path(self.file_path)
            if not path.is_file():
                self.failed.emit(f"File not found: {self.file_path}")
                return

            self.progress.emit(f"Reading {path.name}...", 20)
            self.progress.emit("Extracting text and chunking...", 50)

            result = self.manager.ingest(self.file_path, project_id=self.project_id)

            self.progress.emit("Indexing chunks into full-text & vector store...", 85)
            self.progress.emit("Completed", 100)
            self.completed.emit(result)
        except Exception as exc:
            logger.error("Document ingestion worker failed: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class DocumentsView(QWidget):
    """Document management panel for viewing, adding, and deleting indexed documents."""

    document_indexed = Signal(object)  # Emitted when a new document is successfully indexed
    document_deleted = Signal(str)  # Emitted when a document is deleted (doc_id)

    def __init__(
        self,
        document_manager: Optional[DocumentManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.document_manager = document_manager
        self._active_workers: List[DocumentIngestionWorker] = []

        self._build_ui()
        if self.document_manager is not None:
            self.refresh_documents()

    def set_document_manager(self, manager: DocumentManager) -> None:
        """Update active document manager instance and reload table."""
        self.document_manager = manager
        self.refresh_documents()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header Title and Subtitle
        header_box = QVBoxLayout()
        header_box.setSpacing(4)
        title = QLabel("📁   Documents & Context")
        title.setObjectName("PageTitle")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        header_box.addWidget(title)

        subtitle = QLabel(
            "Index local documents (PDF, Word, Markdown, Text) for private retrieval-augmented generation (RAG)."
        )
        subtitle.setObjectName("Muted")
        subtitle.setStyleSheet("color: #888888; font-size: 13px;")
        header_box.addWidget(subtitle)
        layout.addLayout(header_box)

        # Ingestion Action Card
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); "
            "border-radius: 8px; padding: 12px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(10)

        card_title = QLabel("Upload & Index Document")
        card_title.setStyleSheet("font-weight: bold; font-size: 15px;")
        card_layout.addWidget(card_title)

        info_lbl = QLabel(
            "Supported formats: PDF (.pdf), Word (.docx), Markdown (.md), Plain text (.txt), CSV, JSON, LOG. Maximum 50 MB per file."
        )
        info_lbl.setObjectName("Muted")
        info_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        card_layout.addWidget(info_lbl)

        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("✚  Select Document to Index")
        self.upload_btn.setObjectName("Primary")
        self.upload_btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; padding: 8px 16px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        self.upload_btn.clicked.connect(self._on_select_file)
        btn_row.addWidget(self.upload_btn)
        btn_row.addStretch()

        self.refresh_btn = QPushButton("↻ Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_documents)
        btn_row.addWidget(self.refresh_btn)
        card_layout.addLayout(btn_row)

        # Ingestion Progress Bar & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.hide()
        card_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("Muted")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        card_layout.addWidget(self.status_label)

        layout.addWidget(card)

        # Documents Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Document Name", "Size", "Pages", "Chunks", "Status", "Actions"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { gridline-color: rgba(255, 255, 255, 0.05); }"
            "QHeaderView::section { padding: 6px; font-weight: bold; background-color: rgba(255, 255, 255, 0.04); }"
        )
        layout.addWidget(self.table, 1)

        # Summary footer
        self.count_label = QLabel("0 document(s) indexed.")
        self.count_label.setObjectName("Muted")
        self.count_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(self.count_label)

    def _format_size(self, size_bytes: int) -> str:
        """Format raw byte size into human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def refresh_documents(self) -> None:
        """Reload document list from SQLite database into the table."""
        if self.document_manager is None:
            self.table.setRowCount(0)
            self.count_label.setText("Document manager not configured.")
            return

        try:
            docs = self.document_manager.list_documents()
        except Exception as exc:
            logger.error("Failed to list documents: %s", exc)
            self.status_label.setText(f"Error loading documents: {exc}")
            return

        self.table.setRowCount(len(docs))
        for row_idx, doc in enumerate(docs):
            name_item = QTableWidgetItem(doc.get("file_name", "Untitled"))
            name_item.setToolTip(f"ID: {doc.get('id')}\nPath: {doc.get('file_path')}")
            self.table.setItem(row_idx, 0, name_item)

            size_str = self._format_size(int(doc.get("file_size_bytes", 0)))
            size_item = QTableWidgetItem(size_str)
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 1, size_item)

            pages_item = QTableWidgetItem(str(doc.get("page_count", 1)))
            pages_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 2, pages_item)

            chunks_item = QTableWidgetItem(str(doc.get("chunk_count", 0)))
            chunks_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 3, chunks_item)

            status_item = QTableWidgetItem(str(doc.get("status", "INDEXED")))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 4, status_item)

            # Delete action button
            del_btn = QPushButton("Delete")
            del_btn.setStyleSheet(
                "QPushButton { background-color: #dc2626; color: white; padding: 4px 10px; border-radius: 4px; font-size: 11px; }"
                "QPushButton:hover { background-color: #b91c1c; }"
            )
            doc_id = doc.get("id")
            file_name = doc.get("file_name", "Document")
            del_btn.clicked.connect(lambda checked=False, d_id=doc_id, fn=file_name: self._on_delete_document(d_id, fn))
            self.table.setCellWidget(row_idx, 5, del_btn)

        self.count_label.setText(f"{len(docs)} document(s) indexed for RAG context.")

    def _on_select_file(self) -> None:
        """Open file dialog and launch background ingestion worker."""
        if self.document_manager is None:
            QMessageBox.warning(self, "Context Unavailable", "Document manager is not initialized.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document to Index",
            "",
            "Supported Documents (*.pdf *.docx *.txt *.md *.csv *.json *.log);;All Files (*.*)",
        )
        if not file_path:
            return

        self.upload_btn.setEnabled(False)
        self.progress_bar.setValue(10)
        self.progress_bar.show()
        self.status_label.setText(f"Preparing to index {Path(file_path).name}...")

        worker = DocumentIngestionWorker(self.document_manager, file_path, parent=self)
        self._active_workers.append(worker)

        worker.progress.connect(self._on_ingestion_progress)
        worker.completed.connect(self._on_ingestion_completed)
        worker.failed.connect(self._on_ingestion_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        worker.start()

    def _on_ingestion_progress(self, msg: str, percentage: int) -> None:
        self.progress_bar.setValue(percentage)
        self.status_label.setText(msg)

    def _on_ingestion_completed(self, result: DocumentIngestionResult) -> None:
        self.upload_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.hide()

        status_msg = f"✓ Successfully indexed '{result.file_name}' ({result.chunk_count} chunks, {result.page_count} pages)."
        if getattr(result, "duplicate", False):
            status_msg = f"ℹ Document '{result.file_name}' was already indexed ({result.chunk_count} chunks)."
        self.status_label.setText(status_msg)

        self.refresh_documents()
        self.document_indexed.emit(result)

    def _on_ingestion_failed(self, error_message: str) -> None:
        self.upload_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText(f"✗ Ingestion failed: {error_message}")
        QMessageBox.warning(self, "Ingestion Error", f"Failed to ingest document:\n{error_message}")

    def _on_delete_document(self, document_id: str, file_name: str) -> None:
        """Prompt user confirmation and delete document from SQLite and vector store."""
        confirm = QMessageBox.question(
            self,
            "Delete Document",
            f"Are you sure you want to delete '{file_name}'?\n\nThis will remove the document and all its indexed chunks from retrieval.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        if self.document_manager is None:
            return

        try:
            success = self.document_manager.delete(document_id)
            if success:
                self.status_label.setText(f"Deleted '{file_name}'.")
                self.refresh_documents()
                self.document_deleted.emit(document_id)
            else:
                self.status_label.setText(f"Document '{file_name}' not found.")
        except Exception as exc:
            logger.error("Failed to delete document %s: %s", document_id, exc)
            QMessageBox.warning(self, "Delete Error", f"Could not delete document:\n{exc}")

    def _cleanup_worker(self, worker: DocumentIngestionWorker) -> None:
        if worker in self._active_workers:
            self._active_workers.remove(worker)

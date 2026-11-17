"""Unit tests for Relay AI RAG Integration in ChatWorker and RoutedChatWorker.

Validates:
- Retrieval of relevant document chunks before provider dispatch
- Injection of structured reference context with document names and page attribution
- Handling of empty retrieval results (no fake context injected)
- Resilient error handling (retrieval errors do not crash chat streaming)
- Source attribution signal emission (context_retrieved)
- Offline preference enforcement (prefer_offline passed accurately)
"""

from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.context.retrieval.hybrid import (
    HybridConfig,
    HybridRetriever,
    HybridSearchResult,
    format_rag_context,
)
from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.providers.base import AIProvider
from src.routing.types import FailureReason, RoutingCandidate, RoutingDecision
from src.routing.worker import RoutedChatWorker
from src.ui.main_window import ChatWorker

app = QApplication.instance() or QApplication([])


class DummyRAGProvider(AIProvider):
    """Mock provider recording received messages for context verification."""

    def __init__(self, provider_id: str = "mock-rag"):
        self.provider_id = provider_id
        self.display_name = "Mock RAG Provider"
        self.is_local = True
        self.received_messages: List[Any] = []

    def validate_connection(self) -> bool:
        return True

    def get_available_models(self) -> list:
        return []

    def stream(self, messages: List[Any], model: str, **kwargs) -> Iterator[StreamChunk]:
        self.received_messages = list(messages)
        yield StreamChunk(delta="RAG Response based on context", is_final=False)
        yield StreamChunk(delta="", is_final=True)

    def generate(self, messages: List[Any], model: str, **kwargs) -> str:
        self.received_messages = list(messages)
        return "RAG Response based on context"


class TestChatRAGIntegration(unittest.TestCase):
    """Test suite for HybridRetriever connection to ChatWorker and RoutedChatWorker."""

    def setUp(self):
        self.provider = DummyRAGProvider()
        self.mock_retriever = MagicMock(spec=HybridRetriever)

        # Sample search result
        self.sample_chunk = HybridSearchResult(
            id="c1",
            document_id="doc-123",
            project_id="global",
            chunk_index=0,
            content="Relay AI is a multi-model desktop AI assistant supporting local and cloud LLMs.",
            metadata={"source_file": "relay_architecture.pdf", "page_number": 3},
            rrf_score=0.032,
            dense_rank=1,
            sparse_rank=1,
        )

    def test_format_rag_context_structure_and_attribution(self):
        """Verify format_rag_context formats source names, page numbers, and instructions."""
        formatted = format_rag_context([self.sample_chunk])

        self.assertIn("--- Context from Indexed Documents ---", formatted)
        self.assertIn("[Source 1: relay_architecture.pdf (Page 3)]", formatted)
        self.assertIn("Relay AI is a multi-model desktop AI assistant", formatted)
        self.assertIn("Instructions: Use the above indexed document context", formatted)

    def test_format_rag_context_empty(self):
        """Verify format_rag_context returns empty string when chunks list is empty."""
        self.assertEqual(format_rag_context([]), "")

    def test_chat_worker_retrieves_and_injects_context(self):
        """Verify ChatWorker queries retriever, emits context_retrieved, and injects context."""
        self.mock_retriever.search.return_value = [self.sample_chunk]

        user_msg = Message(MessageRole.USER, "What is Relay AI?")
        worker = ChatWorker(
            provider=self.provider,
            messages=[user_msg],
            model="mock-model",
            retriever=self.mock_retriever,
            prefer_offline=True,
        )

        captured_chunks = []
        worker.context_retrieved.connect(lambda chunks: captured_chunks.extend(chunks))

        worker.run()

        # 1. Verify retriever was called with user prompt and prefer_offline=True
        self.mock_retriever.search.assert_called_once_with(
            "What is Relay AI?",
            match_count=5,
            prefer_offline=True,
        )

        # 2. Verify signal was emitted with the retrieved chunk
        self.assertEqual(len(captured_chunks), 1)
        self.assertEqual(captured_chunks[0].id, "c1")

        # 3. Verify messages sent to provider contain system instruction with RAG context
        self.assertGreaterEqual(len(self.provider.received_messages), 2)
        system_msg = self.provider.received_messages[0]
        self.assertEqual(system_msg.role, MessageRole.SYSTEM)
        self.assertIn("relay_architecture.pdf", system_msg.content)
        self.assertIn("Page 3", system_msg.content)

    def test_chat_worker_no_chunks_leaves_messages_unchanged(self):
        """Verify that when no chunks match, no fake context is added."""
        self.mock_retriever.search.return_value = []

        user_msg = Message(MessageRole.USER, "Obscure topic with no docs")
        worker = ChatWorker(
            provider=self.provider,
            messages=[user_msg],
            model="mock-model",
            retriever=self.mock_retriever,
        )

        captured_chunks = []
        worker.context_retrieved.connect(lambda chunks: captured_chunks.extend(chunks))

        worker.run()

        # No chunks emitted
        self.assertEqual(len(captured_chunks), 0)
        # Message count unchanged (only user message)
        self.assertEqual(len(self.provider.received_messages), 1)
        self.assertEqual(self.provider.received_messages[0].content, "Obscure topic with no docs")

    def test_chat_worker_retrieval_exception_fails_gracefully(self):
        """Verify that a database/retrieval error does not crash the chat worker."""
        self.mock_retriever.search.side_effect = RuntimeError("SQLite database locked")

        user_msg = Message(MessageRole.USER, "Hello")
        worker = ChatWorker(
            provider=self.provider,
            messages=[user_msg],
            model="mock-model",
            retriever=self.mock_retriever,
        )

        completed_texts = []
        worker.completed.connect(lambda txt: completed_texts.append(txt))

        # Must execute cleanly without unhandled exception
        worker.run()

        self.assertEqual(len(completed_texts), 1)
        self.assertEqual(completed_texts[0], "RAG Response based on context")

    def test_routed_chat_worker_retrieves_and_injects_context(self):
        """Verify RoutedChatWorker queries retriever and injects RAG context across attempts."""
        self.mock_retriever.search.return_value = [self.sample_chunk]

        candidate = RoutingCandidate(
            provider_id="mock-rag",
            model_id="mock-model",
            display_name="Mock Model",
            composite_score=0.9,
        )
        decision = RoutingDecision(
            primary=candidate,
            fallback_chain=[],
        )

        mock_pm = MagicMock()
        mock_pm.get_provider.return_value = self.provider

        worker = RoutedChatWorker(
            decision=decision,
            prompt="Tell me about Relay AI architecture",
            history=[],
            provider_manager=mock_pm,
            retriever=self.mock_retriever,
            prefer_offline=True,
        )

        captured_chunks = []
        worker.context_retrieved.connect(lambda chunks: captured_chunks.extend(chunks))

        worker.run()

        # Verify retriever called
        self.mock_retriever.search.assert_called_once_with(
            "Tell me about Relay AI architecture",
            match_count=5,
            prefer_offline=True,
        )
        self.assertEqual(len(captured_chunks), 1)

        # Verify provider received RAG context in system message
        self.assertGreaterEqual(len(self.provider.received_messages), 2)
        system_msg = self.provider.received_messages[0]
        self.assertEqual(system_msg.role, MessageRole.SYSTEM)
        self.assertIn("relay_architecture.pdf", system_msg.content)


if __name__ == "__main__":
    unittest.main()

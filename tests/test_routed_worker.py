import os
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from src.core.events import StreamChunk
from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.routing.types import FailureReason, RoutingCandidate, RoutingDecision
from src.routing.worker import RoutedChatWorker, categorize_failure

app = QApplication.instance() or QApplication([])


class MockStreamProvider(AIProvider):
    """Mock AIProvider for testing RoutedChatWorker without network calls or API keys."""

    def __init__(
        self,
        provider_id: str,
        stream_chunks: Optional[List[StreamChunk]] = None,
        exception_to_raise: Optional[Exception] = None,
        fail_after_n_chunks: Optional[int] = None,
        sleep_between_chunks: float = 0.0,
    ):
        self.provider_id = provider_id
        self.display_name = provider_id.title()
        self.is_local = (provider_id == "ollama")
        self.stream_chunks = stream_chunks or []
        self.exception_to_raise = exception_to_raise
        self.fail_after_n_chunks = fail_after_n_chunks
        self.sleep_between_chunks = sleep_between_chunks
        self.recorded_calls: List[Dict[str, Any]] = []

    def stream(self, messages, model, **kwargs):
        self.recorded_calls.append({
            "messages": [Message(m.role, m.content) for m in messages],
            "model": model,
        })

        if self.exception_to_raise and self.fail_after_n_chunks is None:
            raise self.exception_to_raise

        for idx, chunk in enumerate(self.stream_chunks):
            if self.fail_after_n_chunks is not None and idx >= self.fail_after_n_chunks:
                if self.exception_to_raise:
                    raise self.exception_to_raise
                raise RuntimeError("Failed after emitting chunks")

            if self.sleep_between_chunks > 0:
                time.sleep(self.sleep_between_chunks)
            yield chunk

    def generate(self, messages, model, **kwargs):
        return "".join(c.delta for c in self.stream_chunks)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo("default-model", "Default Model", self.provider_id)]


class TestRoutedWorker(unittest.TestCase):
    def setUp(self):
        self.mock_cred_store = MagicMock(spec=CredentialStore)
        self.provider_mgr = ProviderManager(credential_store=self.mock_cred_store)
        self.provider_mgr._providers.clear()

        # Build mock providers
        self.p_groq = MockStreamProvider(
            "groq",
            stream_chunks=[StreamChunk("Groq "), StreamChunk("response.")],
        )
        self.p_openai = MockStreamProvider(
            "openai",
            stream_chunks=[StreamChunk("OpenAI "), StreamChunk("fallback response.")],
        )
        self.p_ollama = MockStreamProvider(
            "ollama",
            stream_chunks=[StreamChunk("Ollama "), StreamChunk("local response.")],
        )

        self.provider_mgr.register_provider(self.p_groq)
        self.provider_mgr.register_provider(self.p_openai)
        self.provider_mgr.register_provider(self.p_ollama)

        # Standard 3-candidate routing decision
        self.cand_primary = RoutingCandidate(
            provider_id="groq",
            model_id="openai/gpt-oss-120b",
            display_name="GPT-OSS 120B (Groq)",
            composite_score=0.92,
        )
        self.cand_fb1 = RoutingCandidate(
            provider_id="openai",
            model_id="gpt-4.1",
            display_name="GPT-4.1 (OpenAI)",
            composite_score=0.88,
        )
        self.cand_fb2 = RoutingCandidate(
            provider_id="ollama",
            model_id="llama3.2:3b",
            display_name="Llama 3.2 3B (Ollama)",
            composite_score=0.75,
        )
        self.decision = RoutingDecision(
            primary=self.cand_primary,
            fallback_chain=[self.cand_fb1, self.cand_fb2],
        )

    def test_primary_success(self):
        """Primary candidate succeeds on attempt 0 with full streaming and no fallback."""
        prompt = "Hello AI"
        history = [Message(MessageRole.USER, "Old prompt"), Message(MessageRole.ASSISTANT, "Old answer")]

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt=prompt,
            history=history,
            provider_manager=self.provider_mgr,
        )

        started_events = []
        chunks = []
        success_events = []
        failed_events = []
        fallback_events = []

        worker.attempt_started.connect(lambda p, m, idx: started_events.append((p, m, idx)))
        worker.chunk_received.connect(lambda d: chunks.append(d))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))
        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.fallback_started.connect(lambda fp, fm, tp, tm, fidx: fallback_events.append((fp, fm, tp, tm, fidx)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0], ("groq", "openai/gpt-oss-120b", 0))
        self.assertEqual(chunks, ["Groq ", "response."])
        self.assertEqual(len(failed_events), 0)
        self.assertEqual(len(fallback_events), 0)

        self.assertEqual(len(success_events), 1)
        final_p, final_m, final_txt, final_msgs = success_events[0]
        self.assertEqual(final_p, "groq")
        self.assertEqual(final_m, "openai/gpt-oss-120b")
        self.assertEqual(final_txt, "Groq response.")

        # Exactly 2 messages to commit: user prompt and final assistant response
        self.assertEqual(len(final_msgs), 2)
        self.assertEqual(final_msgs[0].role, MessageRole.USER)
        self.assertEqual(final_msgs[0].content, "Hello AI")
        self.assertEqual(final_msgs[1].role, MessageRole.ASSISTANT)
        self.assertEqual(final_msgs[1].content, "Groq response.")

    def test_primary_failure_followed_by_successful_fallback(self):
        """Primary fails immediately with rate limit; fallback 1 succeeds cleanly."""
        self.p_groq.exception_to_raise = ProviderRateLimitError("Rate limit exceeded for groq", provider_id="groq")

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Compute this math problem",
            history=None,
            provider_manager=self.provider_mgr,
        )

        started_events = []
        failed_events = []
        fallback_events = []
        success_events = []

        worker.attempt_started.connect(lambda p, m, idx: started_events.append((p, m, idx)))
        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.fallback_started.connect(lambda fp, fm, tp, tm, fidx: fallback_events.append((fp, fm, tp, tm, fidx)))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        # Both groq (0) and openai (1) should have started
        self.assertEqual(len(started_events), 2)
        self.assertEqual(started_events[0], ("groq", "openai/gpt-oss-120b", 0))
        self.assertEqual(started_events[1], ("openai", "gpt-4.1", 1))

        # Groq failed with rate_limit
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0][0], "groq")
        self.assertEqual(failed_events[0][2], "rate_limit")

        # Fallback started signal emitted
        self.assertEqual(len(fallback_events), 1)
        self.assertEqual(fallback_events[0], ("groq", "openai/gpt-oss-120b", "openai", "gpt-4.1", 1))

        # Final success emitted by OpenAI
        self.assertEqual(len(success_events), 1)
        self.assertEqual(success_events[0][0], "openai")
        self.assertEqual(success_events[0][2], "OpenAI fallback response.")

    def test_all_three_attempts_fail(self):
        """When primary and both fallbacks fail, final_failure is emitted with full attempt audit."""
        self.p_groq.exception_to_raise = ProviderConnectionError("Connection failed", provider_id="groq")
        self.p_openai.exception_to_raise = ProviderTimeoutError("Gateway timed out", provider_id="openai")
        self.p_ollama.exception_to_raise = ProviderAuthenticationError("Unauthorized", provider_id="ollama")

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Summarize article",
            history=None,
            provider_manager=self.provider_mgr,
        )

        started_events = []
        failed_events = []
        final_failure_events = []
        success_events = []

        worker.attempt_started.connect(lambda p, m, idx: started_events.append((p, m, idx)))
        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.final_failure.connect(lambda summary, records: final_failure_events.append((summary, records)))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(started_events), 3)
        self.assertEqual(len(failed_events), 3)
        self.assertEqual(len(success_events), 0)

        self.assertEqual(len(final_failure_events), 1)
        summary, records = final_failure_events[0]
        self.assertIn("All 3 routing candidate attempts failed", summary)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].failure_reason, FailureReason.CONNECTION_ERROR)
        self.assertEqual(records[1].failure_reason, FailureReason.TIMEOUT)
        self.assertEqual(records[2].failure_reason, FailureReason.AUTHENTICATION)

    def test_failure_after_partial_streaming_discards_partial(self):
        """If primary fails after partial streaming, partial output is discarded and not mixed with fallback."""
        # Groq will emit 2 chunks, then raise RateLimitError
        self.p_groq.stream_chunks = [
            StreamChunk("Partial tokens from primary "),
            StreamChunk("that will get discarded."),
            StreamChunk("Unreachable chunk"),
        ]
        self.p_groq.fail_after_n_chunks = 2
        self.p_groq.exception_to_raise = ProviderRateLimitError("Rate limit in mid-stream", provider_id="groq")

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Explain recursion",
            history=None,
            provider_manager=self.provider_mgr,
        )

        chunks_streamed = []
        success_events = []
        failed_events = []

        worker.chunk_received.connect(lambda d: chunks_streamed.append(d))
        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        # Both primary chunks and fallback chunks were streamed over the wire
        self.assertIn("Partial tokens from primary ", chunks_streamed)
        self.assertIn("OpenAI ", chunks_streamed)

        # Primary failed
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0][0], "groq")

        # Fallback succeeded
        self.assertEqual(len(success_events), 1)
        final_provider, final_model, final_text, final_msgs = success_events[0]
        self.assertEqual(final_provider, "openai")

        # STRICT VERIFICATION: The partial primary tokens are NOT in final_text!
        self.assertNotIn("Partial tokens", final_text)
        self.assertEqual(final_text, "OpenAI fallback response.")

        # Assistant message to commit contains ONLY the clean fallback response
        self.assertEqual(final_msgs[1].content, "OpenAI fallback response.")

    def test_no_duplicated_user_prompt(self):
        """Across primary failure and fallback execution, user prompt is appended exactly once per provider call."""
        self.p_groq.exception_to_raise = ProviderConnectionError("Groq host down", provider_id="groq")

        history = [Message(MessageRole.USER, "Turn 1"), Message(MessageRole.ASSISTANT, "Turn 2")]
        prompt = "Turn 3 user prompt"

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt=prompt,
            history=history,
            provider_manager=self.provider_mgr,
        )

        success_events = []
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(success_events), 1)
        final_msgs = success_events[0][3]

        # Messages to commit has exactly 1 USER and 1 ASSISTANT message
        self.assertEqual(len(final_msgs), 2)
        self.assertEqual(final_msgs[0].role, MessageRole.USER)
        self.assertEqual(final_msgs[0].content, prompt)

        # Groq received 3 messages
        groq_msgs = self.p_groq.recorded_calls[0]["messages"]
        self.assertEqual(len(groq_msgs), 3)
        self.assertEqual(groq_msgs[2].content, prompt)

        # OpenAI fallback received 3 messages
        openai_msgs = self.p_openai.recorded_calls[0]["messages"]
        self.assertEqual(len(openai_msgs), 3)
        self.assertEqual(openai_msgs[2].content, prompt)
        self.assertEqual([m.role for m in openai_msgs], [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER])

    def test_conversation_history_remains_unchanged(self):
        """Supplied conversation history list and Message instances are never mutated."""
        history = [
            Message(MessageRole.USER, "Original turn 1"),
            Message(MessageRole.ASSISTANT, "Original turn 2"),
        ]
        history_copy = [Message(m.role, m.content) for m in history]

        self.p_groq.exception_to_raise = ProviderRateLimitError("Rate limit", provider_id="groq")

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="New prompt",
            history=history,
            provider_manager=self.provider_mgr,
        )

        worker.start()
        worker.wait(5000)
        app.processEvents()

        # Input history must remain length 2 and identical
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content, history_copy[0].content)
        self.assertEqual(history[1].content, history_copy[1].content)

    def test_cancellation_during_streaming(self):
        """Worker terminates immediately when cancelled during streaming; emits cancelled signal."""
        # Provider yields chunks with delay so we can cancel mid-stream
        self.p_groq.stream_chunks = [
            StreamChunk("Chunk 1 "),
            StreamChunk("Chunk 2 "),
            StreamChunk("Chunk 3 "),
        ]
        self.p_groq.sleep_between_chunks = 0.05

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Cancel me",
            history=None,
            provider_manager=self.provider_mgr,
        )

        cancelled_events = []
        success_events = []
        failure_events = []

        def on_chunk(delta):
            worker.cancel()

        worker.chunk_received.connect(on_chunk, Qt.DirectConnection)
        worker.cancelled.connect(lambda: cancelled_events.append(True))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append(txt))
        worker.final_failure.connect(lambda summary, recs: failure_events.append(summary))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(cancelled_events), 1)
        self.assertEqual(len(success_events), 0)
        self.assertEqual(len(failure_events), 0)
        self.assertTrue(worker.isFinished())

    def test_cancellation_between_attempts(self):
        """When cancelled between failed primary and fallback, worker terminates without attempting fallback."""
        self.p_groq.exception_to_raise = ProviderRateLimitError("Rate limit", provider_id="groq")

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Cancel between attempts",
            history=None,
            provider_manager=self.provider_mgr,
        )

        cancelled_events = []
        started_events = []

        def on_failed(p, m, r, err):
            # Request cancellation right after primary failure
            worker.cancel()

        worker.attempt_started.connect(lambda p, m, idx: started_events.append((p, m, idx)))
        worker.attempt_failed.connect(on_failed, Qt.DirectConnection)
        worker.cancelled.connect(lambda: cancelled_events.append(True))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(cancelled_events), 1)
        # Only Groq attempt (0) was started; OpenAI attempt (1) was cancelled before start
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0][0], "groq")
        self.assertEqual(len(self.p_openai.recorded_calls), 0)

    def test_per_attempt_timeout_enforcement(self):
        """An attempt exceeding timeout_sec is aborted and categorized as FailureReason.TIMEOUT."""
        self.p_groq.stream_chunks = [StreamChunk("Token 1"), StreamChunk("Token 2")]
        self.p_groq.sleep_between_chunks = 0.08  # Total > 0.05s

        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Quick timeout test",
            history=None,
            provider_manager=self.provider_mgr,
            timeout_sec=0.05,  # 50ms timeout
        )

        failed_events = []
        success_events = []

        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        # Groq should have timed out
        self.assertTrue(len(failed_events) >= 1)
        self.assertEqual(failed_events[0][0], "groq")
        self.assertEqual(failed_events[0][2], "timeout")

        # OpenAI fallback should succeed
        self.assertEqual(len(success_events), 1)
        self.assertEqual(success_events[0][0], "openai")

    def test_missing_provider_in_manager_handled_gracefully(self):
        """If a candidate's provider is not registered, worker records failure and moves to next fallback."""
        cand_unregistered = RoutingCandidate(
            provider_id="unregistered_provider",
            model_id="custom_model",
            display_name="Unknown Provider",
            composite_score=0.99,
        )
        decision_with_missing = RoutingDecision(
            primary=cand_unregistered,
            fallback_chain=[self.cand_primary],
        )

        worker = RoutedChatWorker(
            decision=decision_with_missing,
            prompt="Test missing provider",
            history=None,
            provider_manager=self.provider_mgr,
        )

        failed_events = []
        success_events = []

        worker.attempt_failed.connect(lambda p, m, r, err: failed_events.append((p, m, r, err)))
        worker.final_success.connect(lambda p, m, txt, msgs: success_events.append((p, m, txt, msgs)))

        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0][0], "unregistered_provider")
        self.assertEqual(failed_events[0][2], "model_unavailable")

        # Primary groq succeeds
        self.assertEqual(len(success_events), 1)
        self.assertEqual(success_events[0][0], "groq")

    def test_correct_failure_categorization(self):
        """categorize_failure maps all standard and vendor exceptions to proper FailureReason values."""
        self.assertEqual(categorize_failure(ProviderRateLimitError("Quota limit")), FailureReason.RATE_LIMIT)
        self.assertEqual(categorize_failure(ProviderAuthenticationError("Invalid API key")), FailureReason.AUTHENTICATION)
        self.assertEqual(categorize_failure(ProviderTimeoutError("Deadline exceeded")), FailureReason.TIMEOUT)
        self.assertEqual(categorize_failure(TimeoutError("Socket timeout")), FailureReason.TIMEOUT)
        self.assertEqual(categorize_failure(ModelNotFoundError("Model 404")), FailureReason.MODEL_UNAVAILABLE)
        self.assertEqual(categorize_failure(ProviderConnectionError("Connection refused")), FailureReason.CONNECTION_ERROR)
        self.assertEqual(categorize_failure(ConnectionResetError("Peer reset")), FailureReason.CONNECTION_ERROR)

        # Keyword and vendor-style exception text
        self.assertEqual(categorize_failure(RuntimeError("HTTP 429 Too Many Requests")), FailureReason.RATE_LIMIT)
        self.assertEqual(categorize_failure(RuntimeError("HTTP 401 Unauthorized token")), FailureReason.AUTHENTICATION)
        self.assertEqual(categorize_failure(RuntimeError("Model 'gemini-pro' does not exist")), FailureReason.MODEL_UNAVAILABLE)
        self.assertEqual(categorize_failure(RuntimeError("Network unreachable")), FailureReason.CONNECTION_ERROR)
        self.assertEqual(categorize_failure(RuntimeError("Completely unexpected internal error")), FailureReason.UNKNOWN)

    def test_no_paid_api_keys_or_real_network_requests(self):
        """Verify all operations executed with mocks and zero real API key access."""
        worker = RoutedChatWorker(
            decision=self.decision,
            prompt="Offline test",
            history=None,
            provider_manager=self.provider_mgr,
        )
        worker.start()
        worker.wait(5000)
        app.processEvents()

        # Mock credential store should have 0 secret gets
        self.mock_cred_store.get_api_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()

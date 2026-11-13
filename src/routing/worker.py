import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from src.core.events import StreamChunk
from src.core.message import Message, MessageRole
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
from src.routing.types import FailureReason, RoutingCandidate, RoutingDecision

logger = logging.getLogger("RelayAI.RoutedChatWorker")


def categorize_failure(exc: Exception) -> FailureReason:
    """Categorize an exception into a FailureReason enum."""
    if isinstance(exc, ProviderRateLimitError):
        return FailureReason.RATE_LIMIT
    if isinstance(exc, ProviderAuthenticationError):
        return FailureReason.AUTHENTICATION
    if isinstance(exc, (ProviderTimeoutError, TimeoutError)):
        return FailureReason.TIMEOUT
    if isinstance(exc, ModelNotFoundError):
        return FailureReason.MODEL_UNAVAILABLE
    if isinstance(exc, (ProviderConnectionError, ConnectionError)):
        return FailureReason.CONNECTION_ERROR

    # Inspect message content and exception class name for unwrapped or vendor-specific errors
    msg = (str(exc) or "").lower()
    exc_type = type(exc).__name__.lower()

    if any(k in msg or k in exc_type for k in ["rate limit", "quota", "429", "too many requests", "resource_exhausted"]):
        return FailureReason.RATE_LIMIT
    if any(k in msg or k in exc_type for k in ["auth", "api key", "unauthorized", "forbidden", "401", "403", "credential"]):
        return FailureReason.AUTHENTICATION
    if any(k in msg or k in exc_type for k in ["timeout", "timed out", "deadline"]):
        return FailureReason.TIMEOUT
    if any(k in msg or k in exc_type for k in ["model", "not found", "404", "does not exist", "unsupported model"]):
        return FailureReason.MODEL_UNAVAILABLE
    if any(k in msg or k in exc_type for k in ["connection", "connect", "unreachable", "refused", "network", "dns", "socket"]):
        return FailureReason.CONNECTION_ERROR

    return FailureReason.UNKNOWN


@dataclass
class RoutedAttemptRecord:
    """Audit record of a single model candidate attempt."""
    provider_id: str
    model_id: str
    attempt_index: int
    is_success: bool = False
    is_cancelled: bool = False
    failure_reason: Optional[FailureReason] = None
    error_message: Optional[str] = None
    streamed_chars: int = 0
    elapsed_sec: float = 0.0


class RoutedChatWorker(QThread):
    """Background worker executing a RoutingDecision with automatic fallback and streaming.

    DESIGN PRINCIPLES:
    - Thread-safety: All provider and network I/O executes off the Qt GUI thread in QThread.run().
    - Clean Fallback: If an attempt fails after partial streaming, the partial output is discarded,
      the fallback signal is emitted, and the next model starts afresh with identical context.
    - Context Safety: Input conversation history is never mutated. User prompt is appended once,
      and only the final successful assistant response is committed.
    - Cooperative Cancellation: Supports responsive cancellation during streaming and between attempts.
    """

    # Signals for UI and lifecycle tracking
    attempt_started = Signal(str, str, int)  # (provider_id, model_id, attempt_index)
    chunk_received = Signal(str)  # (delta: str)
    stream_chunk = Signal(object)  # (chunk: StreamChunk)
    attempt_failed = Signal(str, str, str, str)  # (provider_id, model_id, failure_reason, error_message)
    fallback_started = Signal(str, str, str, str, int)  # (from_prov, from_model, to_prov, to_model, fallback_idx)
    final_success = Signal(str, str, str, list)  # (provider_id, model_id, response_text, messages_to_commit)
    final_failure = Signal(str, list)  # (summary_message, attempts_records)
    cancelled = Signal()  # ()

    def __init__(
        self,
        decision: RoutingDecision,
        prompt: str,
        history: Optional[List[Message]],
        provider_manager: ProviderManager,
        timeout_sec: float = 60.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.decision = decision
        self.prompt = prompt or ""
        self.history = history
        self.provider_manager = provider_manager
        self.timeout_sec = timeout_sec
        self._is_cancelled = False

    def cancel(self, wait_ms: int = 0) -> None:
        """Cooperatively request cancellation and optionally wait for termination."""
        self._is_cancelled = True
        if wait_ms > 0 and self.isRunning():
            self.wait(wait_ms)

    def is_cancelled(self) -> bool:
        """Check whether cancellation was requested."""
        return self._is_cancelled

    def run(self) -> None:
        """Execute the primary model followed by up to two fallback candidates."""
        if not self.decision or not self.decision.primary:
            self.final_failure.emit("Invalid or missing RoutingDecision.", [])
            return

        # Build ordered list of candidates: Primary first, followed by at most 2 distinct fallbacks
        candidates: List[RoutingCandidate] = [self.decision.primary]
        seen = {(self.decision.primary.provider_id.lower(), self.decision.primary.model_id.lower())}
        for fb in (self.decision.fallback_chain or []):
            key = (fb.provider_id.lower(), fb.model_id.lower())
            if key not in seen:
                seen.add(key)
                candidates.append(fb)
            if len(candidates) >= 3:
                break

        attempts_records: List[RoutedAttemptRecord] = []

        # Prepare messages payload: Deep copy history and append user prompt ONCE per attempt
        # Do not mutate the supplied history list or message objects
        base_history = list(self.history or [])
        clean_history = [
            Message(role=m.role, content=m.content, timestamp=m.timestamp, metadata=dict(m.metadata or {}))
            for m in base_history
        ]
        attempt_messages = clean_history + [Message(role=MessageRole.USER, content=self.prompt)]

        for attempt_idx, candidate in enumerate(candidates):
            # Check cooperative cancellation before starting attempt
            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.attempt_started.emit(candidate.provider_id, candidate.model_id, attempt_idx)
            attempt_start_time = time.perf_counter()
            partial_parts: List[str] = []

            try:
                provider = self.provider_manager.get_provider(candidate.provider_id)
                if not provider:
                    raise ModelNotFoundError(
                        f"Provider '{candidate.provider_id}' is not registered or unavailable.",
                        provider_id=candidate.provider_id,
                    )

                stream_iter = provider.stream(attempt_messages, candidate.model_id)

                for chunk in stream_iter:
                    # Check cooperative cancellation during streaming
                    if self._is_cancelled:
                        self.cancelled.emit()
                        return

                    # Enforce per-attempt timeout
                    elapsed = time.perf_counter() - attempt_start_time
                    if elapsed > self.timeout_sec:
                        raise ProviderTimeoutError(
                            f"Model '{candidate.model_id}' exceeded timeout limit of {self.timeout_sec:.1f}s.",
                            provider_id=candidate.provider_id,
                        )

                    # Extract chunk content
                    delta = getattr(chunk, "delta", chunk if isinstance(chunk, str) else "")
                    if delta:
                        partial_parts.append(delta)
                        self.chunk_received.emit(delta)

                    if isinstance(chunk, StreamChunk):
                        self.stream_chunk.emit(chunk)

                # Attempt completed successfully!
                elapsed = time.perf_counter() - attempt_start_time
                full_text = "".join(partial_parts)

                record = RoutedAttemptRecord(
                    provider_id=candidate.provider_id,
                    model_id=candidate.model_id,
                    attempt_index=attempt_idx,
                    is_success=True,
                    is_cancelled=False,
                    streamed_chars=len(full_text),
                    elapsed_sec=elapsed,
                )
                attempts_records.append(record)

                # Commit only the final successful user prompt and assistant response
                messages_to_commit = [
                    Message(role=MessageRole.USER, content=self.prompt),
                    Message(role=MessageRole.ASSISTANT, content=full_text),
                ]

                self.final_success.emit(
                    candidate.provider_id,
                    candidate.model_id,
                    full_text,
                    messages_to_commit,
                )
                return

            except Exception as exc:
                elapsed = time.perf_counter() - attempt_start_time
                reason = categorize_failure(exc)
                had_partial_stream = len(partial_parts) > 0
                chars_discarded = sum(len(p) for p in partial_parts)

                # Requirement 7: Discard partial output immediately; never combine across models
                partial_parts.clear()

                record = RoutedAttemptRecord(
                    provider_id=candidate.provider_id,
                    model_id=candidate.model_id,
                    attempt_index=attempt_idx,
                    is_success=False,
                    is_cancelled=False,
                    failure_reason=reason,
                    error_message=str(exc),
                    streamed_chars=chars_discarded,
                    elapsed_sec=elapsed,
                )
                attempts_records.append(record)

                discard_info = f" (discarded {chars_discarded} partial characters)" if had_partial_stream else ""
                logger.warning(
                    "Attempt %d failed for [%s/%s]: %s%s. Reason: %s",
                    attempt_idx + 1,
                    candidate.provider_id,
                    candidate.model_id,
                    exc,
                    discard_info,
                    reason.value,
                )

                self.attempt_failed.emit(
                    candidate.provider_id,
                    candidate.model_id,
                    reason.value,
                    str(exc),
                )

                # Check cooperative cancellation before initiating fallback
                if self._is_cancelled:
                    self.cancelled.emit()
                    return

                # If another candidate exists in fallback chain, emit fallback_started
                if attempt_idx + 1 < len(candidates):
                    next_cand = candidates[attempt_idx + 1]
                    fallback_idx = attempt_idx + 1
                    self.fallback_started.emit(
                        candidate.provider_id,
                        candidate.model_id,
                        next_cand.provider_id,
                        next_cand.model_id,
                        fallback_idx,
                    )
                else:
                    # All candidate attempts exhausted
                    summary = (
                        f"All {len(candidates)} routing candidate attempts failed. "
                        f"Last failure on [{candidate.provider_id}/{candidate.model_id}]: {exc}"
                    )
                    self.final_failure.emit(summary, attempts_records)
                    return


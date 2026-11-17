import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QObject, QThread, Signal

from src.core.events import ResponseMetadata
from src.core.message import Message
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager

logger = logging.getLogger("RelayAI.Comparison")


@dataclass
class ComparisonSlotConfig:
    """Configuration for an individual model slot in comparison mode."""
    slot_id: int
    provider_id: str
    model: str
    timeout_sec: float = 60.0


@dataclass
class ComparisonMetrics:
    """Execution metrics collected during generation."""
    elapsed_sec: float = 0.0
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    is_token_count_estimated: bool = False

    def formatted_elapsed(self) -> str:
        return f"{self.elapsed_sec:.2f}s"

    def formatted_tokens(self) -> str:
        if self.total_tokens is not None:
            est = " (~est)" if self.is_token_count_estimated else ""
            return f"{self.total_tokens:,} tokens{est}"
        return "N/A"

    def formatted_speed(self) -> str:
        if self.tokens_per_sec is not None:
            return f"{self.tokens_per_sec:.1f} t/s"
        return "N/A"

    def formatted_cost(self, is_local: bool = False) -> str:
        if is_local:
            return "$0.00 (Local)"
        if self.estimated_cost_usd is not None:
            if self.estimated_cost_usd < 0.0001:
                return "<$0.0001"
            return f"${self.estimated_cost_usd:.4f}"
        return "N/A"


@dataclass
class ComparisonSlotResult:
    """Final result container for a single comparison slot."""
    slot_id: int
    provider_id: str
    model: str
    content: str = ""
    is_success: bool = False
    is_canceled: bool = False
    error_message: Optional[str] = None
    metrics: ComparisonMetrics = field(default_factory=ComparisonMetrics)
    metadata: Optional[ResponseMetadata] = None


def estimate_api_cost(
    provider_id: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    """Calculate approximate USD API cost based on official provider pricing schedules."""
    pid = (provider_id or "").lower()
    m = (model or "").lower()

    if pid == "ollama":
        return 0.0

    p_tokens = prompt_tokens or 0
    c_tokens = completion_tokens or 0
    if p_tokens == 0 and c_tokens == 0:
        return None

    # Pricing per 1,000,000 tokens (USD)
    pricing = {
        # OpenAI
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1": (2.50, 10.00),
        "o3-mini": (1.10, 4.40),
        "o3": (2.50, 10.00),
        # Gemini
        "gemini-3.5-flash": (0.075, 0.30),
        "gemini-2.5-flash": (0.075, 0.30),
        "gemini-3.8-flash": (0.15, 0.60),
        # Groq
        "openai/gpt-oss-120b": (0.15, 0.15),
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "qwen/qwen3.8-27b": (0.20, 0.20),
    }

    rates = None
    for k, v in pricing.items():
        if k in m:
            rates = v
            break

    if not rates:
        # Generic cloud fallback estimate ($0.15 / $0.60 per 1M)
        rates = (0.15, 0.60)

    prompt_cost = (p_tokens / 1_000_000) * rates[0]
    completion_cost = (c_tokens / 1_000_000) * rates[1]
    return prompt_cost + completion_cost


class ModelCompareWorker(QThread):
    """Worker thread dedicated to streaming a single model's response within a comparison session."""

    chunk_received = Signal(int, str)  # (slot_id, delta)
    status_changed = Signal(int, str)  # (slot_id, status_text)
    slot_finished = Signal(int, object)  # (slot_id, ComparisonSlotResult)
    slot_failed = Signal(int, str, object)  # (slot_id, error_message, ComparisonSlotResult)

    def __init__(
        self,
        slot_id: int,
        provider: AIProvider,
        messages: List[Message],
        model: str,
        timeout_sec: float = 60.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.slot_id = slot_id
        self.provider = provider
        self.messages = messages
        self.model = model
        self.timeout_sec = timeout_sec
        self._abort_requested = False

    def abort(self) -> None:
        """Cooperatively request worker termination."""
        self._abort_requested = True

    def run(self) -> None:
        parts: List[str] = []
        start_time = time.perf_counter()
        last_metadata: Optional[ResponseMetadata] = None
        actual_provider_id = self.provider.provider_id
        actual_model = self.model
        self.status_changed.emit(self.slot_id, "Connecting…")

        try:
            stream_iter = self.provider.stream(self.messages, self.model)
            self.status_changed.emit(self.slot_id, "Streaming…")

            for chunk in stream_iter:
                # Cooperative cancellation check
                if self._abort_requested:
                    elapsed = time.perf_counter() - start_time
                    result = ComparisonSlotResult(
                        slot_id=self.slot_id,
                        provider_id=self.provider.provider_id,
                        model=self.model,
                        content="".join(parts),
                        is_success=False,
                        is_canceled=True,
                        error_message="Comparison canceled by user.",
                        metrics=ComparisonMetrics(elapsed_sec=elapsed),
                    )
                    self.slot_finished.emit(self.slot_id, result)
                    return

                # Check configurable timeout
                elapsed_now = time.perf_counter() - start_time
                if elapsed_now > self.timeout_sec:
                    raise TimeoutError(
                        f"Generation timed out after {self.timeout_sec:.1f}s for model {self.model}"
                    )

                if chunk.metadata:
                    last_metadata = chunk.metadata
                    if chunk.metadata.extra.get("automatic_fallback_from"):
                        parts.clear()
                        actual_provider_id = chunk.metadata.provider_id
                        actual_model = chunk.metadata.model

                if chunk.delta:
                    parts.append(chunk.delta)
                    self.chunk_received.emit(self.slot_id, chunk.delta)

            total_elapsed = time.perf_counter() - start_time
            full_text = "".join(parts)

            # Calculate token metrics
            prompt_tokens = getattr(last_metadata, "prompt_tokens", None) if last_metadata else None
            completion_tokens = getattr(last_metadata, "completion_tokens", None) if last_metadata else None
            total_tokens = getattr(last_metadata, "total_tokens", None) if last_metadata else None
            is_estimated = False

            if total_tokens is None and completion_tokens is None:
                # Approximate tokens if provider didn't emit metadata
                estimated_completion = max(1, len(full_text) // 4)
                estimated_prompt = sum(max(1, len(m.content) // 4) for m in self.messages)
                completion_tokens = estimated_completion
                prompt_tokens = estimated_prompt
                total_tokens = estimated_prompt + estimated_completion
                is_estimated = True

            tps = (completion_tokens / total_elapsed) if (completion_tokens and total_elapsed > 0) else None
            cost = estimate_api_cost(
                actual_provider_id,
                actual_model,
                prompt_tokens,
                completion_tokens,
            )

            metrics = ComparisonMetrics(
                elapsed_sec=total_elapsed,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_sec=tps,
                estimated_cost_usd=cost,
                is_token_count_estimated=is_estimated,
            )

            result = ComparisonSlotResult(
                slot_id=self.slot_id,
                provider_id=actual_provider_id,
                model=actual_model,
                content=full_text,
                is_success=True,
                is_canceled=False,
                error_message=None,
                metrics=metrics,
                metadata=last_metadata,
            )
            self.status_changed.emit(self.slot_id, "Completed")
            self.slot_finished.emit(self.slot_id, result)

        except Exception as exc:
            total_elapsed = time.perf_counter() - start_time
            err_msg = str(exc)
            result = ComparisonSlotResult(
                slot_id=self.slot_id,
                provider_id=self.provider.provider_id,
                model=self.model,
                content="".join(parts),
                is_success=False,
                is_canceled=False,
                error_message=err_msg,
                metrics=ComparisonMetrics(elapsed_sec=total_elapsed),
                metadata=last_metadata,
            )
            self.status_changed.emit(self.slot_id, "Error")
            self.slot_failed.emit(self.slot_id, err_msg, result)


class ComparisonCoordinator(QObject):
    """Coordinates parallel model workers, aggregates metrics, and handles cancellation."""

    slot_chunk = Signal(int, str)  # (slot_id, delta)
    slot_status = Signal(int, str)  # (slot_id, status_text)
    slot_completed = Signal(int, object)  # (slot_id, ComparisonSlotResult)
    all_completed = Signal(list)  # (List[ComparisonSlotResult])
    canceled = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.workers: Dict[int, ModelCompareWorker] = {}
        self.results: Dict[int, ComparisonSlotResult] = {}
        self._expected_count: int = 0
        self._is_active: bool = False

    def is_running(self) -> bool:
        """Check whether any worker is currently running."""
        return self._is_active and any(w.isRunning() for w in self.workers.values())

    def start_comparison(
        self,
        slots: List[ComparisonSlotConfig],
        messages: List[Message],
        provider_manager: ProviderManager,
    ) -> bool:
        """Start parallel execution for 2 to 4 model slots with identical message context."""
        if self.is_running():
            logger.warning("Comparison already in progress; cancelling previous run.")
            self.cancel_all()

        self.workers.clear()
        self.results.clear()
        self._expected_count = len(slots)
        self._is_active = True

        for cfg in slots:
            provider = provider_manager.get_provider(cfg.provider_id)
            if not provider:
                err_res = ComparisonSlotResult(
                    slot_id=cfg.slot_id,
                    provider_id=cfg.provider_id,
                    model=cfg.model,
                    is_success=False,
                    error_message=f"Provider '{cfg.provider_id}' is not registered or unavailable.",
                )
                self.results[cfg.slot_id] = err_res
                self.slot_completed.emit(cfg.slot_id, err_res)
                continue

            worker = ModelCompareWorker(
                slot_id=cfg.slot_id,
                provider=provider,
                messages=list(messages),  # identical copy
                model=cfg.model,
                timeout_sec=cfg.timeout_sec,
            )
            worker.chunk_received.connect(self._on_chunk)
            worker.status_changed.connect(self._on_status)
            worker.slot_finished.connect(self._on_slot_finished)
            worker.slot_failed.connect(self._on_slot_failed)

            self.workers[cfg.slot_id] = worker
            worker.start()

        self._check_if_all_done()
        return True

    def _on_chunk(self, slot_id: int, delta: str) -> None:
        self.slot_chunk.emit(slot_id, delta)

    def _on_status(self, slot_id: int, status: str) -> None:
        self.slot_status.emit(slot_id, status)

    def _on_slot_finished(self, slot_id: int, result: ComparisonSlotResult) -> None:
        self.results[slot_id] = result
        self.slot_completed.emit(slot_id, result)
        self._check_if_all_done()

    def _on_slot_failed(self, slot_id: int, error_msg: str, result: ComparisonSlotResult) -> None:
        self.results[slot_id] = result
        self.slot_completed.emit(slot_id, result)
        self._check_if_all_done()

    def _check_if_all_done(self) -> None:
        if self._is_active and len(self.results) >= self._expected_count:
            self._is_active = False
            # Order results by slot_id
            ordered_results = [self.results[k] for k in sorted(self.results.keys())]
            self.all_completed.emit(ordered_results)

    def cancel_all(self, wait_ms: int = 1000) -> None:
        """Cooperatively request all workers to stop and wait safely."""
        self._is_active = False
        for worker in self.workers.values():
            if worker.isRunning():
                worker.abort()

        # Wait cooperatively for threads to exit
        if wait_ms > 0:
            for worker in self.workers.values():
                if worker.isRunning():
                    worker.wait(wait_ms)

        self.canceled.emit()

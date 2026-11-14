from abc import ABC, abstractmethod
import logging
import threading
import time
from typing import List, Optional, Union

from src.agents.context import TaskContext
from src.agents.exceptions import (
    AgentCancelledError,
    AgentConfigurationError,
    AgentCoreError,
    AgentExecutionError,
    AgentTimeoutError,
)
from src.agents.models import (
    AgentConfig,
    AgentExecutionMetadata,
    AgentResult,
    AgentRole,
    AgentStatus,
)
from src.core.events import ResponseMetadata
from src.core.message import Message
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.providers.manager import ProviderManager, get_default_provider_manager

logger = logging.getLogger("RelayAI.AgentCore")


class AgentBase(ABC):
    """Abstract base class representing an intelligent specialized agent.

    Encapsulates role identity, execution lifecycle states, thread-safe cancellation,
    timeout enforcement, and standardized provider dispatch.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        config: AgentConfig,
        description: str = "",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if not agent_id or not isinstance(agent_id, str):
            raise AgentConfigurationError("agent_id must be a non-empty string.")
        if not name or not isinstance(name, str):
            raise AgentConfigurationError("name must be a non-empty string.")
        if not isinstance(role, AgentRole):
            raise AgentConfigurationError(f"role must be an AgentRole enum, got {type(role)}.")
        if not isinstance(config, AgentConfig):
            raise AgentConfigurationError(f"config must be an AgentConfig instance, got {type(config)}.")

        self.agent_id = agent_id.strip()
        self.name = name.strip()
        self.role = role
        self.description = description.strip()
        self.config = config
        self.provider_manager = provider_manager or get_default_provider_manager()

        self.status: AgentStatus = AgentStatus.IDLE
        self._cancel_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._active_task_id: Optional[str] = None

    def validate_config(self) -> None:
        """Validate agent configuration and verify that the assigned provider exists."""
        if not self.config.model_id:
            raise AgentConfigurationError(
                f"Agent '{self.agent_id}' has an empty model_id in its configuration.",
                agent_id=self.agent_id,
            )
        provider = self.provider_manager.get_provider(self.config.provider_id)
        if not provider:
            raise AgentConfigurationError(
                f"Agent '{self.agent_id}' references provider '{self.config.provider_id}', which is not registered.",
                agent_id=self.agent_id,
            )

    def cancel(self) -> None:
        """Thread-safely request cooperative cancellation of the active execution turn."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def reset(self) -> None:
        """Reset the agent back to IDLE state. Rejected if currently RUNNING."""
        with self._lifecycle_lock:
            if self.status == AgentStatus.RUNNING:
                raise AgentExecutionError(
                    f"Cannot reset agent '{self.agent_id}' while in RUNNING state.",
                    agent_id=self.agent_id,
                )
            self.status = AgentStatus.IDLE
            self._cancel_event.clear()
            self._active_task_id = None

    def execute(self, context: TaskContext) -> AgentResult:
        """Public execution entry point with strict lifecycle validation, timeout enforcement,
        and thread-safe cancellation."""
        # 1. State transition check: Prevent concurrent execution on the same agent instance
        with self._lifecycle_lock:
            if self.status == AgentStatus.RUNNING:
                raise AgentExecutionError(
                    f"Agent '{self.agent_id}' is already RUNNING on task '{self._active_task_id}'. "
                    f"Concurrent execution on the same instance is prohibited.",
                    agent_id=self.agent_id,
                )
            is_pre_cancelled = self._cancel_event.is_set() or context.is_cancelled()
            self.status = AgentStatus.RUNNING
            self._active_task_id = context.task_id

        started_at = time.time()
        metadata = AgentExecutionMetadata(
            started_at=started_at,
            provider_id=self.config.provider_id,
            model_id=self.config.model_id,
        )

        try:
            if is_pre_cancelled:
                return self._finalize_result(
                    AgentStatus.CANCELLED,
                    output="",
                    error="Execution cancelled before invocation.",
                    context=context,
                    metadata=metadata,
                )

            # 2. Pre-execution checks
            self.validate_config()

            if self._is_turn_cancelled(context):
                return self._finalize_result(
                    AgentStatus.CANCELLED,
                    output="",
                    error="Execution cancelled before invocation.",
                    context=context,
                    metadata=metadata,
                )

            # 3. Controlled execution with bounded retry policy
            output, resp_meta = self._execute_with_retries(context, metadata)

            if self._is_turn_cancelled(context):
                return self._finalize_result(
                    AgentStatus.CANCELLED,
                    output="",
                    error="Execution cancelled during invocation.",
                    context=context,
                    metadata=metadata,
                )

            # 4. Extract verified token counts if provided by ResponseMetadata
            if resp_meta:
                metadata.prompt_tokens = resp_meta.prompt_tokens
                metadata.completion_tokens = resp_meta.completion_tokens
                metadata.total_tokens = resp_meta.total_tokens
                if hasattr(resp_meta, "critic_review"):
                    metadata.critic_review = getattr(resp_meta, "critic_review")
                if hasattr(resp_meta, "extra") and resp_meta.extra:
                    metadata.extra.update(resp_meta.extra)

            return self._finalize_result(
                AgentStatus.COMPLETED,
                output=output,
                error=None,
                context=context,
                metadata=metadata,
            )

        except AgentCancelledError:
            return self._finalize_result(
                AgentStatus.CANCELLED,
                output="",
                error="Execution cancelled by user request.",
                context=context,
                metadata=metadata,
            )

        except (ProviderTimeoutError, AgentTimeoutError) as exc:
            logger.warning("Agent '%s' timed out: %s", self.agent_id, exc)
            return self._finalize_result(
                AgentStatus.FAILED,
                output="",
                error=f"Timeout after {time.time() - started_at:.2f}s: {exc}",
                context=context,
                metadata=metadata,
            )

        except (AgentConfigurationError, ProviderRateLimitError, ProviderAuthenticationError, ProviderConnectionError, ModelNotFoundError, ProviderError) as exc:
            logger.error("Agent '%s' execution configuration/provider failure: %s", self.agent_id, exc)
            return self._finalize_result(
                AgentStatus.FAILED,
                output="",
                error=str(exc),
                context=context,
                metadata=metadata,
            )

        except Exception as exc:
            logger.exception("Agent '%s' unhandled execution exception: %s", self.agent_id, exc)
            return self._finalize_result(
                AgentStatus.FAILED,
                output="",
                error=f"Unexpected error: {exc}",
                context=context,
                metadata=metadata,
            )

    def _execute_with_retries(
        self, context: TaskContext, metadata: AgentExecutionMetadata
    ) -> tuple[str, Optional[ResponseMetadata]]:
        """Dispatch task to provider, retrying on transient failures up to max_retries."""
        attempts = 0
        max_attempts = 1 + self.config.max_retries
        last_exception: Optional[Exception] = None

        while attempts < max_attempts:
            if self._is_turn_cancelled(context):
                raise AgentCancelledError("Cancelled before attempt execution.", agent_id=self.agent_id)

            attempts += 1
            metadata.attempt_count = attempts

            try:
                return self._run(context)
            except (ProviderConnectionError, ProviderTimeoutError) as exc:
                last_exception = exc
                if attempts < max_attempts and not self._is_turn_cancelled(context):
                    logger.info("Agent '%s' attempt %d failed (%s). Retrying...", self.agent_id, attempts, exc)
                    time.sleep(0.5)
                    continue
                raise
            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise AgentExecutionError("Execution failed without explicit exception.", agent_id=self.agent_id)

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        """Default provider invocation logic. Builds messages and streams/generates response."""
        provider = self.provider_manager.get_provider(self.config.provider_id)
        if not provider:
            raise AgentConfigurationError(
                f"Provider '{self.config.provider_id}' is not registered.",
                agent_id=self.agent_id,
            )

        messages = context.build_messages(system_instruction=self.config.system_instruction)
        deadline = time.time() + self.config.timeout_sec

        # Prefer streaming execution for fine-grained cancellation checking
        if provider.supports_streaming:
            collected_chunks: List[str] = []
            final_meta: Optional[ResponseMetadata] = None

            kwargs = dict(self.config.extra_params)
            kwargs["timeout"] = self.config.timeout_sec

            for chunk in provider.stream(messages, self.config.model_id, **kwargs):
                if self._is_turn_cancelled(context):
                    raise AgentCancelledError("Cancelled during streaming.", agent_id=self.agent_id)

                if time.time() > deadline:
                    raise AgentTimeoutError(
                        f"Execution exceeded configured timeout of {self.config.timeout_sec}s.",
                        agent_id=self.agent_id,
                    )

                text = chunk.delta if hasattr(chunk, "delta") else str(chunk)
                if text:
                    collected_chunks.append(text)

                if hasattr(chunk, "metadata") and chunk.metadata:
                    final_meta = chunk.metadata

            return "".join(collected_chunks).strip(), final_meta

        # Fallback to synchronous generation
        kwargs = dict(self.config.extra_params)
        kwargs["timeout"] = self.config.timeout_sec
        output = provider.generate(messages, self.config.model_id, **kwargs)
        return output.strip(), None

    def _is_turn_cancelled(self, context: TaskContext) -> bool:
        """Inspect both internal cancel event and external context cancel probes."""
        return self._cancel_event.is_set() or context.is_cancelled()

    def _finalize_result(
        self,
        final_status: AgentStatus,
        output: str,
        error: Optional[str],
        context: TaskContext,
        metadata: AgentExecutionMetadata,
    ) -> AgentResult:
        """Record timing, update lifecycle state under lock, and construct AgentResult."""
        finished_at = time.time()
        metadata.finished_at = finished_at
        metadata.duration_sec = max(0.0, finished_at - metadata.started_at)

        with self._lifecycle_lock:
            self.status = final_status
            self._active_task_id = None

        return AgentResult(
            agent_id=self.agent_id,
            agent_role=self.role,
            task_id=context.task_id,
            status=final_status,
            output=output,
            error=error,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"<AgentBase id={self.agent_id!r} role={self.role.value!r} status={self.status.value!r}>"

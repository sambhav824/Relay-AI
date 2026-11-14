from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentRole, AgentStatus
from src.agents.registry import AgentRegistry
from src.workflow.context_pipeline import ContextTransferPipeline
from src.workflow.exceptions import (
    BudgetExceededError,
    InvalidRevisionTargetError,
    WorkflowCancelledError,
    WorkflowConfigurationError,
    WorkflowExecutionError,
    WorkflowTimeoutError,
)
from src.workflow.models import (
    CriticReviewResult,
    StepExecutionRecord,
    StepFailurePolicy,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStepConfig,
)

logger = logging.getLogger("RelayAI.WorkflowEngine")


class WorkflowEngine:
    """Coordinates and executes multi-agent sequential and revision workflows.

    Features:
    - Thread-safe state machine and concurrency prevention.
    - Strict context isolation and structured output handoff.
    - Cooperative cancellation and between-step pause/resume.
    - Wall-clock timeout monitoring and token budget enforcement.
    - Bounded revision loops with structured CriticReviewResult evaluation.
    - Configurable failure handling policies (ABORT, RETRY, SKIP, FALLBACK).
    """

    def __init__(
        self,
        config: WorkflowConfig,
        registry: AgentRegistry,
        on_step_start: Optional[Any] = None,
        on_step_finish: Optional[Any] = None,
        on_revision: Optional[Any] = None,
    ):
        self.config = config
        self.registry = registry
        self.on_step_start = on_step_start
        self.on_step_finish = on_step_finish
        self.on_revision = on_revision

        self._status: WorkflowStatus = WorkflowStatus.PENDING
        self._lifecycle_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

        self._active_agent: Optional[AgentBase] = None
        self._step_records: List[StepExecutionRecord] = []
        self._current_task_id: Optional[str] = None

    def get_status(self) -> WorkflowStatus:
        """Return current workflow lifecycle status."""
        return self._status

    def cancel(self) -> None:
        """Thread-safely request cooperative cancellation across the active workflow and agent."""
        self._cancel_event.set()
        self._pause_event.clear()  # Unblock paused thread if waiting
        if self._active_agent is not None:
            try:
                self._active_agent.cancel()
            except Exception:
                pass

    def pause(self) -> None:
        """Signal workflow to pause execution safely between steps."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume execution of a paused workflow."""
        self._pause_event.clear()

    def is_paused(self) -> bool:
        """Check if workflow is currently paused or pending pause between steps."""
        return self._pause_event.is_set()

    def execute(self, base_context: TaskContext) -> WorkflowResult:
        """Execute the configured workflow steps sequentially with context accumulation and safety limits.

        Raises:
            WorkflowExecutionError: If workflow is already RUNNING.
            WorkflowConfigurationError: If step validation fails.
        """
        # 1. State transition check: Prevent concurrent execution on same instance
        with self._lifecycle_lock:
            if self._status == WorkflowStatus.RUNNING:
                raise WorkflowExecutionError(
                    f"Workflow '{self.config.workflow_id}' is already RUNNING on task '{self._current_task_id}'. "
                    f"Concurrent execution on the same WorkflowEngine instance is prohibited.",
                    workflow_id=self.config.workflow_id,
                )
            self._status = WorkflowStatus.RUNNING
            self._cancel_event.clear()
            self._pause_event.clear()
            self._step_records = []
            self._current_task_id = base_context.task_id

        started_at = time.time()
        deadline = started_at + self.config.timeout_sec
        pipeline = ContextTransferPipeline(base_context)

        # 2. Pre-flight dependency and configuration validation
        self._validate_workflow_structure()

        step_map: Dict[str, WorkflowStepConfig] = {s.step_id: s for s in self.config.steps}
        completed_step_ids: Set[str] = set()
        failed_step_ids: List[str] = []
        skipped_step_ids: List[str] = []
        revision_counts: Dict[str, int] = {}
        total_step_executions = 0

        # Build initial step execution queue
        execution_queue: List[str] = [s.step_id for s in self.config.steps]
        current_idx = 0

        try:
            while current_idx < len(execution_queue):
                # Check cancellation
                if self._cancel_event.is_set():
                    return self._finalize_workflow(
                        WorkflowStatus.CANCELLED,
                        pipeline,
                        started_at,
                        failed_step_ids,
                        skipped_step_ids,
                        error="Workflow execution cancelled by user request.",
                    )

                # Check pause between steps
                if self._pause_event.is_set():
                    with self._lifecycle_lock:
                        self._status = WorkflowStatus.PAUSED
                    logger.info("Workflow '%s' paused between steps.", self.config.workflow_id)
                    while self._pause_event.is_set() and not self._cancel_event.is_set():
                        time.sleep(0.05)

                    if self._cancel_event.is_set():
                        return self._finalize_workflow(
                            WorkflowStatus.CANCELLED,
                            pipeline,
                            started_at,
                            failed_step_ids,
                            skipped_step_ids,
                            error="Workflow cancelled while paused.",
                        )
                    with self._lifecycle_lock:
                        self._status = WorkflowStatus.RUNNING

                # Check wall-clock timeout
                if time.time() > deadline:
                    raise WorkflowTimeoutError(
                        f"Workflow exceeded global timeout limit of {self.config.timeout_sec}s.",
                        workflow_id=self.config.workflow_id,
                    )

                # Check global execution ceiling
                if total_step_executions >= self.config.max_steps:
                    logger.warning(
                        "Workflow reached maximum execution ceiling of %d steps. Halting pipeline.",
                        self.config.max_steps,
                    )
                    break

                step_id = execution_queue[current_idx]
                step_cfg = step_map[step_id]

                # Check if independent step is already completed and was NOT invalidated by revision
                if step_id in completed_step_ids:
                    logger.info("Step '%s' is independent and already completed; skipping redundant re-execution.", step_id)
                    current_idx += 1
                    continue

                # Check step dependencies
                missing_deps = [dep for dep in step_cfg.dependencies if dep not in completed_step_ids]
                if missing_deps:
                    if step_cfg.is_optional:
                        logger.info("Skipping optional step '%s' due to missing dependencies: %s", step_id, missing_deps)
                        skipped_step_ids.append(step_id)
                        current_idx += 1
                        continue
                    else:
                        raise WorkflowExecutionError(
                            f"Step '{step_id}' missing required dependencies: {missing_deps}",
                            workflow_id=self.config.workflow_id,
                        )

                # Check token budget before starting step
                self._check_token_budget()

                # Execute step
                total_step_executions += 1
                revision_iter = revision_counts.get(step_id, 0)

                if self.on_step_start:
                    try:
                        agent_obj = self.registry.get(step_cfg.agent_id)
                        agent_role = agent_obj.role if agent_obj else AgentRole.CUSTOM
                        self.on_step_start(step_id, step_cfg.agent_id, agent_role, revision_iter)
                    except Exception as exc:
                        logger.error("Error in on_step_start callback: %s", exc)

                step_record = self._execute_step(
                    step_cfg=step_cfg,
                    pipeline=pipeline,
                    revision_iter=revision_iter,
                )
                self._step_records.append(step_record)

                if self.on_step_finish:
                    try:
                        self.on_step_finish(step_record)
                    except Exception as exc:
                        logger.error("Error in on_step_finish callback: %s", exc)

                # Check wall-clock timeout and token budget post-step
                if time.time() > deadline:
                    raise WorkflowTimeoutError(
                        f"Workflow exceeded global timeout limit of {self.config.timeout_sec}s.",
                        workflow_id=self.config.workflow_id,
                    )
                self._check_token_budget()

                if step_record.status == AgentStatus.CANCELLED:
                    return self._finalize_workflow(
                        WorkflowStatus.CANCELLED,
                        pipeline,
                        started_at,
                        failed_step_ids,
                        skipped_step_ids,
                        error="Step execution cancelled.",
                    )

                if step_record.status == AgentStatus.FAILED:
                    handled = self._handle_step_failure(
                        step_cfg,
                        step_record,
                        failed_step_ids,
                        skipped_step_ids,
                        completed_step_ids,
                        pipeline,
                    )
                    if not handled:
                        return self._finalize_workflow(
                            WorkflowStatus.FAILED,
                            pipeline,
                            started_at,
                            failed_step_ids,
                            skipped_step_ids,
                            error=f"Step '{step_id}' failed: {step_record.error}",
                        )
                    current_idx += 1
                    continue

                # Step Succeeded
                completed_step_ids.add(step_id)

                # Evaluate structured Critic review if applicable
                if step_cfg.is_critic_step and step_record.critic_review:
                    review = step_record.critic_review
                    if review.revision_required:
                        target = review.target_step_id
                        self._validate_revision_target(step_cfg, target, step_map)

                        cur_rev_count = revision_counts.get(target, 0)
                        if cur_rev_count < self.config.max_revision_loops:
                            revision_counts[target] = cur_rev_count + 1
                            logger.info(
                                "Critic requested revision for step '%s' (Loop %d/%d): %s",
                                target,
                                cur_rev_count + 1,
                                self.config.max_revision_loops,
                                review.reason,
                            )
                            if self.on_revision:
                                try:
                                    self.on_revision(target, review.reason or "", review.feedback or "", cur_rev_count + 1)
                                except Exception as exc:
                                    logger.error("Error in on_revision callback: %s", exc)

                            # Pipeline feedback to target step
                            pipeline.record_step_output(
                                step_id,
                                step_record.agent_role,
                                step_record.result.output if step_record.result else "",
                                critic_review=review,
                            )

                            # Downstream dependency invalidation
                            downstream = self._get_downstream_dependents(target)
                            invalidated = {target} | downstream | {step_id}
                            completed_step_ids.difference_update(invalidated)

                            # Invalidate outputs in pipeline to prevent stale reuse
                            role_map = {}
                            for s in self.config.steps:
                                ag = self.registry.get(s.agent_id)
                                if ag:
                                    role_map[s.step_id] = ag.role
                            pipeline.invalidate_steps(invalidated, role_map)

                            # Rewind queue to target step
                            target_idx = [i for i, sid in enumerate(execution_queue) if sid == target]
                            if target_idx:
                                current_idx = target_idx[0]
                                continue
                        else:
                            logger.warning(
                                "Revision limit (%d) reached for target '%s'. Continuing pipeline without further revision.",
                                self.config.max_revision_loops,
                                target,
                            )
                            pipeline.clear_critic_feedback()
                    else:
                        # Critic approved without revision; clear active feedback
                        pipeline.clear_critic_feedback()

                # Accumulate step output into context pipeline
                output_text = step_record.result.output if step_record.result else ""
                pipeline.record_step_output(step_id, step_record.agent_role, output_text)
                current_idx += 1

            # Determine final status
            unfinished_required = [
                s.step_id for s in self.config.steps
                if not s.is_optional and s.step_id not in completed_step_ids
            ]
            if unfinished_required or failed_step_ids:
                final_status = WorkflowStatus.PARTIAL_SUCCESS if completed_step_ids else WorkflowStatus.FAILED
            elif skipped_step_ids:
                final_status = WorkflowStatus.PARTIAL_SUCCESS
            else:
                final_status = WorkflowStatus.COMPLETED

            return self._finalize_workflow(
                final_status,
                pipeline,
                started_at,
                failed_step_ids,
                skipped_step_ids,
                error=None,
            )

        except (WorkflowTimeoutError, BudgetExceededError, WorkflowExecutionError, InvalidRevisionTargetError) as exc:
            logger.error("Workflow halted: %s", exc)
            return self._finalize_workflow(
                WorkflowStatus.FAILED,
                pipeline,
                started_at,
                failed_step_ids,
                skipped_step_ids,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected workflow engine exception: %s", exc)
            return self._finalize_workflow(
                WorkflowStatus.FAILED,
                pipeline,
                started_at,
                failed_step_ids,
                skipped_step_ids,
                error=f"Unexpected error: {exc}",
            )

    def _execute_step(
        self,
        step_cfg: WorkflowStepConfig,
        pipeline: ContextTransferPipeline,
        revision_iter: int,
    ) -> StepExecutionRecord:
        """Execute a single workflow step using the configured agent, supporting retries."""
        agent = self.registry.get_required(step_cfg.agent_id)
        self._active_agent = agent

        started_at = time.time()
        record = StepExecutionRecord(
            step_id=step_cfg.step_id,
            agent_id=agent.agent_id,
            agent_role=agent.role,
            status=AgentStatus.RUNNING,
            started_at=started_at,
            revision_iteration=revision_iter,
        )

        attempts = 0
        max_attempts = 1 + step_cfg.max_retries
        last_result: Optional[AgentResult] = None

        while attempts < max_attempts:
            if self._cancel_event.is_set():
                record.status = AgentStatus.CANCELLED
                record.finished_at = time.time()
                record.duration_sec = max(0.0, record.finished_at - started_at)
                self._active_agent = None
                return record

            attempts += 1
            record.attempt_count = attempts

            # Prepare defensive step context incorporating revision feedback if targeted
            feedback_str = None
            if (
                pipeline.critic_feedback
                and pipeline.critic_feedback.revision_required
                and pipeline.critic_feedback.target_step_id == step_cfg.step_id
                and revision_iter > 0
            ):
                feedback_str = pipeline.critic_feedback.feedback

            step_context = pipeline.prepare_context_for_step(
                step_config=step_cfg,
                cancel_event=self._cancel_event,
                revision_feedback=feedback_str,
            )

            # Reset agent state if previously finished
            if agent.status != AgentStatus.IDLE:
                agent.reset()

            result = agent.execute(step_context)
            last_result = result
            record.result = result

            if result.status == AgentStatus.COMPLETED:
                record.status = AgentStatus.COMPLETED
                break

            if result.status == AgentStatus.CANCELLED:
                record.status = AgentStatus.CANCELLED
                break

            # Handle retryable failure
            if attempts < max_attempts and not self._cancel_event.is_set():
                logger.info("Step '%s' attempt %d failed (%s). Retrying...", step_cfg.step_id, attempts, result.error)
                time.sleep(0.5)

        record.finished_at = time.time()
        record.duration_sec = max(0.0, record.finished_at - started_at)
        record.status = last_result.status if last_result else AgentStatus.FAILED
        record.error = last_result.error if last_result else "Execution failed without result."

        # If critic step, attach structured review if present in agent result metadata or agent
        if step_cfg.is_critic_step:
            review_obj = getattr(last_result.metadata, "critic_review", None) if (last_result and last_result.metadata) else None
            if not review_obj:
                review_obj = getattr(agent, "critic_review_to_attach", None)
            if isinstance(review_obj, CriticReviewResult):
                record.critic_review = review_obj

        self._active_agent = None
        return record

    def _handle_step_failure(
        self,
        step_cfg: WorkflowStepConfig,
        record: StepExecutionRecord,
        failed_steps: List[str],
        skipped_steps: List[str],
        completed_step_ids: Set[str],
        pipeline: ContextTransferPipeline,
    ) -> bool:
        """Apply the step's failure policy. Returns True if pipeline may continue, False to abort."""
        failed_steps.append(step_cfg.step_id)

        if step_cfg.failure_policy == StepFailurePolicy.ABORT or step_cfg.failure_policy == StepFailurePolicy.RETRY_THEN_ABORT:
            return False

        if step_cfg.failure_policy == StepFailurePolicy.SKIP_STEP:
            if step_cfg.is_optional:
                logger.warning("Optional step '%s' failed; skipping per policy.", step_cfg.step_id)
                skipped_steps.append(step_cfg.step_id)
                return True
            logger.error("Non-optional step '%s' failed under SKIP_STEP policy; aborting.", step_cfg.step_id)
            return False

        if step_cfg.failure_policy == StepFailurePolicy.FALLBACK_STEP:
            if step_cfg.fallback_agent_id:
                logger.info("Step '%s' failed; executing fallback agent '%s'.", step_cfg.step_id, step_cfg.fallback_agent_id)
                fallback_cfg = WorkflowStepConfig(
                    step_id=f"{step_cfg.step_id}_fallback",
                    agent_id=step_cfg.fallback_agent_id,
                    step_name=f"{step_cfg.step_name} (Fallback)",
                    instructions=step_cfg.instructions,
                    failure_policy=StepFailurePolicy.ABORT,
                    max_retries=0,
                )
                # Attempt fallback execution
                fb_record = self._execute_step(fallback_cfg, pipeline, 0)
                self._step_records.append(fb_record)
                if fb_record.status == AgentStatus.COMPLETED:
                    if step_cfg.step_id in failed_steps:
                        failed_steps.remove(step_cfg.step_id)
                    completed_step_ids.add(step_cfg.step_id)
                    output_text = fb_record.result.output if fb_record.result else ""
                    pipeline.record_step_output(step_cfg.step_id, fb_record.agent_role, output_text)
                    return True
                return False
            return False

        return False

    def _check_token_budget(self) -> None:
        """Check whether cumulative verified tokens have exceeded the configured budget."""
        if self.config.token_budget is None:
            return

        cumulative_tokens = sum(
            rec.result.metadata.total_tokens
            for rec in self._step_records
            if rec.result and rec.result.metadata and rec.result.metadata.total_tokens is not None
        )

        if cumulative_tokens > self.config.token_budget:
            raise BudgetExceededError(
                f"Cumulative tokens ({cumulative_tokens}) exceeded workflow token budget of {self.config.token_budget}.",
                workflow_id=self.config.workflow_id,
            )

    def _validate_workflow_structure(self) -> None:
        """Validate step configuration integrity, agent existence, and dependency sanity."""
        known_step_ids: Set[str] = set()

        for step in self.config.steps:
            if step.step_id in known_step_ids:
                raise WorkflowConfigurationError(
                    f"Duplicate step_id '{step.step_id}' found in workflow configuration.",
                    workflow_id=self.config.workflow_id,
                )
            known_step_ids.add(step.step_id)

            # Ensure agent exists in registry
            if not self.registry.get(step.agent_id):
                raise WorkflowConfigurationError(
                    f"Step '{step.step_id}' references unknown agent_id '{step.agent_id}'.",
                    workflow_id=self.config.workflow_id,
                )

            # Ensure fallback agent exists if policy is FALLBACK_STEP
            if step.failure_policy == StepFailurePolicy.FALLBACK_STEP and step.fallback_agent_id:
                if not self.registry.get(step.fallback_agent_id):
                    raise WorkflowConfigurationError(
                        f"Step '{step.step_id}' references unknown fallback_agent_id '{step.fallback_agent_id}'.",
                        workflow_id=self.config.workflow_id,
                    )

            # Validate dependencies reference earlier or known steps
            for dep in step.dependencies:
                if dep == step.step_id:
                    raise WorkflowConfigurationError(
                        f"Step '{step.step_id}' cannot depend on itself.",
                        workflow_id=self.config.workflow_id,
                    )

    def _validate_revision_target(
        self,
        critic_step: WorkflowStepConfig,
        target_step_id: Optional[str],
        step_map: Dict[str, WorkflowStepConfig],
    ) -> None:
        """Validate that a critic's requested revision target is valid and permitted."""
        if not target_step_id:
            raise InvalidRevisionTargetError(
                f"Critic step '{critic_step.step_id}' requested revision but gave no target_step_id.",
                workflow_id=self.config.workflow_id,
            )
        if target_step_id not in step_map:
            raise InvalidRevisionTargetError(
                f"Critic step '{critic_step.step_id}' requested unknown revision target '{target_step_id}'.",
                workflow_id=self.config.workflow_id,
            )
        if critic_step.allowed_revision_targets and target_step_id not in critic_step.allowed_revision_targets:
            raise InvalidRevisionTargetError(
                f"Target '{target_step_id}' is not in allowed_revision_targets for step '{critic_step.step_id}'.",
                workflow_id=self.config.workflow_id,
            )

    def _get_downstream_dependents(self, target_step_id: str) -> Set[str]:
        """Find all step IDs that directly or transitively depend on target_step_id."""
        dependents: Set[str] = set()
        to_check = [target_step_id]

        while to_check:
            current = to_check.pop()
            for step in self.config.steps:
                if step.step_id not in dependents and step.step_id != target_step_id:
                    is_dep = (current in step.dependencies)
                    if not is_dep and step.required_inputs:
                        is_dep = (current in step.required_inputs)
                    if not is_dep and step.is_critic_step and step.allowed_revision_targets:
                        is_dep = (current in step.allowed_revision_targets)
                    if is_dep:
                        dependents.add(step.step_id)
                        to_check.append(step.step_id)

        return dependents

    def _finalize_workflow(
        self,
        final_status: WorkflowStatus,
        pipeline: ContextTransferPipeline,
        started_at: float,
        failed_steps: List[str],
        skipped_steps: List[str],
        error: Optional[str],
    ) -> WorkflowResult:
        """Assemble WorkflowResult and reset internal engine state under lock."""
        finished_at = time.time()
        total_duration = max(0.0, finished_at - started_at)

        # Token aggregation: preserve None when unavailable without fabricating metrics
        prompt_tokens_list = [
            r.result.metadata.prompt_tokens
            for r in self._step_records
            if r.result and r.result.metadata and r.result.metadata.prompt_tokens is not None
        ]
        completion_tokens_list = [
            r.result.metadata.completion_tokens
            for r in self._step_records
            if r.result and r.result.metadata and r.result.metadata.completion_tokens is not None
        ]
        total_tokens_list = [
            r.result.metadata.total_tokens
            for r in self._step_records
            if r.result and r.result.metadata and r.result.metadata.total_tokens is not None
        ]

        tot_prompt = sum(prompt_tokens_list) if prompt_tokens_list else None
        tot_comp = sum(completion_tokens_list) if completion_tokens_list else None
        tot_tokens = sum(total_tokens_list) if total_tokens_list else None

        # Determine final substantive output (prefer FINALIZER, then CREATOR, then last non-critic step)
        final_out = ""
        for rec in reversed(self._step_records):
            if rec.status == AgentStatus.COMPLETED and rec.result and rec.result.output:
                if rec.agent_role == AgentRole.FINALIZER:
                    final_out = rec.result.output
                    break
                elif rec.agent_role == AgentRole.CREATOR and not final_out:
                    final_out = rec.result.output
                elif not final_out and rec.agent_role != AgentRole.CRITIC:
                    final_out = rec.result.output

        if not final_out:
            for rec in reversed(self._step_records):
                if rec.status == AgentStatus.COMPLETED and rec.result and rec.result.output:
                    final_out = rec.result.output
                    break

        # Collect latest valid step outputs only
        step_outputs = {
            k: v for k, v in pipeline.accumulated_outputs.items()
            if k in {s.step_id for s in self.config.steps}
        }

        with self._lifecycle_lock:
            self._status = final_status
            self._active_agent = None
            self._current_task_id = None

        return WorkflowResult(
            workflow_id=self.config.workflow_id,
            task_id=pipeline.task_id,
            status=final_status,
            final_output=final_out,
            step_records=list(self._step_records),
            total_duration_sec=total_duration,
            total_prompt_tokens=tot_prompt,
            total_completion_tokens=tot_comp,
            total_tokens=tot_tokens,
            failed_steps=list(failed_steps),
            skipped_steps=list(skipped_steps),
            error_summary=error,
            step_outputs=step_outputs,
        )

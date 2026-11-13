import copy
import threading
from typing import Any, Dict, List, Optional

from src.agents.context import TaskContext
from src.agents.models import AgentRole
from src.core.message import Message, MessageRole
from src.workflow.models import CriticReviewResult, WorkflowStepConfig


class ContextTransferPipeline:
    """Manages context accumulation, scoping, and isolation between workflow steps.

    Guarantees:
    1. Root user objective and prior conversation history remain strictly immutable.
    2. Outputs from previous steps are structured and selectively provided to downstream agents.
    3. Every step receives a clean, defensive copy of TaskContext.
    """

    def __init__(self, base_context: TaskContext):
        self.task_id: str = base_context.task_id
        self.root_prompt: str = base_context.user_prompt
        self.root_history: List[Message] = [
            Message(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                metadata=dict(m.metadata) if m.metadata else {},
            )
            for m in base_context.conversation_history
        ]
        self.accumulated_outputs: Dict[str, str] = dict(base_context.previous_outputs)
        self.shared_metadata: Dict[str, Any] = copy.deepcopy(base_context.shared_metadata)
        self.critic_feedback: Optional[CriticReviewResult] = None

    def record_step_output(
        self,
        step_id: str,
        agent_role: AgentRole,
        output: str,
        critic_review: Optional[CriticReviewResult] = None,
    ) -> None:
        """Record the output of a completed step into the cumulative pipeline state."""
        if output:
            # Store by step_id and by agent_role
            self.accumulated_outputs[step_id] = output
            self.accumulated_outputs[agent_role.value] = output

        if critic_review is not None:
            self.critic_feedback = critic_review

    def prepare_context_for_step(
        self,
        step_config: WorkflowStepConfig,
        cancel_event: threading.Event,
        revision_feedback: Optional[str] = None,
    ) -> TaskContext:
        """Build an isolated, defensively-copied TaskContext tailored for the upcoming step."""
        # 1. Filter outputs if required_inputs is specified
        if step_config.required_inputs is not None:
            scoped_outputs = {
                k: v
                for k, v in self.accumulated_outputs.items()
                if k in step_config.required_inputs
            }
        else:
            scoped_outputs = dict(self.accumulated_outputs)

        # 2. Prepare step-specific instructions incorporating revision feedback if present
        step_instructions = step_config.instructions or ""
        if revision_feedback:
            revision_note = f"### CRITIQUE & REVISION FEEDBACK:\n{revision_feedback.strip()}"
            if step_instructions:
                step_instructions = f"{step_instructions}\n\n{revision_note}"
            else:
                step_instructions = revision_note

        # 3. Construct defensive TaskContext
        return TaskContext(
            task_id=self.task_id,
            user_prompt=self.root_prompt,  # Pristine immutable root prompt
            conversation_history=[
                Message(m.role, m.content, m.timestamp, dict(m.metadata) if m.metadata else {})
                for m in self.root_history
            ],
            instructions=step_instructions,
            input_data={"step_id": step_config.step_id, "step_name": step_config.step_name},
            previous_outputs=scoped_outputs,
            shared_metadata=copy.deepcopy(self.shared_metadata),
            cancel_event=cancel_event,
        )


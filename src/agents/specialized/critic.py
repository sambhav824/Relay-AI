import logging
from typing import Optional

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentConfig, AgentRole
from src.agents.specialized.contracts import (
    StructuredContractError,
    extract_json_block,
)
from src.core.events import ResponseMetadata
from src.providers.manager import ProviderManager
from src.workflow.models import CriticReviewResult

logger = logging.getLogger("RelayAI.CriticAgent")

DEFAULT_CRITIC_SYSTEM_INSTRUCTION = """You are the specialized Critic Agent in Relay AI.
Your objective is to evaluate the Creator's output against the user objective, constraints, and execution plan.

EVALUATION CRITERIA:
1. Correctness: Are there syntax errors, logic flaws, or security vulnerabilities?
2. Completeness: Were all required components delivered?
3. Instruction Compliance: Did the Creator adhere to all user constraints and formats?
4. Quality: Is the deliverable production-ready?
5. Factual Grounding: Verify that no invented dates, unconfirmed technologies, or hallucinated details were introduced.

MANDATORY OUTPUT FORMAT:
You MUST output a structured JSON evaluation enclosed in ```json ... ``` with the following exact schema:

For APPROVAL (work is satisfactory, no revision needed):
{
  "revision_required": false,
  "target_step_id": null,
  "reason": "Clear explanation of why the output is approved",
  "feedback": "Approved. Deliverable meets all requirements."
}

For REVISION REQUEST (actionable flaws detected):
{
  "revision_required": true,
  "target_step_id": "step_create",
  "reason": "Specific summary of why revision is needed",
  "feedback": "Actionable, precise instructions for the Creator on how to fix the flaw"
}

FAIL-SAFE RULES:
- Only request a revision when there is an actual, actionable defect. Avoid endless nitpicking.
- If requesting revision, 'target_step_id' and 'feedback' MUST be non-empty strings.
- Never output malformed JSON.
"""


class CriticAgent(AgentBase):
    """Specialized AI agent that rigorously evaluates deliverables, enforcing fail-safe review standards."""

    def __init__(
        self,
        agent_id: str = "critic",
        name: str = "Critic Agent",
        config: Optional[AgentConfig] = None,
        description: str = "Evaluates deliverables for correctness, security, and completeness, producing structured reviews.",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if config is None:
            config = AgentConfig(
                provider_id="ollama",
                model_id="llama3.2:3b",
                temperature=0.1,  # Low temperature for strict evaluation
                system_instruction=DEFAULT_CRITIC_SYSTEM_INSTRUCTION,
            )
        elif not config.system_instruction:
            config.system_instruction = DEFAULT_CRITIC_SYSTEM_INSTRUCTION

        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.CRITIC,
            config=config,
            description=description,
            provider_manager=provider_manager,
        )

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        raw_output, meta = super()._run(context)

        if not raw_output or not raw_output.strip():
            raise StructuredContractError(
                "Critic produced an empty or whitespace response. An empty review cannot approve work.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        # 1. Extract JSON block
        json_data = extract_json_block(raw_output)
        if not json_data or not isinstance(json_data, dict):
            raise StructuredContractError(
                "Critic output does not contain a valid JSON review block. Malformed review rejected to prevent silent approval.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        # 2. Strict validation of CriticReviewResult fields
        review = self._validate_review_data(json_data, raw_output)

        # Attach validated CriticReviewResult to ResponseMetadata for WorkflowEngine
        if meta is None:
            meta = ResponseMetadata(model=self.config.model_id, provider_id=self.config.provider_id)
        setattr(meta, "critic_review", review)

        return raw_output, meta

    def _validate_review_data(self, data: dict, raw_output: str) -> CriticReviewResult:
        """Validate Critic review data according to strict fail-safe rules.
        
        Raises:
            StructuredContractError: If review data is malformed or violates contract.
        """
        if "revision_required" not in data:
            raise StructuredContractError(
                "Critic review missing mandatory boolean field 'revision_required'.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        rev_req = data["revision_required"]
        if not isinstance(rev_req, bool):
            raise StructuredContractError(
                f"Critic review 'revision_required' must be a boolean (true/false), got {rev_req!r}.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        target = data.get("target_step_id")
        feedback = str(data.get("feedback", "")).strip()
        reason = str(data.get("reason", "")).strip()

        if rev_req:
            # Mandatory fields when revision is requested
            if not target or not isinstance(target, str) or not target.strip():
                raise StructuredContractError(
                    "Critic requested revision but provided an empty or invalid 'target_step_id'.",
                    raw_content=raw_output,
                    agent_id=self.agent_id,
                )
            if not feedback:
                raise StructuredContractError(
                    "Critic requested revision but provided no actionable 'feedback'.",
                    raw_content=raw_output,
                    agent_id=self.agent_id,
                )
            target = target.strip()
        else:
            # Approval: target_step_id should be None or string
            if target is not None:
                target = str(target).strip() if str(target).strip() else None

        return CriticReviewResult(
            revision_required=rev_req,
            feedback=feedback,
            target_step_id=target,
            reason=reason or ("Revision requested" if rev_req else "Approved"),
        )


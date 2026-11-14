import logging
import re
from typing import Optional

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentConfig, AgentRole
from src.agents.specialized.contracts import (
    PlanMilestone,
    PlanOutput,
    StructuredContractError,
    extract_json_block,
)
from src.core.events import ResponseMetadata
from src.providers.manager import ProviderManager

logger = logging.getLogger("RelayAI.PlannerAgent")

DEFAULT_PLANNER_SYSTEM_INSTRUCTION = """You are the specialized Planner Agent in Relay AI.
Your objective is to deeply understand the user's objective, break complex tasks into logical sequential or parallel milestones, identify dependencies, determine constraints, and outline the expected deliverable.

MANDATORY OUTPUT FORMAT:
You must provide a structured JSON block enclosed in ```json ... ``` with the following exact schema:
{
  "task_objective": "Clear concise summary of the primary objective",
  "deliverable": "Specific description of the expected final deliverable",
  "milestones": [
    {
      "step_number": 1,
      "title": "Short title of step",
      "description": "Actionable description of the work to be done",
      "dependencies": []
    },
    {
      "step_number": 2,
      "title": "Short title of dependent step",
      "description": "Actionable description",
      "dependencies": [1]
    }
  ],
  "constraints": ["Constraint 1", "Constraint 2"],
  "risks": ["Potential risk 1", "Risk mitigation 2"]
}

You may accompany the JSON block with explanatory markdown planning prose.
Never omit the required JSON block.
"""


class PlannerAgent(AgentBase):
    """Specialized AI agent that deconstructs tasks into validated milestone plans."""

    def __init__(
        self,
        agent_id: str = "planner",
        name: str = "Planner Agent",
        config: Optional[AgentConfig] = None,
        description: str = "Deconstructs user objectives into structured milestones, dependencies, and constraints.",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if config is None:
            config = AgentConfig(
                provider_id="ollama",
                model_id="llama3.2:3b",
                temperature=0.3,  # Structured planning benefits from low temperature
                system_instruction=DEFAULT_PLANNER_SYSTEM_INSTRUCTION,
            )
        elif not config.system_instruction:
            config.system_instruction = DEFAULT_PLANNER_SYSTEM_INSTRUCTION

        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.PLANNER,
            config=config,
            description=description,
            provider_manager=provider_manager,
        )

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        raw_output, meta = super()._run(context)

        if not raw_output or not raw_output.strip():
            raise StructuredContractError(
                "Planner produced an empty or whitespace response.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        # 1. Attempt JSON block extraction
        json_data = extract_json_block(raw_output)
        plan: Optional[PlanOutput] = None

        if json_data:
            try:
                plan = PlanOutput.from_dict(json_data, raw_content=raw_output)
            except StructuredContractError as exc:
                exc.agent_id = self.agent_id
                raise exc
            except Exception as exc:
                raise StructuredContractError(
                    f"Failed to validate Planner structured output: {exc}",
                    raw_content=raw_output,
                    agent_id=self.agent_id,
                ) from exc

        # 2. Markdown parsing fallback if no JSON block found
        if plan is None:
            plan = self._parse_markdown_plan(raw_output, context.user_prompt)

        # Attach validated plan object to metadata
        if meta is None:
            meta = ResponseMetadata(model=self.config.model_id, provider_id=self.config.provider_id)
        if not hasattr(meta, "extra") or meta.extra is None:
            meta.extra = {}
        meta.extra["plan"] = plan

        return raw_output, meta

    def _parse_markdown_plan(self, text: str, fallback_objective: str) -> PlanOutput:
        """Parse structured sections from formatted markdown as a secondary fallback."""
        lines = text.splitlines()
        deliverable = ""
        milestones = []
        constraints = []
        risks = []

        current_section = None
        current_step_num = 1

        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue

            lower_line = line_s.lower()
            if "deliverable" in lower_line and line_s.startswith("#"):
                current_section = "deliverable"
                continue
            elif "milestone" in lower_line and line_s.startswith("#"):
                current_section = "milestones"
                continue
            elif "constraint" in lower_line and line_s.startswith("#"):
                current_section = "constraints"
                continue
            elif "risk" in lower_line and line_s.startswith("#"):
                current_section = "risks"
                continue

            if current_section == "deliverable" and not deliverable:
                deliverable = line_s.lstrip("-*# ")
            elif current_section == "milestones" and (line_s.startswith("-") or line_s.startswith("*") or re.match(r"^\d+\.", line_s)):
                clean_item = re.sub(r"^(\d+\.|\-|\*)\s*", "", line_s)
                milestones.append(
                    PlanMilestone(
                        step_number=current_step_num,
                        title=f"Milestone {current_step_num}",
                        description=clean_item,
                        dependencies=[current_step_num - 1] if current_step_num > 1 else [],
                    )
                )
                current_step_num += 1
            elif current_section == "constraints" and (line_s.startswith("-") or line_s.startswith("*")):
                constraints.append(line_s.lstrip("-* "))
            elif current_section == "risks" and (line_s.startswith("-") or line_s.startswith("*")):
                risks.append(line_s.lstrip("-* "))

        if not deliverable:
            deliverable = "Completed task deliverable"

        if not milestones:
            raise StructuredContractError(
                "Could not extract a valid structured plan or milestones from Planner output.",
                raw_content=text,
                agent_id=self.agent_id,
            )

        plan = PlanOutput(
            task_objective=fallback_objective or "User Task Objective",
            deliverable=deliverable,
            milestones=milestones,
            constraints=constraints,
            risks=risks,
            raw_content=text,
        )
        plan.validate()
        return plan


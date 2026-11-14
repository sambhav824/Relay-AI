from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional

from src.agents.exceptions import AgentExecutionError


class StructuredContractError(AgentExecutionError):
    """Raised when an agent's response violates its mandatory structured contract."""
    def __init__(self, message: str, raw_content: str = "", agent_id: Optional[str] = None):
        super().__init__(message, agent_id=agent_id)
        self.raw_content = raw_content


@dataclass
class PlanMilestone:
    """An individual sequential or parallel milestone in an execution plan."""
    step_number: int
    title: str
    description: str
    dependencies: List[int] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.step_number, int) or self.step_number <= 0:
            raise StructuredContractError(f"Milestone 'step_number' must be a positive integer, got {self.step_number!r}.")
        if not self.title or not isinstance(self.title, str) or not self.title.strip():
            raise StructuredContractError(f"Milestone {self.step_number} requires a non-empty string for 'title'.")
        if not self.description or not isinstance(self.description, str) or not self.description.strip():
            raise StructuredContractError(f"Milestone {self.step_number} requires a non-empty string for 'description'.")
        if not isinstance(self.dependencies, list) or not all(isinstance(d, int) for d in self.dependencies):
            raise StructuredContractError(f"Milestone {self.step_number} 'dependencies' must be a list of milestone integers.")


@dataclass
class PlanOutput:
    """Strictly validated structured output produced by a Planner Agent."""
    task_objective: str
    deliverable: str
    milestones: List[PlanMilestone] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    raw_content: str = ""

    def validate(self) -> None:
        if not self.task_objective or not isinstance(self.task_objective, str) or not self.task_objective.strip():
            raise StructuredContractError("PlanOutput requires a non-empty string for 'task_objective'.", raw_content=self.raw_content)
        if not self.deliverable or not isinstance(self.deliverable, str) or not self.deliverable.strip():
            raise StructuredContractError("PlanOutput requires a non-empty string for 'deliverable'.", raw_content=self.raw_content)
        if not self.milestones or not isinstance(self.milestones, list):
            raise StructuredContractError("PlanOutput must define at least one milestone in 'milestones'.", raw_content=self.raw_content)

        known_steps = set()
        for m in self.milestones:
            if not isinstance(m, PlanMilestone):
                raise StructuredContractError(f"Expected PlanMilestone instance, got {type(m)}.", raw_content=self.raw_content)
            m.validate()
            if m.step_number in known_steps:
                raise StructuredContractError(f"Duplicate milestone step_number '{m.step_number}' in plan.", raw_content=self.raw_content)
            known_steps.add(m.step_number)

        # Validate dependency references
        for m in self.milestones:
            for dep in m.dependencies:
                if dep == m.step_number:
                    raise StructuredContractError(f"Milestone {m.step_number} cannot depend on itself.", raw_content=self.raw_content)
                if dep not in known_steps:
                    raise StructuredContractError(
                        f"Milestone {m.step_number} references non-existent milestone dependency {dep}.",
                        raw_content=self.raw_content,
                    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_content: str = "") -> "PlanOutput":
        if not isinstance(data, dict):
            raise StructuredContractError("Plan data must be a dictionary.", raw_content=raw_content)

        milestones_raw = data.get("milestones", [])
        if not isinstance(milestones_raw, list):
            raise StructuredContractError("'milestones' field must be a list.", raw_content=raw_content)

        parsed_milestones = []
        for idx, m_item in enumerate(milestones_raw, start=1):
            if isinstance(m_item, dict):
                parsed_milestones.append(
                    PlanMilestone(
                        step_number=m_item.get("step_number", idx),
                        title=str(m_item.get("title", "")).strip(),
                        description=str(m_item.get("description", "")).strip(),
                        dependencies=m_item.get("dependencies", []),
                    )
                )
            elif isinstance(m_item, str) and m_item.strip():
                parsed_milestones.append(
                    PlanMilestone(
                        step_number=idx,
                        title=f"Milestone {idx}",
                        description=m_item.strip(),
                        dependencies=[],
                    )
                )

        constraints = [str(c).strip() for c in data.get("constraints", []) if str(c).strip()]
        risks = [str(r).strip() for r in data.get("risks", []) if str(r).strip()]

        plan = cls(
            task_objective=str(data.get("task_objective", "")).strip(),
            deliverable=str(data.get("deliverable", "")).strip(),
            milestones=parsed_milestones,
            constraints=constraints,
            risks=risks,
            raw_content=raw_content,
        )
        plan.validate()
        return plan


@dataclass
class ResearchOutput:
    """Strictly validated structured output produced by a Researcher Agent."""
    verified_facts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    information_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_content: str = ""

    def validate(self) -> None:
        if not isinstance(self.verified_facts, list):
            raise StructuredContractError("ResearchOutput 'verified_facts' must be a list.", raw_content=self.raw_content)
        if not isinstance(self.assumptions, list):
            raise StructuredContractError("ResearchOutput 'assumptions' must be a list.", raw_content=self.raw_content)
        if not isinstance(self.information_gaps, list):
            raise StructuredContractError("ResearchOutput 'information_gaps' must be a list.", raw_content=self.raw_content)
        if not isinstance(self.recommendations, list):
            raise StructuredContractError("ResearchOutput 'recommendations' must be a list.", raw_content=self.raw_content)

        # Must contain substantive findings in at least verified facts or assumptions
        if not self.verified_facts and not self.assumptions and not self.recommendations:
            raise StructuredContractError(
                "ResearchOutput must contain at least one verified fact, assumption, or recommendation.",
                raw_content=self.raw_content,
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_content: str = "") -> "ResearchOutput":
        if not isinstance(data, dict):
            raise StructuredContractError("Research data must be a dictionary.", raw_content=raw_content)

        res = cls(
            verified_facts=[str(f).strip() for f in data.get("verified_facts", []) if str(f).strip()],
            assumptions=[str(a).strip() for a in data.get("assumptions", []) if str(a).strip()],
            information_gaps=[str(g).strip() for g in data.get("information_gaps", []) if str(g).strip()],
            recommendations=[str(r).strip() for r in data.get("recommendations", []) if str(r).strip()],
            raw_content=raw_content,
        )
        res.validate()
        return res


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first valid JSON object block from markdown or raw text."""
    if not text or not text.strip():
        return None

    # 1. Search for fenced ```json ... ``` blocks
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except Exception:
            pass

    # 2. Search for any outer { ... } block
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    # 3. Direct JSON parse
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


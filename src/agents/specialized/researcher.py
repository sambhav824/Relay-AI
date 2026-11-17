import logging
from typing import Optional

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentConfig, AgentRole
from src.agents.specialized.contracts import (
    ResearchOutput,
    StructuredContractError,
    extract_json_block,
)
from src.core.events import ResponseMetadata
from src.providers.manager import ProviderManager

logger = logging.getLogger("RelayAI.ResearcherAgent")

DEFAULT_RESEARCHER_SYSTEM_INSTRUCTION = """You are the specialized Researcher Agent in Relay AI.
Your objective is to analyze the task context, technical constraints, and domain background to extract factual findings and highlight unknowns.

STRICT GROUNDING & TRUTHFULNESS RULE:
- NEVER claim or hallucinate that you have browsed the live internet, queried external APIs, or inspected offline repositories unless an explicit tool execution result is provided in context.
- Explicitly distinguish verified facts found directly in the prompt/context from assumptions and technical inferences.
- When the user provides explicit corrections or updates, the new information immediately supersedes and overrides outdated information.
- Answer only what the research inquiry requires without unnecessary preamble.

MANDATORY OUTPUT FORMAT:
You must provide a structured JSON block enclosed in ```json ... ``` with the following exact schema:
{
  "verified_facts": ["Fact directly established in user objective or context 1", "Fact 2"],
  "assumptions": ["Inference or assumption made for technical design 1", "Assumption 2"],
  "information_gaps": ["Missing detail or ambiguity needing attention 1"],
  "recommendations": ["Key technical recommendation 1", "Recommendation 2"]
}

You may accompany the JSON block with explanatory technical research prose.
Never omit the required JSON block.
"""


class ResearcherAgent(AgentBase):
    """Specialized AI agent that analyzes context, identifies facts vs assumptions, and maps gaps."""

    def __init__(
        self,
        agent_id: str = "researcher",
        name: str = "Researcher Agent",
        config: Optional[AgentConfig] = None,
        description: str = "Researches domain requirements, distinguishing verified facts from assumptions and identifying gaps.",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if config is None:
            config = AgentConfig(
                provider_id="ollama",
                model_id="llama3.2:3b",
                temperature=0.2,
                system_instruction=DEFAULT_RESEARCHER_SYSTEM_INSTRUCTION,
            )
        elif not config.system_instruction:
            config.system_instruction = DEFAULT_RESEARCHER_SYSTEM_INSTRUCTION

        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.RESEARCHER,
            config=config,
            description=description,
            provider_manager=provider_manager,
        )

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        raw_output, meta = super()._run(context)

        if not raw_output or not raw_output.strip():
            raise StructuredContractError(
                "Researcher produced an empty or whitespace response.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        # 1. Attempt JSON block extraction
        json_data = extract_json_block(raw_output)
        research: Optional[ResearchOutput] = None

        if json_data:
            try:
                research = ResearchOutput.from_dict(json_data, raw_content=raw_output)
            except StructuredContractError as exc:
                exc.agent_id = self.agent_id
                raise exc
            except Exception as exc:
                raise StructuredContractError(
                    f"Failed to validate Researcher structured output: {exc}",
                    raw_content=raw_output,
                    agent_id=self.agent_id,
                ) from exc

        # 2. Markdown parsing fallback if no JSON block found
        if research is None:
            research = self._parse_markdown_research(raw_output)

        # Attach validated research object to metadata
        if meta is None:
            meta = ResponseMetadata(model=self.config.model_id, provider_id=self.config.provider_id)
        if not hasattr(meta, "extra") or meta.extra is None:
            meta.extra = {}
        meta.extra["research"] = research

        return raw_output, meta

    def _parse_markdown_research(self, text: str) -> ResearchOutput:
        """Parse structured sections from formatted markdown research."""
        lines = text.splitlines()
        verified_facts = []
        assumptions = []
        information_gaps = []
        recommendations = []

        current_section = None

        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue

            lower_line = line_s.lower()
            if "fact" in lower_line and line_s.startswith("#"):
                current_section = "facts"
                continue
            elif "assumption" in lower_line and line_s.startswith("#"):
                current_section = "assumptions"
                continue
            elif ("gap" in lower_line or "missing" in lower_line) and line_s.startswith("#"):
                current_section = "gaps"
                continue
            elif "recommendation" in lower_line and line_s.startswith("#"):
                current_section = "recommendations"
                continue

            if line_s.startswith("-") or line_s.startswith("*"):
                item = line_s.lstrip("-* ")
                if current_section == "facts":
                    verified_facts.append(item)
                elif current_section == "assumptions":
                    assumptions.append(item)
                elif current_section == "gaps":
                    information_gaps.append(item)
                elif current_section == "recommendations":
                    recommendations.append(item)

        if not verified_facts and not assumptions and not recommendations:
            raise StructuredContractError(
                "Could not extract valid structured facts, assumptions, or recommendations from Researcher output.",
                raw_content=text,
                agent_id=self.agent_id,
            )

        research = ResearchOutput(
            verified_facts=verified_facts,
            assumptions=assumptions,
            information_gaps=information_gaps,
            recommendations=recommendations,
            raw_content=text,
        )
        research.validate()
        return research


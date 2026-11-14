import logging
from typing import Optional

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentConfig, AgentRole
from src.agents.specialized.contracts import StructuredContractError
from src.core.events import ResponseMetadata
from src.providers.manager import ProviderManager

logger = logging.getLogger("RelayAI.CreatorAgent")

DEFAULT_CREATOR_SYSTEM_INSTRUCTION = """You are the specialized Creator Agent in Relay AI.
Your objective is to produce the substantive technical deliverable (source code, architecture specifications, API schemas, or documentation) requested by the user.

EXECUTION PRINCIPLES:
1. Ground your work strictly on the approved execution plan and research findings provided in the previous stage outputs.
2. Comply strictly with all user constraints, formats, and design patterns.
3. If CRITIQUE & REVISION FEEDBACK is provided in your instructions, you MUST prioritize resolving every issue identified by the Critic without introducing new flaws.
4. Output clean, complete, production-ready deliverables with clear structure and no unfinished placeholders.
"""


class CreatorAgent(AgentBase):
    """Specialized AI agent that creates substantive implementations, incorporating revisions."""

    def __init__(
        self,
        agent_id: str = "creator",
        name: str = "Creator Agent",
        config: Optional[AgentConfig] = None,
        description: str = "Generates substantive implementations and deliverables, incorporating plan, research, and critique feedback.",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if config is None:
            config = AgentConfig(
                provider_id="ollama",
                model_id="llama3.2:3b",
                temperature=0.5,
                system_instruction=DEFAULT_CREATOR_SYSTEM_INSTRUCTION,
            )
        elif not config.system_instruction:
            config.system_instruction = DEFAULT_CREATOR_SYSTEM_INSTRUCTION

        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.CREATOR,
            config=config,
            description=description,
            provider_manager=provider_manager,
        )

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        raw_output, meta = super()._run(context)

        if not raw_output or not raw_output.strip():
            raise StructuredContractError(
                "Creator produced an empty or whitespace response.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        return raw_output, meta


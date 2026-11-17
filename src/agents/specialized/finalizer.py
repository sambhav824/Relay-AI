import logging
from typing import Optional

from src.agents.base import AgentBase
from src.agents.context import TaskContext
from src.agents.models import AgentConfig, AgentRole
from src.agents.specialized.contracts import StructuredContractError
from src.core.events import ResponseMetadata
from src.providers.manager import ProviderManager

logger = logging.getLogger("RelayAI.FinalizerAgent")

DEFAULT_FINALIZER_SYSTEM_INSTRUCTION = """You are the specialized Finalizer Agent in Relay AI.
Your objective is to produce the final, definitive, client-facing response by consolidating all approved outputs from previous stages.

EDITORIAL STANDARDS:
1. Cohesion: Synthesize the approved deliverables into a polished, unified, professional output.
2. Noise Elimination: Strip away internal scaffolding, prompt instructions, critic debate notes, and debugging scratchpad comments.
3. Integrity: Preserve all crucial technical details, code blocks, instructions, and user constraints.
4. Quality Gate: NEVER reintroduce flawed implementations or patterns that were flagged and rejected during the critique stage.
5. Brevity & Accuracy: Deliver only what the user requested. Never append unrequested roadmaps, summaries, or invented milestones.
"""


class FinalizerAgent(AgentBase):
    """Specialized AI agent that synthesizes approved outputs into a clean final user deliverable."""

    def __init__(
        self,
        agent_id: str = "finalizer",
        name: str = "Finalizer Agent",
        config: Optional[AgentConfig] = None,
        description: str = "Consolidates and refines approved agent outputs into a coherent, client-ready deliverable.",
        provider_manager: Optional[ProviderManager] = None,
    ):
        if config is None:
            config = AgentConfig(
                provider_id="ollama",
                model_id="llama3.2:3b",
                temperature=0.3,
                system_instruction=DEFAULT_FINALIZER_SYSTEM_INSTRUCTION,
            )
        elif not config.system_instruction:
            config.system_instruction = DEFAULT_FINALIZER_SYSTEM_INSTRUCTION

        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.FINALIZER,
            config=config,
            description=description,
            provider_manager=provider_manager,
        )

    def _run(self, context: TaskContext) -> tuple[str, Optional[ResponseMetadata]]:
        raw_output, meta = super()._run(context)

        if not raw_output or not raw_output.strip():
            raise StructuredContractError(
                "Finalizer produced an empty or whitespace response.",
                raw_content=raw_output,
                agent_id=self.agent_id,
            )

        return raw_output, meta


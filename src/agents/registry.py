import logging
import threading
from typing import Dict, List, Optional

from src.agents.base import AgentBase
from src.agents.exceptions import (
    AgentConfigurationError,
    AgentNotFoundError,
    DuplicateAgentIdError,
)
from src.agents.models import AgentRole, AgentStatus

logger = logging.getLogger("RelayAI.AgentRegistry")


class AgentRegistry:
    """Thread-safe registry for agent definitions, instances, and role discovery."""

    def __init__(self):
        self._agents: Dict[str, AgentBase] = {}
        self._lock = threading.RLock()

    def register(self, agent: AgentBase, overwrite: bool = False) -> None:
        """Register an agent instance.

        Raises:
            DuplicateAgentIdError: If agent_id already exists and overwrite is False.
            AgentConfigurationError: If overwriting an agent that is currently RUNNING.
        """
        if not isinstance(agent, AgentBase):
            raise AgentConfigurationError(f"Expected AgentBase instance, got {type(agent)}.")

        with self._lock:
            existing = self._agents.get(agent.agent_id)
            if existing is not None:
                if not overwrite:
                    raise DuplicateAgentIdError(
                        f"Agent with ID '{agent.agent_id}' is already registered. Set overwrite=True to replace.",
                        agent_id=agent.agent_id,
                    )
                if existing.status == AgentStatus.RUNNING:
                    raise AgentConfigurationError(
                        f"Cannot overwrite agent '{agent.agent_id}' while it is actively RUNNING.",
                        agent_id=agent.agent_id,
                    )

            self._agents[agent.agent_id] = agent
            logger.debug("Registered agent '%s' (role: %s).", agent.agent_id, agent.role.value)

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry.

        Raises:
            AgentConfigurationError: If the agent is currently RUNNING.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False

            if agent.status == AgentStatus.RUNNING:
                raise AgentConfigurationError(
                    f"Cannot unregister agent '{agent_id}' while it is actively RUNNING.",
                    agent_id=agent_id,
                )

            del self._agents[agent_id]
            logger.debug("Unregistered agent '%s'.", agent_id)
            return True

    def get(self, agent_id: str) -> Optional[AgentBase]:
        """Retrieve an agent by its unique identifier, or None if not found."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_required(self, agent_id: str) -> AgentBase:
        """Retrieve an agent by ID or raise AgentNotFoundError."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found in registry.", agent_id=agent_id)
            return agent

    def list_agents(self, role: Optional[AgentRole] = None) -> List[AgentBase]:
        """Return a snapshot list of registered agents, optionally filtered by role."""
        with self._lock:
            if role is None:
                return list(self._agents.values())
            return [a for a in self._agents.values() if a.role == role]

    def count(self) -> int:
        """Return total count of registered agents."""
        with self._lock:
            return len(self._agents)

    def clear(self) -> None:
        """Clear all registered agents. Primarily used for test isolation."""
        with self._lock:
            running = [a.agent_id for a in self._agents.values() if a.status == AgentStatus.RUNNING]
            if running:
                raise AgentConfigurationError(
                    f"Cannot clear registry while agents are actively RUNNING: {running}"
                )
            self._agents.clear()


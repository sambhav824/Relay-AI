"""Exception hierarchy for Relay AI Agent Core."""


class AgentCoreError(Exception):
    """Base exception for all agent core operations."""

    def __init__(self, message: str, agent_id: str = ""):
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id

    def __str__(self) -> str:
        if self.agent_id:
            return f"[{self.agent_id}] {self.message}"
        return self.message


class AgentConfigurationError(AgentCoreError):
    """Raised when an agent is configured with invalid, incomplete, or missing parameters."""
    pass


class AgentExecutionError(AgentCoreError):
    """Raised when an agent encounters an unrecoverable failure during execution."""
    pass


class AgentTimeoutError(AgentCoreError):
    """Raised when an agent execution exceeds its configured timeout duration."""
    pass


class AgentCancelledError(AgentCoreError):
    """Raised when an agent execution is interrupted by a cooperative cancellation request."""
    pass


class DuplicateAgentIdError(AgentCoreError):
    """Raised when attempting to register an agent whose ID is already present in the registry."""
    pass


class AgentNotFoundError(AgentCoreError):
    """Raised when an operation references an agent ID that does not exist in the registry."""
    pass


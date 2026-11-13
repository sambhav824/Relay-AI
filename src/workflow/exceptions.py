"""Exception hierarchy for Relay AI Workflow Engine."""


class WorkflowCoreError(Exception):
    """Base exception for all workflow operations."""

    def __init__(self, message: str, workflow_id: str = ""):
        super().__init__(message)
        self.message = message
        self.workflow_id = workflow_id

    def __str__(self) -> str:
        if self.workflow_id:
            return f"[{self.workflow_id}] {self.message}"
        return self.message


class WorkflowConfigurationError(WorkflowCoreError):
    """Raised when a workflow configuration has invalid steps, dependencies, or cycle errors."""
    pass


class WorkflowExecutionError(WorkflowCoreError):
    """Raised when an unrecoverable step failure or concurrency conflict halts execution."""
    pass


class WorkflowTimeoutError(WorkflowCoreError):
    """Raised when overall workflow execution exceeds its configured timeout duration."""
    pass


class WorkflowCancelledError(WorkflowCoreError):
    """Raised when workflow execution is interrupted by a cooperative cancellation request."""
    pass


class BudgetExceededError(WorkflowCoreError):
    """Raised when measured token or cost usage exceeds the configured workflow budget."""
    pass


class RevisionLoopLimitError(WorkflowCoreError):
    """Raised when revision loops exceed configured bounds."""
    pass


class InvalidRevisionTargetError(WorkflowCoreError):
    """Raised when a critic requests revision for an unknown or unpermitted step."""
    pass


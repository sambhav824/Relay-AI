from typing import Optional


class TeamError(Exception):
    """Base exception for all AI Team related errors."""
    def __init__(self, message: str, team_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.team_id = team_id

    def __str__(self) -> str:
        if self.team_id:
            return f"[{self.team_id}] {self.message}"
        return self.message


class TeamConfigurationError(TeamError):
    """Raised when an AI Team configuration, member specification, or validation fails."""
    pass


class TeamExecutionError(TeamError):
    """Raised when an error occurs during AI Team execution orchestration."""
    pass


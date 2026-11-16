class ProviderError(Exception):
    """Base exception for all AI provider operations."""
    def __init__(self, message: str, provider_id: str = ""):
        super().__init__(message)
        self.message = message
        self.provider_id = provider_id

    def __str__(self) -> str:
        if self.provider_id:
            return f"[{self.provider_id.upper()}] {self.message}"
        return self.message


class ProviderConnectionError(ProviderError):
    """Raised when communication with the provider fails (e.g. host unreachable)."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication fails (e.g. invalid or missing API key)."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limits / quotas are exceeded."""
    pass


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not found or not pulled."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when the provider endpoint takes too long to respond."""
    pass


# Backward-compatibility aliases
ProviderAuthError = ProviderAuthenticationError
ProviderNotFoundError = ModelNotFoundError



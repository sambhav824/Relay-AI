import logging
from typing import Dict, List, Optional

from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.models import ModelInfo
from src.providers.ollama import OllamaProvider
from src.providers.gemini import GeminiProvider
from src.providers.groq import GroqProvider
from src.providers.openai import OpenAIProvider
from src.providers.openrouter import OpenRouterProvider
from src.providers.mock import MockProvider
from src.providers.cost_protection import UsageLimits

logger = logging.getLogger("RelayAI.ProviderManager")


class ProviderManager:
    """Manages AI model providers, credentials, and active dispatch for Relay AI."""

    def __init__(self, credential_store: Optional[CredentialStore] = None):
        self.credential_store = credential_store or CredentialStore()
        self._providers: Dict[str, AIProvider] = {}
        self._active_provider_id: str = "ollama"

        # Register default local Ollama provider
        self.register_provider(OllamaProvider())
        # Register Google Gemini cloud provider
        self.register_provider(GeminiProvider(credential_store=self.credential_store))
        # Register Groq cloud provider
        self.register_provider(GroqProvider(credential_store=self.credential_store))
        # Register OpenAI cloud provider
        self.register_provider(OpenAIProvider(credential_store=self.credential_store))
        # Register OpenRouter cloud provider
        self.register_provider(OpenRouterProvider(credential_store=self.credential_store))
        openai = self.get_provider("openai")
        if isinstance(openai, OpenAIProvider):
            openai.configure_fallback(self.get_provider("ollama"))
            if openai.has_credentials(self.credential_store):
                self._active_provider_id = "openai"

    def register_provider(self, provider: AIProvider) -> None:
        """Register an AI provider instance."""
        if not isinstance(provider, AIProvider):
            raise TypeError(f"Provider must implement AIProvider, got {type(provider)}")
        self._providers[provider.provider_id.lower()] = provider
        logger.info("Registered AI provider: %s (%s)", provider.display_name, provider.provider_id)

    def get_provider(self, provider_id: str) -> Optional[AIProvider]:
        """Retrieve a registered provider by its identifier."""
        return self._providers.get(provider_id.lower())

    def list_providers(self) -> List[AIProvider]:
        """List all registered AI providers."""
        return list(self._providers.values())

    def set_active_provider(self, provider_id: str) -> bool:
        """Set the active provider for chat operations."""
        pid = provider_id.lower()
        if pid in self._providers:
            self._active_provider_id = pid
            logger.info("Active provider switched to: %s", pid)
            return True
        logger.warning("Attempted to set unknown active provider: %s", provider_id)
        return False

    def get_active_provider(self) -> AIProvider:
        """Get the currently selected AI provider."""
        provider = self.get_provider(self._active_provider_id)
        if not provider:
            if self._providers:
                return next(iter(self._providers.values()))
            fallback = OllamaProvider()
            self.register_provider(fallback)
            return fallback
        return provider

    def get_available_models(self, provider_id: Optional[str] = None) -> List[ModelInfo]:
        """Get available models for a specific provider or aggregated across all providers."""
        if provider_id:
            provider = self.get_provider(provider_id)
            return provider.get_available_models() if provider else []

        # Return models for active provider
        active = self.get_active_provider()
        return active.get_available_models()

    def validate_provider(self, provider_id: str) -> bool:
        """Check if a provider is reachable and operational."""
        provider = self.get_provider(provider_id)
        if not provider:
            return False
        return provider.validate_connection()

    def configure_usage_protection(self, limits: UsageLimits, user_id_provider=None) -> None:
        """Apply usage settings to the existing OpenAI provider implementation."""
        provider = self.get_provider("openai")
        if isinstance(provider, OpenAIProvider):
            provider.configure_usage(limits, user_id_provider)

    def set_automatic_fallback(self, enabled: bool) -> None:
        """Enable or disable OpenAI-to-Ollama fallback without restarting the app."""
        provider = self.get_provider("openai")
        if isinstance(provider, OpenAIProvider):
            provider.configure_fallback(self.get_provider("ollama"), enabled)


# Global default manager singleton
_default_manager: Optional[ProviderManager] = None


def get_default_provider_manager() -> ProviderManager:
    """Return the global ProviderManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ProviderManager()
    return _default_manager

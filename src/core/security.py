import logging
from typing import Optional

logger = logging.getLogger("RelayAI.Security")

SERVICE_NAME = "RelayAI"


class CredentialStore:
    """Manages secure storage of API credentials via the OS keyring (Windows Credential Manager)."""

    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name

    def set_api_key(self, provider_id: str, api_key: str) -> bool:
        """Securely store an API key for a given provider."""
        if not provider_id or not api_key:
            return False
        try:
            import keyring
            keyring.set_password(self.service_name, provider_id.strip().lower(), api_key.strip())
            logger.info("API key securely stored for provider: %s", provider_id)
            return True
        except Exception as exc:
            logger.error("Failed to store API key in Windows Credential Manager: %s", type(exc).__name__)
            return False

    def get_api_key(self, provider_id: str) -> Optional[str]:
        """Retrieve an API key for a given provider from the secure store or environment."""
        if not provider_id:
            return None
        pid = provider_id.strip().lower()
        # 1. Try OS Credential Manager
        try:
            import keyring
            key = keyring.get_password(self.service_name, pid)
            if key:
                return key.strip()
        except Exception as exc:
            logger.debug("Keyring access skipped/failed: %s", type(exc).__name__)

        # 2. Fallback to standard environment variables (Render, Linux, Docker, CI)
        import os
        env_vars = [f"{pid.upper()}_API_KEY", f"{pid.upper()}_KEY"]
        if pid == "gemini":
            env_vars.extend(["GOOGLE_API_KEY", "GEMINI_KEY"])
        for var in env_vars:
            val = os.environ.get(var)
            if val and val.strip():
                return val.strip()

        return None

    def delete_api_key(self, provider_id: str) -> bool:
        """Remove a stored API key for a given provider."""
        if not provider_id:
            return False
        try:
            import keyring
            keyring.delete_password(self.service_name, provider_id.strip().lower())
            logger.info("API key deleted for provider: %s", provider_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete API key from Windows Credential Manager: %s", type(exc).__name__)
            return False

    def has_api_key(self, provider_id: str) -> bool:
        """Check if an API key exists for a given provider without exposing its value."""
        key = self.get_api_key(provider_id)
        return bool(key and key.strip())


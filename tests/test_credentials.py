import os
import unittest
from unittest.mock import patch, MagicMock

from src.core.security import CredentialStore


class TestCredentialStore(unittest.TestCase):
    def setUp(self):
        self.store = CredentialStore(service_name="TestRelayAI")

    def test_environment_variable_precedence_over_keyring(self):
        """Server-side environment variables must take precedence for cloud/Render deployments."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "env_groq_key_999"}, clear=False):
            with patch("keyring.get_password", return_value="keyring_groq_key_111"):
                key = self.store.get_api_key("groq")
                self.assertEqual(key, "env_groq_key_999")

    def test_keyring_fallback_when_no_env_var(self):
        """Native desktop app relies on keyring/Windows Credential Locker when no env var is present."""
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GROQ_")}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("keyring.get_password", return_value="desktop_keyring_key"):
                key = self.store.get_api_key("groq")
                self.assertEqual(key, "desktop_keyring_key")

    def test_gemini_env_var_variants(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google_api_key_val"}, clear=False):
            with patch("keyring.get_password", return_value=None):
                key = self.store.get_api_key("gemini")
                self.assertEqual(key, "google_api_key_val")

    def test_has_api_key_truthfulness(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-valid-key"}, clear=False):
            self.assertTrue(self.store.has_api_key("openai"))


if __name__ == "__main__":
    unittest.main()

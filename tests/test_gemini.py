import unittest
from unittest.mock import MagicMock, patch

from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from src.providers.gemini import GeminiProvider


class TestGeminiProvider(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=CredentialStore)
        self.mock_store.get_api_key.return_value = "fake_test_key"
        self.provider = GeminiProvider(credential_store=self.mock_store)

    def test_missing_api_key_raises_auth_error(self):
        self.mock_store.get_api_key.return_value = None
        provider = GeminiProvider(credential_store=self.mock_store)
        with self.assertRaises(ProviderAuthenticationError):
            list(provider.stream([Message("user", "hi")]))

    def test_contents_and_system_instruction_preparation(self):
        messages = [
            Message(MessageRole.SYSTEM, "You are a coding assistant."),
            Message(MessageRole.USER, "Write a function."),
            Message(MessageRole.ASSISTANT, "def foo(): pass"),
            Message(MessageRole.USER, "Now test it."),
        ]
        contents, config = self.provider._prepare_contents(messages)
        self.assertIsNotNone(config)
        self.assertEqual(config.system_instruction, "You are a coding assistant.")
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(contents[2].role, "user")

    @patch("google.genai.Client")
    def test_streaming_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk1.candidates = []

        chunk2 = MagicMock()
        chunk2.text = "World!"
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        chunk2.candidates = [candidate]

        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]

        chunks = list(self.provider.stream([Message("user", "Hello")], model="gemini-3.8-flash"))
        # 2 content chunks + 1 final completion chunk
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].delta, "Hello ")
        self.assertFalse(chunks[0].is_final)
        self.assertEqual(chunks[1].delta, "World!")
        self.assertFalse(chunks[1].is_final)
        self.assertEqual(chunks[2].delta, "")
        self.assertTrue(chunks[2].is_final)

    @patch("google.genai.Client")
    def test_generate_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = "Full generated answer."
        mock_client.models.generate_content.return_value = mock_resp

        result = self.provider.generate([Message("user", "Hi")], model="gemini-3.8-flash")
        self.assertEqual(result, "Full generated answer.")

    @patch("google.genai.Client")
    def test_generate_empty_response_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_client.models.generate_content.return_value = mock_resp

        with self.assertRaises(ProviderError):
            self.provider.generate([Message("user", "Hi")])

    @patch("google.genai.Client")
    def test_validate_connection_success_and_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.list.return_value = [MagicMock()]

        self.assertTrue(self.provider.validate_connection())

        mock_client.models.list.side_effect = RuntimeError("Network error")
        self.assertFalse(self.provider.validate_connection())

    @patch("google.genai.Client")
    def test_get_available_models(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        m1 = MagicMock()
        m1.name = "models/gemini-3.8-flash"
        m1.display_name = "Gemini 3.8 Flash"
        m1.supported_actions = ["generateContent"]

        m2 = MagicMock()
        m2.name = "models/embedding-001"
        m2.supported_actions = ["embedContent"]

        mock_client.models.list.return_value = [m1, m2]

        models = self.provider.get_available_models()
        model_ids = [m.id for m in models]
        self.assertIn("gemini-3.8-flash", model_ids)
        self.assertNotIn("embedding-001", model_ids)

    def test_error_mapping_auth_error(self):
        exc = Exception("API_KEY_INVALID: User not authorized")
        with self.assertRaises(ProviderAuthenticationError):
            self.provider._handle_exception(exc, "gemini-3.8-flash")

    def test_error_mapping_rate_limit(self):
        exc = Exception("RESOURCE_EXHAUSTED: Rate limit exceeded")
        with self.assertRaises(ProviderRateLimitError):
            self.provider._handle_exception(exc, "gemini-3.8-flash")

    def test_error_mapping_model_not_found(self):
        exc = Exception("NOT_FOUND: Model not found")
        with self.assertRaises(ModelNotFoundError):
            self.provider._handle_exception(exc, "gemini-unknown")


if __name__ == "__main__":
    unittest.main()


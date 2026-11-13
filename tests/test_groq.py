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
from src.providers.groq import GroqProvider


class TestGroqProvider(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=CredentialStore)
        self.mock_store.get_api_key.return_value = "fake_groq_test_key"
        self.provider = GroqProvider(credential_store=self.mock_store)

    def test_missing_api_key_raises_auth_error(self):
        self.mock_store.get_api_key.return_value = None
        provider = GroqProvider(credential_store=self.mock_store)
        with self.assertRaises(ProviderAuthenticationError):
            list(provider.stream([Message("user", "hi")]))

    def test_message_formatting(self):
        messages = [
            Message(MessageRole.SYSTEM, "You are a helpful assistant."),
            Message(MessageRole.USER, "What is Groq?"),
            Message(MessageRole.ASSISTANT, "Groq is an ultra-fast LPU engine."),
        ]
        formatted = self.provider._prepare_messages(messages)
        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(formatted[1], {"role": "user", "content": "What is Groq?"})
        self.assertEqual(formatted[2], {"role": "assistant", "content": "Groq is an ultra-fast LPU engine."})

    @patch("groq.Groq")
    def test_streaming_success(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Ultra "), finish_reason=None)]

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="fast!"), finish_reason="stop")]

        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        chunks = list(self.provider.stream([Message("user", "Hello")], model="openai/gpt-oss-120b"))
        # 2 content chunks + 1 final completion chunk
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].delta, "Ultra ")
        self.assertFalse(chunks[0].is_final)
        self.assertEqual(chunks[1].delta, "fast!")
        self.assertFalse(chunks[1].is_final)
        self.assertEqual(chunks[2].delta, "")
        self.assertTrue(chunks[2].is_final)

    @patch("groq.Groq")
    def test_generate_success(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Complete Groq response."
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = self.provider.generate([Message("user", "Hi")], model="openai/gpt-oss-120b")
        self.assertEqual(result, "Complete Groq response.")

    @patch("groq.Groq")
    def test_validate_connection_success_and_failure(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock(data=[])

        self.assertTrue(self.provider.validate_connection())

        mock_client.models.list.side_effect = RuntimeError("Connection failed")
        self.assertFalse(self.provider.validate_connection())

    @patch("groq.Groq")
    def test_get_available_models(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        m1 = MagicMock(id="openai/gpt-oss-120b")
        m2 = MagicMock(id="qwen/qwen3.8-27b")
        m3 = MagicMock(id="whisper-large-v3")  # Audio model, should be excluded

        mock_client.models.list.return_value = MagicMock(data=[m1, m2, m3])

        models = self.provider.get_available_models()
        model_ids = [m.id for m in models]
        self.assertIn("openai/gpt-oss-120b", model_ids)
        self.assertIn("qwen/qwen3.8-27b", model_ids)
        self.assertNotIn("whisper-large-v3", model_ids)

    def test_error_mapping_auth_error(self):
        exc = Exception("invalid_api_key: The provided API key is invalid.")
        setattr(exc, "status_code", 401)
        with self.assertRaises(ProviderAuthenticationError):
            self.provider._handle_exception(exc, "openai/gpt-oss-120b")

    def test_error_mapping_rate_limit(self):
        exc = Exception("rate_limit_exceeded: Rate limit reached")
        setattr(exc, "status_code", 429)
        with self.assertRaises(ProviderRateLimitError):
            self.provider._handle_exception(exc, "openai/gpt-oss-120b")

    def test_error_mapping_model_not_found(self):
        exc = Exception("model_not_found: The model does not exist")
        setattr(exc, "status_code", 404)
        with self.assertRaises(ModelNotFoundError):
            self.provider._handle_exception(exc, "unknown-model")


if __name__ == "__main__":
    unittest.main()


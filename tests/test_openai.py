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
from src.providers.openai import OpenAIProvider, DEFAULT_OPENAI_MODEL


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=CredentialStore)
        self.mock_store.get_api_key.return_value = "fake_openai_test_key"
        self.provider = OpenAIProvider(credential_store=self.mock_store)

    def test_missing_api_key_raises_auth_error(self):
        self.mock_store.get_api_key.return_value = None
        provider = OpenAIProvider(credential_store=self.mock_store)
        with self.assertRaises(ProviderAuthenticationError):
            list(provider.stream([Message("user", "hi")]))

    def test_message_formatting(self):
        messages = [
            Message(MessageRole.SYSTEM, "You are a helpful assistant."),
            Message(MessageRole.USER, "What is OpenAI?"),
            Message(MessageRole.ASSISTANT, "OpenAI is an AI research company."),
        ]
        formatted = self.provider._prepare_messages(messages)
        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(formatted[1], {"role": "user", "content": "What is OpenAI?"})
        self.assertEqual(formatted[2], {"role": "assistant", "content": "OpenAI is an AI research company."})

    @patch("openai.OpenAI")
    def test_streaming_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)]

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="World!"), finish_reason="stop")]

        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        chunks = list(self.provider.stream([Message("user", "Hello")], model=DEFAULT_OPENAI_MODEL))
        # 2 content chunks + 1 final completion chunk
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].delta, "Hello ")
        self.assertFalse(chunks[0].is_final)
        self.assertEqual(chunks[1].delta, "World!")
        self.assertFalse(chunks[1].is_final)
        self.assertEqual(chunks[2].delta, "")
        self.assertTrue(chunks[2].is_final)

    @patch("openai.OpenAI")
    def test_generate_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Complete OpenAI response."
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = self.provider.generate([Message("user", "Hi")], model=DEFAULT_OPENAI_MODEL)
        self.assertEqual(result, "Complete OpenAI response.")

    @patch("openai.OpenAI")
    def test_validate_connection_success_and_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list.return_value = MagicMock(data=[])

        self.assertTrue(self.provider.validate_connection())

        mock_client.models.list.side_effect = RuntimeError("Connection failed")
        self.assertFalse(self.provider.validate_connection())

    @patch("openai.OpenAI")
    def test_get_available_models(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        m1 = MagicMock(id="gpt-4o")
        m2 = MagicMock(id="gpt-4o-mini")
        m3 = MagicMock(id="whisper-1")  # Audio model, should be excluded
        m4 = MagicMock(id="dall-e-3")   # Image model, should be excluded

        mock_client.models.list.return_value = MagicMock(data=[m1, m2, m3, m4])

        models = self.provider.get_available_models()
        model_ids = [m.id for m in models]
        self.assertIn("gpt-4o", model_ids)
        self.assertIn("gpt-4o-mini", model_ids)
        self.assertNotIn("whisper-1", model_ids)
        self.assertNotIn("dall-e-3", model_ids)
        # gpt-4o-mini ranked first
        self.assertEqual(models[0].id, "gpt-4o-mini")

    def test_error_mapping_auth_error(self):
        exc = Exception("invalid_api_key: Incorrect API key provided.")
        setattr(exc, "status_code", 401)
        with self.assertRaises(ProviderAuthenticationError):
            self.provider._handle_exception(exc, DEFAULT_OPENAI_MODEL)

    def test_error_mapping_rate_limit(self):
        exc = Exception("rate_limit_exceeded: Rate limit reached")
        setattr(exc, "status_code", 429)
        with self.assertRaises(ProviderRateLimitError):
            self.provider._handle_exception(exc, DEFAULT_OPENAI_MODEL)

    def test_error_mapping_quota_exhaustion(self):
        exc = Exception("You have no credits remaining. Add credits to continue using the API.")
        setattr(exc, "status_code", 429)
        with self.assertRaises(ProviderRateLimitError) as ctx:
            self.provider._handle_exception(exc, DEFAULT_OPENAI_MODEL)
        self.assertIn("credits remaining", str(ctx.exception))

    def test_error_mapping_model_not_found(self):
        exc = Exception("model_not_found: The model gpt-old does not exist")
        setattr(exc, "status_code", 404)
        with self.assertRaises(ModelNotFoundError):
            self.provider._handle_exception(exc, "gpt-old")

    def test_error_mapping_connection_timeout(self):
        exc = Exception("APITimeoutError: Request timed out")
        with self.assertRaises(ProviderConnectionError):
            self.provider._handle_exception(exc, DEFAULT_OPENAI_MODEL)

    def test_live_key_connection_if_configured(self):
        real_store = CredentialStore()
        if real_store.has_api_key("openai"):
            live_provider = OpenAIProvider(credential_store=real_store)
            valid = live_provider.validate_connection()
            self.assertTrue(valid, "Expected live OpenAI connection validation to succeed with configured key")


if __name__ == "__main__":
    unittest.main()


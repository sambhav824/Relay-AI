import unittest
from unittest.mock import MagicMock, patch
from src.core.events import StreamChunk
from src.core.message import Message, MessageRole
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.providers.ollama import OllamaProvider, generate, stream_chat


class DummyProvider(AIProvider):
    provider_id = "dummy"
    display_name = "Dummy Provider"
    is_local = False

    def generate(self, messages, model, **kwargs):
        return f"Dummy response to: {messages[-1].content}"

    def stream(self, messages, model, **kwargs):
        yield StreamChunk(delta="Hello ")
        yield StreamChunk(delta="World", is_final=True)

    def validate_connection(self):
        return True

    def get_available_models(self):
        return [ModelInfo(id="dummy-1", name="Dummy One", provider_id="dummy")]


class TestProviderArchitecture(unittest.TestCase):
    def test_provider_manager_registration_and_active(self):
        mgr = ProviderManager()
        # Default Ollama, Gemini, Groq, and OpenAI are registered
        self.assertIsNotNone(mgr.get_provider("ollama"))
        self.assertIsNotNone(mgr.get_provider("gemini"))
        self.assertIsNotNone(mgr.get_provider("groq"))
        self.assertIsNotNone(mgr.get_provider("openai"))
        self.assertEqual(mgr.get_active_provider().provider_id, "ollama")

        # Register DummyProvider
        dummy = DummyProvider()
        mgr.register_provider(dummy)
        self.assertEqual(mgr.get_provider("dummy"), dummy)
        self.assertEqual(len(mgr.list_providers()), 5)

        # Switch active provider
        self.assertTrue(mgr.set_active_provider("dummy"))
        self.assertEqual(mgr.get_active_provider(), dummy)

        # Query models
        models = mgr.get_available_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "dummy-1")

    def test_ollama_provider_normalize_messages(self):
        provider = OllamaProvider()
        raw_msgs = [
            {"role": "system", "content": "Be concise"},
            Message(MessageRole.USER, "Hello"),
            "Raw string prompt",
        ]
        normalized = provider.normalize_messages(raw_msgs)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0].role, MessageRole.SYSTEM)
        self.assertEqual(normalized[1].role, MessageRole.USER)
        self.assertEqual(normalized[2].role, MessageRole.USER)
        self.assertEqual(normalized[2].content, "Raw string prompt")

    def test_ollama_provider_streaming_mock(self):
        provider = OllamaProvider()
        fake_lines = [
            b'{"message": {"role": "assistant", "content": "Hi "}, "done": false}',
            b'{"message": {"role": "assistant", "content": "there!"}, "done": true, "eval_count": 5}',
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = fake_lines
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("httpx.stream", return_value=mock_resp):
            chunks = list(provider.stream([Message("user", "Hello")], model="llama3.2:3b"))
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].delta, "Hi ")
            self.assertFalse(chunks[0].is_final)
            self.assertEqual(chunks[1].delta, "there!")
            self.assertTrue(chunks[1].is_final)
            self.assertIsNotNone(chunks[1].metadata)
            self.assertEqual(chunks[1].metadata.total_tokens, 5)

    def test_legacy_backward_compatibility(self):
        fake_lines = [
            b'{"message": {"role": "assistant", "content": "Legacy "}, "done": false}',
            b'{"message": {"role": "assistant", "content": "call"}, "done": true}',
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = fake_lines
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("httpx.stream", return_value=mock_resp):
            legacy_chunks = list(stream_chat([{"role": "user", "content": "hi"}]))
            self.assertEqual("".join(legacy_chunks), "Legacy call")

    def test_provider_exceptions(self):
        err = ProviderConnectionError("Cannot reach localhost", provider_id="ollama")
        self.assertEqual(str(err), "[OLLAMA] Cannot reach localhost")
        self.assertEqual(err.provider_id, "ollama")


if __name__ == "__main__":
    unittest.main()

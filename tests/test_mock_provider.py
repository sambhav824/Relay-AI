"""Unit tests for MockProvider."""

import unittest

from src.core.message import Message, MessageRole
from src.providers.exceptions import ModelNotFoundError, ProviderAuthenticationError, ProviderTimeoutError
from src.providers.mock import MockProvider


class TestMockProvider(unittest.TestCase):
    def test_basic_generation(self):
        p = MockProvider(default_response="Mock answer")
        resp = p.generate([Message(MessageRole.USER, "hello")])
        self.assertEqual(resp, "Mock answer")
        self.assertEqual(len(p.recorded_calls), 1)

    def test_streaming(self):
        p = MockProvider(default_response="Alpha beta gamma")
        chunks = list(p.stream([Message(MessageRole.USER, "stream test")]))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[-1].is_final)
        self.assertIsNotNone(chunks[-1].metadata)
        self.assertEqual(chunks[-1].metadata.model, "mock-model-1")

    def test_model_not_found(self):
        p = MockProvider(models=["model-a"])
        with self.assertRaises(ModelNotFoundError):
            p.generate([Message(MessageRole.USER, "test")], model="model-b")

    def test_error_simulation(self):
        p = MockProvider(simulate_error=ProviderTimeoutError("Simulated timeout", provider_id="mock"))
        with self.assertRaises(ProviderTimeoutError):
            p.generate([Message(MessageRole.USER, "test")])

        self.assertFalse(
            MockProvider(simulate_error=ProviderAuthenticationError("Auth failed", provider_id="mock")).validate_connection()
        )


if __name__ == "__main__":
    unittest.main()

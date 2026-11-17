"""Unit tests for OpenRouter AI provider adapter."""

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from src.core.message import Message, MessageRole
from src.core.security import CredentialStore
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.providers.openrouter import OpenRouterProvider


class TestOpenRouterProvider(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=CredentialStore)
        self.mock_store.get_api_key.return_value = "sk-or-v1-testkey123"
        self.mock_store.has_api_key.return_value = True
        self.provider = OpenRouterProvider(credential_store=self.mock_store)

    def test_provider_initialization_and_metadata(self):
        self.assertEqual(self.provider.provider_id, "openrouter")
        self.assertEqual(self.provider.display_name, "OpenRouter")
        self.assertFalse(self.provider.is_local)
        self.assertTrue(self.provider.supports_streaming)
        self.assertTrue(self.provider.has_credentials())

    def test_missing_credentials_raises_auth_error(self):
        store = MagicMock(spec=CredentialStore)
        store.get_api_key.return_value = None
        store.has_api_key.return_value = False
        p = OpenRouterProvider(credential_store=store)
        self.assertFalse(p.has_credentials())
        with self.assertRaises(ProviderAuthenticationError):
            list(p.stream([Message(MessageRole.USER, "hello")]))

    def test_stream_successful_response(self):
        sse_lines = [
            b'data: {"id":"gen-1","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"id":"gen-1","choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            b'data: {"id":"gen-1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
            b'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [line.decode("utf-8") for line in sse_lines]

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.return_value.__enter__.return_value = mock_response

        provider = OpenRouterProvider(credential_store=self.mock_store, http_client=mock_client)
        chunks = list(provider.stream([Message(MessageRole.USER, "hi")]))

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].delta, "Hello")
        self.assertFalse(chunks[0].is_final)
        self.assertEqual(chunks[1].delta, " world")
        self.assertFalse(chunks[1].is_final)
        self.assertTrue(chunks[2].is_final)
        self.assertIsNotNone(chunks[2].metadata)
        self.assertEqual(chunks[2].metadata.prompt_tokens, 5)
        self.assertEqual(chunks[2].metadata.completion_tokens, 2)
        self.assertEqual(chunks[2].metadata.total_tokens, 7)

    def test_generate_aggregates_stream(self):
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"One "},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"Two"},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [l.decode("utf-8") for l in sse_lines]
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.return_value.__enter__.return_value = mock_response

        provider = OpenRouterProvider(credential_store=self.mock_store, http_client=mock_client)
        full_text = provider.generate([Message(MessageRole.USER, "hi")])
        self.assertEqual(full_text, "One Two")

    def test_error_classification(self):
        error_cases = [
            (401, '{"error":{"message":"Invalid API key"}}', ProviderAuthenticationError),
            (403, '{"error":{"message":"Forbidden"}}', ProviderAuthenticationError),
            (402, '{"error":{"message":"Insufficient credits"}}', ProviderBillingError),
            (429, '{"error":{"message":"Rate limit exceeded"}}', ProviderRateLimitError),
            (429, '{"error":{"message":"Monthly quota reached"}}', ProviderQuotaError),
            (404, '{"error":{"message":"Model not found"}}', ModelNotFoundError),
            (500, '{"error":{"message":"Internal server error"}}', ProviderServerError),
            (503, '{"error":{"message":"Service unavailable"}}', ProviderServerError),
        ]

        for code, body, expected_exc in error_cases:
            with self.subTest(code=code, body=body):
                mock_resp = MagicMock()
                mock_resp.status_code = code
                mock_resp.read.return_value = body.encode("utf-8")
                mock_client = MagicMock(spec=httpx.Client)
                mock_client.stream.return_value.__enter__.return_value = mock_resp

                provider = OpenRouterProvider(credential_store=self.mock_store, http_client=mock_client)
                with self.assertRaises(expected_exc):
                    list(provider.stream([Message(MessageRole.USER, "hi")]))

    def test_connection_and_timeout_errors(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = httpx.ConnectError("Connection refused")
        provider = OpenRouterProvider(credential_store=self.mock_store, http_client=mock_client)
        with self.assertRaises(ProviderConnectionError):
            list(provider.stream([Message(MessageRole.USER, "hi")]))

        mock_client.stream.side_effect = httpx.TimeoutException("Timed out")
        with self.assertRaises(ProviderTimeoutError):
            list(provider.stream([Message(MessageRole.USER, "hi")]))

    def test_get_available_models(self):
        models_payload = {
            "data": [
                {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "context_length": 131072},
                {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "context_length": 200000},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = models_payload

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_resp

        provider = OpenRouterProvider(credential_store=self.mock_store, http_client=mock_client)
        models = provider.get_available_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].id, "meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(models[0].context_window, 131072)
        self.assertEqual(models[1].id, "anthropic/claude-3.5-sonnet")


if __name__ == "__main__":
    unittest.main()

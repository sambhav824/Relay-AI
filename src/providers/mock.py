"""Configurable Mock AI Provider for reliable, offline unit testing."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Union

from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.providers.models import ModelInfo


class MockProvider(AIProvider):
    """Mock AI Provider for unit tests with configurable responses and error triggers."""

    provider_id: str = "mock"
    display_name: str = "Mock Provider"
    is_local: bool = True
    supports_streaming: bool = True

    def __init__(
        self,
        default_response: str = "Mock response",
        models: Optional[List[str]] = None,
        is_local: bool = True,
        simulate_error: Optional[Exception] = None,
    ):
        self.default_response = default_response
        self._models = models or ["mock-model-1", "mock-model-2"]
        self.is_local = is_local
        self.simulate_error = simulate_error
        self.recorded_calls: List[Dict[str, Any]] = []
        self._is_connected = True

    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = "mock-model-1",
        **kwargs,
    ) -> Iterator[StreamChunk]:
        self.recorded_calls.append({"messages": messages, "model": model, "kwargs": kwargs})

        if self.simulate_error:
            raise self.simulate_error

        if model not in self._models:
            raise ModelNotFoundError(f"Model '{model}' not found in mock models.", provider_id=self.provider_id)

        tokens = self.default_response.split(" ")
        for i, word in enumerate(tokens):
            is_final = i == len(tokens) - 1
            delta = word + (" " if not is_final else "")
            meta = None
            if is_final:
                meta = ResponseMetadata(
                    model=model,
                    provider_id=self.provider_id,
                    prompt_tokens=10,
                    completion_tokens=len(tokens),
                    total_tokens=10 + len(tokens),
                    finish_reason="stop",
                )
            yield StreamChunk(delta=delta, is_final=is_final, metadata=meta)

    def generate(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = "mock-model-1",
        **kwargs,
    ) -> str:
        self.recorded_calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        if self.simulate_error:
            raise self.simulate_error
        if model not in self._models:
            raise ModelNotFoundError(f"Model '{model}' not found in mock models.", provider_id=self.provider_id)
        return self.default_response

    def validate_connection(self) -> bool:
        if self.simulate_error and isinstance(self.simulate_error, (ProviderConnectionError, ProviderAuthenticationError)):
            return False
        return self._is_connected

    def get_available_models(self) -> List[ModelInfo]:
        if self.simulate_error and isinstance(self.simulate_error, ProviderConnectionError):
            raise self.simulate_error
        return [
            ModelInfo(
                id=m,
                name=f"Mock Model {m}",
                provider_id=self.provider_id,
                context_window=8192,
                is_local=self.is_local,
            )
            for m in self._models
        ]

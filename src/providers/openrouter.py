"""OpenRouter cloud AI provider adapter for Relay AI."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.providers.models import ModelInfo

logger = logging.getLogger("RelayAI.OpenRouter")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterProvider(AIProvider):
    """Cloud AI provider routing through the OpenRouter API."""

    provider_id: str = "openrouter"
    display_name: str = "OpenRouter"
    is_local: bool = False
    supports_streaming: bool = True

    def __init__(
        self,
        credential_store: Optional[CredentialStore] = None,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 60.0,
        http_client: Optional[httpx.Client] = None,
    ):
        self.credential_store = credential_store or CredentialStore()
        self._manual_api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external_client = http_client is not None
        self._client = http_client

    def _get_api_key(self) -> Optional[str]:
        if self._manual_api_key:
            return self._manual_api_key.strip()
        key = self.credential_store.get_api_key(self.provider_id)
        return key.strip() if key else None

    def _get_headers(self) -> Dict[str, str]:
        key = self._get_api_key()
        if not key:
            raise ProviderAuthenticationError(
                "OpenRouter API key not configured. Please set your key in Settings.",
                provider_id=self.provider_id,
            )
        return {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://relay.ai",
            "X-Title": "Relay AI Desktop",
            "Content-Type": "application/json",
        }

    def _map_http_error(self, status_code: int, response_text: str) -> ProviderError:
        text = response_text[:500]
        try:
            parsed = json.loads(response_text)
            err = parsed.get("error", {})
            if isinstance(err, dict):
                text = err.get("message") or text
        except Exception:
            pass

        if status_code in (401, 403):
            return ProviderAuthenticationError(f"Authentication failed: {text}", provider_id=self.provider_id)
        if status_code == 402:
            return ProviderBillingError(f"OpenRouter credit/billing limit exceeded: {text}", provider_id=self.provider_id)
        if status_code == 429:
            lower = text.lower()
            if "quota" in lower or "credit" in lower or "balance" in lower:
                return ProviderQuotaError(f"Quota exceeded: {text}", provider_id=self.provider_id)
            return ProviderRateLimitError(f"Rate limit exceeded: {text}", provider_id=self.provider_id)
        if status_code == 404:
            return ModelNotFoundError(f"Requested model not found: {text}", provider_id=self.provider_id)
        if status_code >= 500:
            return ProviderServerError(f"OpenRouter upstream server error ({status_code}): {text}", provider_id=self.provider_id)
        return ProviderError(f"OpenRouter API error ({status_code}): {text}", provider_id=self.provider_id)

    def _prepare_messages(
        self, messages: Union[str, List[Union[Message, Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Convert standard Message objects to OpenRouter message dicts (with vision support)."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        from src.core.images import encode_image_to_base64_data_uri

        normalized = self.normalize_messages(messages)
        formatted = []
        for m in normalized:
            imgs = getattr(m, "images", None) or (m.get("images") if isinstance(m, dict) else None)
            role_val = m.role.value if hasattr(m, "role") else m.get("role", "user")
            if imgs:
                content_parts = [{"type": "text", "text": m.content}]
                for img_path in imgs:
                    uri = encode_image_to_base64_data_uri(img_path)
                    if uri:
                        content_parts.append({"type": "image_url", "image_url": {"url": uri}})
                formatted.append({"role": role_val, "content": content_parts})
            else:
                formatted.append({"role": role_val, "content": m.content})
        return formatted

    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = DEFAULT_OPENROUTER_MODEL,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        headers = self._get_headers()
        formatted_messages = self._prepare_messages(messages)
        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
        }
        if "temperature" in kwargs:
            payload["temperature"] = float(kwargs["temperature"])
        if "max_tokens" in kwargs:
            payload["max_tokens"] = int(kwargs["max_tokens"])
        elif "max_output_tokens" in kwargs:
            payload["max_tokens"] = int(kwargs["max_output_tokens"])

        timeout = kwargs.get("timeout", self.timeout)
        endpoint = f"{self.base_url}/chat/completions"

        try:
            client = self._client or httpx.Client(timeout=timeout)
            should_close = self._client is None
            try:
                with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        content = response.read().decode("utf-8", errors="replace")
                        raise self._map_http_error(response.status_code, content)

                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            usage = data.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                                total_tokens = usage.get("total_tokens", total_tokens)

                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content_chunk = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason")

                            if content_chunk or finish_reason:
                                meta = None
                                if finish_reason or usage:
                                    meta = ResponseMetadata(
                                        model=model,
                                        provider_id=self.provider_id,
                                        finish_reason=str(finish_reason) if finish_reason else None,
                                        prompt_tokens=prompt_tokens or None,
                                        completion_tokens=completion_tokens or None,
                                        total_tokens=total_tokens or None,
                                    )
                                yield StreamChunk(
                                    delta=content_chunk,
                                    is_final=bool(finish_reason),
                                    metadata=meta,
                                )
            finally:
                if should_close:
                    client.close()

        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to OpenRouter at {self.base_url}: {exc}",
                provider_id=self.provider_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "OpenRouter request timed out.",
                provider_id=self.provider_id,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Unexpected OpenRouter error: {exc}", provider_id=self.provider_id) from exc

    def generate(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = DEFAULT_OPENROUTER_MODEL,
        **kwargs,
    ) -> str:
        chunks = []
        for chunk in self.stream(messages, model=model, **kwargs):
            if chunk.delta:
                chunks.append(chunk.delta)
        return "".join(chunks)

    def validate_connection(self) -> bool:
        if not self.has_credentials():
            return False
        try:
            models = self.get_available_models()
            return len(models) > 0
        except Exception as exc:
            logger.warning("OpenRouter validation failed: %s", exc)
            return False

    def get_available_models(self) -> List[ModelInfo]:
        headers = self._get_headers()
        endpoint = f"{self.base_url}/models"
        client = self._client or httpx.Client(timeout=10.0)
        should_close = self._client is None
        try:
            resp = client.get(endpoint, headers=headers)
            if resp.status_code != 200:
                raise self._map_http_error(resp.status_code, resp.text)
            data = resp.json()
            models_data = data.get("data", [])
            results: List[ModelInfo] = []
            for item in models_data:
                mid = item.get("id")
                name = item.get("name") or mid
                if mid:
                    results.append(
                        ModelInfo(
                            id=mid,
                            name=name,
                            provider_id=self.provider_id,
                            context_window=item.get("context_length", 4096),
                            is_local=False,
                        )
                    )
            return results
        except httpx.RequestError as exc:
            raise ProviderConnectionError(f"Failed to fetch OpenRouter models: {exc}", provider_id=self.provider_id) from exc
        finally:
            if should_close:
                client.close()

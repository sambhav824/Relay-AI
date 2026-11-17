import json
from typing import Any, Dict, Iterator, List, Union

import httpx

from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from src.providers.models import ModelInfo

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{DEFAULT_OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL = f"{DEFAULT_OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{DEFAULT_OLLAMA_HOST}/api/tags"


class OllamaProvider(AIProvider):
    """Local Ollama AI model provider supporting streaming and chat."""

    provider_id: str = "ollama"
    display_name: str = "Ollama (Local)"
    is_local: bool = True
    supports_streaming: bool = True

    def __init__(self, base_url: str = DEFAULT_OLLAMA_HOST):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"

    def _format_messages(self, normalized: List[Message]) -> List[Dict[str, Any]]:
        """Format normalized messages for Ollama API with vision base64 support."""
        from src.core.images import encode_image_to_raw_base64
        results = []
        for m in normalized:
            d = {"role": m.role.value, "content": m.content}
            imgs = getattr(m, "images", None)
            if imgs:
                raw_b64s = []
                for p in imgs:
                    raw = encode_image_to_raw_base64(p)
                    if raw:
                        raw_b64s.append(raw)
                if raw_b64s:
                    d["images"] = raw_b64s
            results.append(d)
        return results

    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = "llama3.2:3b",
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Yield response chunks in real time from Ollama's streaming chat endpoint."""
        normalized = self.normalize_messages(messages)
        payload = {
            "model": model,
            "messages": self._format_messages(normalized),
            "stream": True,
        }
        max_output_tokens = kwargs.get("max_output_tokens")
        if max_output_tokens is not None:
            payload["options"] = {"num_predict": max(1, int(max_output_tokens))}

        try:
            with httpx.stream(
                "POST",
                self.chat_url,
                json=payload,
                timeout=httpx.Timeout(180.0, connect=5.0),
            ) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(
                        f"Model '{model}' not found on local Ollama. Run 'ollama pull {model}' to install.",
                        provider_id=self.provider_id,
                    )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk_text = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    meta = None
                    if done:
                        meta = ResponseMetadata(
                            model=model,
                            provider_id=self.provider_id,
                            total_tokens=data.get("eval_count"),
                            prompt_tokens=data.get("prompt_eval_count"),
                            finish_reason="stop",
                        )

                    if chunk_text or done:
                        yield StreamChunk(delta=chunk_text, is_final=done, metadata=meta)

                    if done:
                        break

        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.",
                provider_id=self.provider_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Ollama took too long to respond. Please try again.",
                provider_id=self.provider_id,
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise ProviderError(
                f"Ollama API error ({exc.response.status_code}): {detail}",
                provider_id=self.provider_id,
            ) from exc
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(str(exc), provider_id=self.provider_id) from exc

    def generate(
        self,
        messages: Union[str, List[Union[Message, Dict[str, Any]]]],
        model: str = "llama3.2:3b",
        **kwargs,
    ) -> str:
        """Call the local Ollama endpoint and return the complete response text."""
        if isinstance(messages, str):
            payload = {"model": model, "prompt": messages, "stream": False}
            endpoint = self.generate_url
        else:
            normalized = self.normalize_messages(messages)
            payload = {
                "model": model,
                "messages": self._format_messages(normalized),
                "stream": False,
            }
            endpoint = self.chat_url
        max_output_tokens = kwargs.get("max_output_tokens")
        if max_output_tokens is not None:
            payload["options"] = {"num_predict": max(1, int(max_output_tokens))}

        try:
            response = httpx.post(
                endpoint,
                json=payload,
                timeout=httpx.Timeout(180.0, connect=5.0),
            )
            if response.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{model}' not found on local Ollama.",
                    provider_id=self.provider_id,
                )
            response.raise_for_status()
            data = response.json()

            if "message" in data:
                answer = data.get("message", {}).get("content", "").strip()
            else:
                answer = data.get("response", "").strip()

            if not answer:
                raise ProviderError("Ollama returned an empty response.", provider_id=self.provider_id)
            return answer

        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.",
                provider_id=self.provider_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Ollama took too long to respond. Please try again.",
                provider_id=self.provider_id,
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise ProviderError(
                f"Ollama API error ({exc.response.status_code}): {detail}",
                provider_id=self.provider_id,
            ) from exc

    def validate_connection(self) -> bool:
        """Verify reachability of the local Ollama instance."""
        try:
            response = httpx.get(self.tags_url, timeout=httpx.Timeout(2.0, connect=2.0))
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[ModelInfo]:
        """Query installed models from Ollama."""
        try:
            response = httpx.get(self.tags_url, timeout=httpx.Timeout(3.0, connect=2.0))
            response.raise_for_status()
            data = response.json()
            models_data = data.get("models", [])
            results = []
            for item in models_data:
                model_name = item.get("name") or item.get("model", "")
                if model_name:
                    results.append(
                        ModelInfo(
                            id=model_name,
                            name=model_name,
                            provider_id=self.provider_id,
                            is_local=True,
                            description=f"Size: {item.get('size', 0) // (1024*1024)} MB",
                        )
                    )
            if results:
                return results
        except Exception:
            pass

        # Fallback to default expected model if query fails or is empty
        return [
            ModelInfo(
                id="llama3.2:3b",
                name="llama3.2:3b",
                provider_id=self.provider_id,
                is_local=True,
                description="Default local model",
            )
        ]


# Default singleton instance
_default_ollama_provider = OllamaProvider()


# Backward-compatible module-level functions
def stream_chat(messages, model: str = "llama3.2:3b") -> Iterator[str]:
    """Legacy helper: yields string chunks from Ollama stream."""
    for chunk in _default_ollama_provider.stream(messages, model):
        if chunk.delta:
            yield chunk.delta


def generate(prompt: Union[str, List[Any]], model: str = "llama3.2:3b") -> str:
    """Legacy helper: generates full response from Ollama."""
    return _default_ollama_provider.generate(prompt, model)

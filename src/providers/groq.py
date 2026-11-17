import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.providers.models import ModelInfo

logger = logging.getLogger("RelayAI.Groq")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqProvider(AIProvider):
    """Ultra-low latency cloud AI provider using the official Groq Python SDK."""

    provider_id: str = "groq"
    display_name: str = "Groq"
    is_local: bool = False
    supports_streaming: bool = True

    def __init__(
        self,
        credential_store: Optional[CredentialStore] = None,
        api_key: Optional[str] = None,
    ):
        self.credential_store = credential_store or CredentialStore()
        self._manual_api_key = api_key

    def _get_api_key(self) -> Optional[str]:
        """Fetch API key from manual override, server environment, or CredentialStore."""
        if self._manual_api_key and str(self._manual_api_key).strip():
            return str(self._manual_api_key).strip()

        # Check server environment variables directly (Render, Linux containers, Cloud)
        import os
        for var_name in ("GROQ_API_KEY", "GROQ_KEY", "GROQ_API_TOKEN"):
            val = os.environ.get(var_name)
            if val and val.strip():
                return val.strip()

        # Fallback to credential store (which checks env vars and OS keyring)
        if self.credential_store:
            key = self.credential_store.get_api_key(self.provider_id)
            if key and key.strip():
                return key.strip()

        return None

    def has_credentials(self, credential_store: Optional[Any] = None) -> bool:
        """Check whether valid Groq credentials exist without exposing the key."""
        key = self._get_api_key()
        if key:
            return True
        store = credential_store or self.credential_store
        return bool(store and store.has_api_key(self.provider_id))

    def _get_client(self):
        """Construct the official groq.Groq client."""
        api_key = self._get_api_key()
        if not api_key:
            raise ProviderAuthenticationError(
                "Groq API key not found. Please configure GROQ_API_KEY in server environment variables or via application Settings.",
                provider_id=self.provider_id,
            )
        try:
            import groq
            return groq.Groq(api_key=api_key)
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize Groq client: {exc}",
                provider_id=self.provider_id,
            ) from exc

    def _prepare_messages(
        self, messages: Union[str, List[Union[Message, Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Convert standard Message objects to OpenAI/Groq compatible message dicts (with vision support)."""
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
        model: str = DEFAULT_GROQ_MODEL,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream response chunks in real time from Groq."""
        client = self._get_client()
        formatted_messages = self._prepare_messages(messages)

        try:
            response_stream = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=True,
            )

            for chunk in response_stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta_content = getattr(choice.delta, "content", "") or ""
                finish_reason = getattr(choice, "finish_reason", None)

                meta = ResponseMetadata(
                    model=model,
                    provider_id=self.provider_id,
                    finish_reason=str(finish_reason) if finish_reason else None,
                )

                if delta_content:
                    yield StreamChunk(delta=delta_content, is_final=False, metadata=meta)

            yield StreamChunk(
                delta="",
                is_final=True,
                metadata=ResponseMetadata(model=model, provider_id=self.provider_id, finish_reason="stop"),
            )

        except Exception as exc:
            self._handle_exception(exc, model)

    def generate(
        self,
        messages: Union[str, List[Union[Message, Dict[str, Any]]]],
        model: str = DEFAULT_GROQ_MODEL,
        **kwargs,
    ) -> str:
        """Generate a complete text response from Groq."""
        client = self._get_client()
        formatted_messages = self._prepare_messages(messages)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=False,
            )
            if not response.choices:
                raise ProviderError("Groq returned an empty response.", provider_id=self.provider_id)
            content = response.choices[0].message.content or ""
            return content
        except Exception as exc:
            self._handle_exception(exc, model)

    def validate_connection(self) -> bool:
        """Validate Groq API key and reachability."""
        try:
            client = self._get_client()
            _ = client.models.list()
            return True
        except Exception:
            return False

    def get_available_models(self) -> List[ModelInfo]:
        """Fetch available text generation models from Groq."""
        fallback_models = [
            ModelInfo(id="openai/gpt-oss-120b", name="GPT-OSS 120B", provider_id=self.provider_id, is_local=False, description="Flagship high-performance reasoning model on Groq"),
            ModelInfo(id="qwen/qwen3.8-27b", name="Qwen 3.8 27B", provider_id=self.provider_id, is_local=False, description="Fast, capable open-weights chat model"),
            ModelInfo(id="openai/gpt-oss-20b", name="GPT-OSS 20B", provider_id=self.provider_id, is_local=False, description="High-speed efficient generation model"),
        ]

        try:
            client = self._get_client()
            models_list = client.models.list()
            results = []
            for m in models_list.data:
                mid = m.id.lower()
                # Exclude audio/whisper and guard models from chat selector
                if "whisper" in mid or "guard" in mid:
                    continue
                display_name = m.id.split("/")[-1].replace("-", " ").title()
                results.append(
                    ModelInfo(
                        id=m.id,
                        name=f"{display_name} ({m.id})",
                        provider_id=self.provider_id,
                        is_local=False,
                        description=f"Active Groq model: {m.id}",
                    )
                )
            if results:
                # Rank 120b and qwen first if present
                def sort_key(model: ModelInfo):
                    if "120b" in model.id:
                        return 0
                    if "qwen" in model.id:
                        return 1
                    if "20b" in model.id:
                        return 2
                    return 3
                results.sort(key=sort_key)
                return results
        except Exception:
            pass

        return fallback_models

    def _handle_exception(self, exc: Exception, model_name: str) -> None:
        """Map Groq SDK exceptions to standardized ProviderError hierarchy."""
        err_str = str(exc)
        status_code = getattr(exc, "status_code", None)

        # Check for authentication failures
        if (
            status_code in (401, 403)
            or "invalid_api_key" in err_str
            or "AuthenticationError" in type(exc).__name__
            or "PermissionDeniedError" in type(exc).__name__
        ):
            raise ProviderAuthenticationError(
                "Invalid or unauthorized Groq API key. Please check your key in Settings.",
                provider_id=self.provider_id,
            ) from exc

        # Check for model not found
        if status_code == 404 or "model_not_found" in err_str or "NotFoundError" in type(exc).__name__:
            raise ModelNotFoundError(
                f"Groq model '{model_name}' was not found or is unavailable: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        # Check for rate limit / quota
        if status_code in (429, 503) or "rate_limit_exceeded" in err_str or "RateLimitError" in type(exc).__name__:
            raise ProviderRateLimitError(
                "Groq rate limit exceeded or quota exhausted. Please try again shortly.",
                provider_id=self.provider_id,
            ) from exc

        # Check for network connection / timeout
        if (
            "APIConnectionError" in type(exc).__name__
            or "APITimeoutError" in type(exc).__name__
            or "timed out" in err_str.lower()
        ):
            raise ProviderConnectionError(
                f"Network error connecting to Groq API: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        raise ProviderError(f"Groq API error: {err_str}", provider_id=self.provider_id) from exc


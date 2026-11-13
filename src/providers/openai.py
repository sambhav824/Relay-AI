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

logger = logging.getLogger("RelayAI.OpenAI")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIProvider(AIProvider):
    """Cloud AI provider using the official OpenAI Python SDK."""

    provider_id: str = "openai"
    display_name: str = "OpenAI"
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
        """Fetch API key from manual override or Windows Credential Manager."""
        if self._manual_api_key:
            return self._manual_api_key.strip()
        key = self.credential_store.get_api_key(self.provider_id)
        return key.strip() if key else None

    def _get_client(self):
        """Construct the official openai.OpenAI client."""
        api_key = self._get_api_key()
        if not api_key:
            raise ProviderAuthenticationError(
                "OpenAI API key not found. Please configure your key in Settings or Windows Credential Manager.",
                provider_id=self.provider_id,
            )
        try:
            import openai
            return openai.OpenAI(api_key=api_key)
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize OpenAI client: {exc}",
                provider_id=self.provider_id,
            ) from exc

    def _prepare_messages(
        self, messages: Union[str, List[Union[Message, Dict[str, Any]]]]
    ) -> List[Dict[str, str]]:
        """Convert standard Message objects to OpenAI compatible message dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        normalized = self.normalize_messages(messages)
        return [{"role": m.role.value, "content": m.content} for m in normalized]

    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream response chunks in real time from OpenAI."""
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
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> str:
        """Generate a complete text response from OpenAI."""
        client = self._get_client()
        formatted_messages = self._prepare_messages(messages)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=False,
            )
            if not response.choices:
                raise ProviderError("OpenAI returned an empty response.", provider_id=self.provider_id)
            content = response.choices[0].message.content or ""
            return content
        except Exception as exc:
            self._handle_exception(exc, model)

    def validate_connection(self) -> bool:
        """Validate OpenAI API key and reachability via models list endpoint."""
        try:
            client = self._get_client()
            _ = client.models.list()
            return True
        except Exception:
            return False

    def get_available_models(self) -> List[ModelInfo]:
        """Fetch available text generation models from OpenAI or return verified catalogue."""
        fallback_models = [
            ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini (Recommended)", provider_id=self.provider_id, is_local=False, description="Fast, affordable, high-intelligence model"),
            ModelInfo(id="gpt-4o", name="GPT-4o", provider_id=self.provider_id, is_local=False, description="Flagship high-intelligence multimodal model"),
            ModelInfo(id="gpt-4.1-mini", name="GPT-4.1 Mini", provider_id=self.provider_id, is_local=False, description="Next-generation efficient chat model"),
            ModelInfo(id="gpt-4.1", name="GPT-4.1", provider_id=self.provider_id, is_local=False, description="Next-generation flagship reasoning model"),
            ModelInfo(id="o3", name="O3", provider_id=self.provider_id, is_local=False, description="Advanced deep reasoning model"),
        ]

        try:
            client = self._get_client()
            models_list = client.models.list()
            discovered = []
            excluded_keywords = ["audio", "realtime", "vision", "tts", "transcribe", "moderation", "embedding", "dall-e", "whisper", "instruct", "search-api"]

            for m in models_list.data:
                mid = m.id.lower()
                # Check chat-capable models
                if any(k in mid for k in ["gpt-4", "gpt-5", "o1", "o3", "chatgpt"]):
                    if not any(ex in mid for ex in excluded_keywords):
                        display_name = m.id.replace("-", " ").title()
                        discovered.append(
                            ModelInfo(
                                id=m.id,
                                name=f"{display_name} ({m.id})",
                                provider_id=self.provider_id,
                                is_local=False,
                                description=f"OpenAI model: {m.id}",
                            )
                        )

            if discovered:
                # Rank 4o-mini, 4o, 4.1 first
                def sort_key(model: ModelInfo):
                    mid = model.id.lower()
                    if mid == "gpt-4o-mini":
                        return 0
                    if mid == "gpt-4o":
                        return 1
                    if "4.1-mini" in mid:
                        return 2
                    if "4.1" in mid:
                        return 3
                    if "o3" in mid:
                        return 4
                    return 5

                discovered.sort(key=sort_key)
                return discovered
        except Exception:
            pass

        return fallback_models

    def _handle_exception(self, exc: Exception, model_name: str) -> None:
        """Map OpenAI SDK exceptions to standardized ProviderError hierarchy."""
        err_str = str(exc)
        status_code = getattr(exc, "status_code", None)

        if (
            status_code in (401, 403)
            or "invalid_api_key" in err_str.lower()
            or "AuthenticationError" in type(exc).__name__
            or "PermissionDeniedError" in type(exc).__name__
        ):
            raise ProviderAuthenticationError(
                "Invalid or unauthorized OpenAI API key. Please check your credentials in Settings.",
                provider_id=self.provider_id,
            ) from exc

        if status_code == 404 or "model_not_found" in err_str.lower() or "NotFoundError" in type(exc).__name__:
            raise ModelNotFoundError(
                f"OpenAI model '{model_name}' was not found or is deprecated: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        if (
            status_code == 429
            or "rate_limit_exceeded" in err_str.lower()
            or "RateLimitError" in type(exc).__name__
            or "insufficient_quota" in err_str.lower()
        ):
            if "quota" in err_str.lower() or "billing" in err_str.lower() or "credits" in err_str.lower():
                raise ProviderRateLimitError(
                    "OpenAI account has no credits remaining. Add credits to your account at https://platform.openai.com/settings/organization/billing/.",
                    provider_id=self.provider_id,
                ) from exc
            raise ProviderRateLimitError(
                f"OpenAI rate limit exceeded: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        if (
            "APIConnectionError" in type(exc).__name__
            or "APITimeoutError" in type(exc).__name__
            or "timed out" in err_str.lower()
        ):
            raise ProviderConnectionError(
                f"Network error connecting to OpenAI API: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        raise ProviderError(f"OpenAI API error: {err_str}", provider_id=self.provider_id) from exc


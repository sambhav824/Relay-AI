import logging
import time
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
    ProviderBillingError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.providers.cost_protection import UsageLimitExceeded, UsageLimits, UsageTracker
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
        usage_tracker: Optional[UsageTracker] = None,
    ):
        self.credential_store = credential_store or CredentialStore()
        self._manual_api_key = api_key
        self.usage_tracker = usage_tracker or UsageTracker()
        self.usage_limits = UsageLimits()
        self.user_id_provider = lambda: "local"
        self.fallback_provider: Optional[AIProvider] = None
        self.automatic_fallback_enabled = True

    def configure_usage(self, limits: UsageLimits, user_id_provider=None) -> None:
        self.usage_limits = UsageLimits.from_mapping(vars(limits))
        if user_id_provider:
            self.user_id_provider = user_id_provider

    def configure_fallback(self, provider: Optional[AIProvider], enabled: bool = True) -> None:
        self.fallback_provider = provider
        self.automatic_fallback_enabled = bool(enabled)

    def _get_api_key(self) -> Optional[str]:
        """Fetch API key from manual override, server environment, or CredentialStore."""
        if self._manual_api_key and str(self._manual_api_key).strip():
            return str(self._manual_api_key).strip()

        # Check server environment variables directly (Render, Linux containers, Cloud)
        import os
        for var_name in ("OPENAI_API_KEY", "OPENAI_KEY"):
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
        """Check whether valid OpenAI credentials exist without exposing the key."""
        key = self._get_api_key()
        if key:
            return True
        store = credential_store or self.credential_store
        return bool(store and store.has_api_key(self.provider_id))

    def _get_client(self):
        """Construct the official openai.OpenAI client."""
        api_key = self._get_api_key()
        if not api_key:
            raise ProviderAuthenticationError(
                "OpenAI API key not found. Please configure OPENAI_API_KEY in server environment variables or via application Settings.",
                provider_id=self.provider_id,
            )
        try:
            import openai
            return openai.OpenAI(api_key=api_key)
        except Exception as exc:
            raise ProviderConnectionError(
                "Failed to initialize the OpenAI client.",
                provider_id=self.provider_id,
            ) from exc

    def _prepare_messages(
        self, messages: Union[str, List[Union[Message, Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Convert standard Message objects to OpenAI compatible message dicts (with vision support)."""
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
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream response chunks in real time from OpenAI."""
        formatted_messages = self._prepare_messages(messages)
        output_limit = self._output_limit(kwargs)
        try:
            request_id = self.usage_tracker.begin_request(
                self.user_id_provider(), model, formatted_messages, output_limit, self.usage_limits
            )
        except UsageLimitExceeded as exc:
            raise ProviderError(str(exc), provider_id=self.provider_id) from exc

        input_tokens = output_tokens = None
        sent_content = False
        try:
            for attempt in range(3):
                try:
                    client = self._get_client()
                    response_stream = client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        stream=True,
                        max_completion_tokens=output_limit,
                        stream_options={"include_usage": True},
                    )

                    finish_reason = "stop"
                    for chunk in response_stream:
                        usage = getattr(chunk, "usage", None)
                        if usage:
                            input_tokens = self._usage_count(getattr(usage, "prompt_tokens", None))
                            output_tokens = self._usage_count(getattr(usage, "completion_tokens", None))
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        delta_content = getattr(choice.delta, "content", "") or ""
                        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                        if delta_content:
                            sent_content = True
                            yield StreamChunk(
                                delta=delta_content,
                                is_final=False,
                                metadata=ResponseMetadata(model=model, provider_id=self.provider_id),
                            )

                    yield StreamChunk(
                        delta="",
                        is_final=True,
                        metadata=ResponseMetadata(
                            model=model,
                            provider_id=self.provider_id,
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=(input_tokens + output_tokens)
                            if input_tokens is not None and output_tokens is not None
                            else None,
                            finish_reason=str(finish_reason),
                        ),
                    )
                    return
                except Exception as exc:
                    try:
                        self._handle_exception(exc, model)
                    except ProviderError as error:
                        if isinstance(error, (ProviderQuotaError, ProviderBillingError)):
                            yield from self._fallback_stream(messages, error, output_limit)
                            return
                        retryable = isinstance(
                            error,
                            (
                                ProviderRateLimitError,
                                ProviderConnectionError,
                                ProviderTimeoutError,
                                ProviderServerError,
                            ),
                        )
                        if retryable and attempt < 2 and not sent_content:
                            time.sleep(0.25 * (2**attempt))
                            continue
                        raise error from exc
        finally:
            self.usage_tracker.complete_request(request_id, input_tokens, output_tokens, self.usage_limits)

    def generate(
        self,
        messages: Union[str, List[Union[Message, Dict[str, Any]]]],
        model: str = DEFAULT_OPENAI_MODEL,
        **kwargs,
    ) -> str:
        """Generate a complete text response from OpenAI."""
        formatted_messages = self._prepare_messages(messages)
        output_limit = self._output_limit(kwargs)
        try:
            request_id = self.usage_tracker.begin_request(
                self.user_id_provider(), model, formatted_messages, output_limit, self.usage_limits
            )
        except UsageLimitExceeded as exc:
            raise ProviderError(str(exc), provider_id=self.provider_id) from exc
        input_tokens = output_tokens = None
        try:
            for attempt in range(3):
                try:
                    client = self._get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        stream=False,
                        max_completion_tokens=output_limit,
                    )
                    if not response.choices:
                        raise ProviderError("OpenAI returned an empty response.", provider_id=self.provider_id)
                    usage = getattr(response, "usage", None)
                    if usage:
                        input_tokens = self._usage_count(getattr(usage, "prompt_tokens", None))
                        output_tokens = self._usage_count(getattr(usage, "completion_tokens", None))
                    return response.choices[0].message.content or ""
                except ProviderError as error:
                    if isinstance(error, (ProviderQuotaError, ProviderBillingError)):
                        return self._fallback_generate(messages, error, output_limit)
                    retryable = isinstance(
                        error,
                        (
                            ProviderRateLimitError,
                            ProviderConnectionError,
                            ProviderTimeoutError,
                            ProviderServerError,
                        ),
                    )
                    if retryable and attempt < 2:
                        time.sleep(0.25 * (2**attempt))
                        continue
                    raise
                except Exception as exc:
                    try:
                        self._handle_exception(exc, model)
                    except ProviderError as error:
                        if isinstance(error, (ProviderQuotaError, ProviderBillingError)):
                            return self._fallback_generate(messages, error, output_limit)
                        retryable = isinstance(
                            error,
                            (
                                ProviderRateLimitError,
                                ProviderConnectionError,
                                ProviderTimeoutError,
                                ProviderServerError,
                            ),
                        )
                        if retryable and attempt < 2:
                            time.sleep(0.25 * (2**attempt))
                            continue
                        raise error from exc
        finally:
            self.usage_tracker.complete_request(request_id, input_tokens, output_tokens, self.usage_limits)

    def _output_limit(self, kwargs: Dict[str, Any]) -> int:
        value = kwargs.get("max_output_tokens", kwargs.get("max_completion_tokens", kwargs.get("max_tokens")))
        try:
            requested = max(1, int(value)) if value is not None else self.usage_limits.max_output_tokens
        except (TypeError, ValueError):
            requested = self.usage_limits.max_output_tokens
        return min(requested, self.usage_limits.max_output_tokens)

    @staticmethod
    def _usage_count(value) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _fallback_stream(self, messages, cause: ProviderError, output_limit: int) -> Iterator[StreamChunk]:
        fallback = self.fallback_provider
        if not self.automatic_fallback_enabled or fallback is None:
            raise cause
        fallback_model = "llama3.2:3b"
        try:
            yield StreamChunk(
                delta="",
                metadata=ResponseMetadata(
                    model=fallback_model,
                    provider_id="ollama",
                    extra={"automatic_fallback_from": self.provider_id, "fallback_reason": type(cause).__name__},
                ),
            )
            yield from fallback.stream(messages, fallback_model, max_output_tokens=output_limit)
        except Exception as fallback_error:
            raise ProviderError(
                "OpenAI is unavailable because of a quota or billing limit, and the local Ollama fallback "
                f"could not complete the request ({type(fallback_error).__name__}).",
                provider_id=self.provider_id,
            ) from fallback_error

    def _fallback_generate(self, messages, cause: ProviderError, output_limit: int) -> str:
        fallback = self.fallback_provider
        if not self.automatic_fallback_enabled or fallback is None:
            raise cause
        try:
            return fallback.generate(messages, "llama3.2:3b", max_output_tokens=output_limit)
        except Exception as fallback_error:
            raise ProviderError(
                "OpenAI is unavailable because of a quota or billing limit, and the local Ollama fallback "
                f"could not complete the request ({type(fallback_error).__name__}).",
                provider_id=self.provider_id,
            ) from fallback_error

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
        body = getattr(exc, "body", None)
        error_body = body.get("error", body) if isinstance(body, dict) else {}
        code = str(error_body.get("code", "")).lower() if isinstance(error_body, dict) else ""
        error_type = str(error_body.get("type", "")).lower() if isinstance(error_body, dict) else ""
        detail = f"{code} {error_type} {err_str}".lower()

        if status_code == 401 or "invalid_api_key" in detail or "authenticationerror" in type(exc).__name__.lower():
            raise ProviderAuthenticationError(
                "OpenAI authentication failed. Check the API key stored in Settings.",
                provider_id=self.provider_id,
            ) from exc

        if any(term in detail for term in ("insufficient_quota", "quota", "credit")):
            raise ProviderQuotaError(
                "OpenAI usage quota or credits are exhausted.",
                provider_id=self.provider_id,
            ) from exc
        if any(term in detail for term in ("billing", "spend_limit", "hard_limit", "billing_hard_limit")):
            raise ProviderBillingError(
                "OpenAI billing or spending limit has been reached.",
                provider_id=self.provider_id,
            ) from exc
        if status_code == 403 or "permissiondenied" in type(exc).__name__.lower():
            raise ProviderPermissionError(
                "The OpenAI account or API key does not have permission to use this model.",
                provider_id=self.provider_id,
            ) from exc

        if status_code == 404 or "model_not_found" in detail or "notfounderror" in type(exc).__name__.lower():
            raise ModelNotFoundError(
                f"OpenAI model '{model_name}' was not found or is unavailable.",
                provider_id=self.provider_id,
            ) from exc

        if status_code == 429 or "rate_limit" in detail or "ratelimiterror" in type(exc).__name__.lower():
            raise ProviderRateLimitError(
                "OpenAI temporarily rate-limited this request.",
                provider_id=self.provider_id,
            ) from exc

        if status_code is not None and 500 <= status_code <= 599:
            raise ProviderServerError(
                f"OpenAI server error (HTTP {status_code}).",
                provider_id=self.provider_id,
            ) from exc

        if "apitimeout" in type(exc).__name__.lower() or "timed out" in err_str.lower():
            raise ProviderTimeoutError("OpenAI request timed out.", provider_id=self.provider_id) from exc
        if "apiconnection" in type(exc).__name__.lower() or "connection" in type(exc).__name__.lower():
            raise ProviderConnectionError(
                "A network error prevented connecting to OpenAI.",
                provider_id=self.provider_id,
            ) from exc
        raise ProviderError("OpenAI request failed. Check provider status and request settings.", provider_id=self.provider_id) from exc

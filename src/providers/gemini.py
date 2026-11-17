import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from src.core.events import ResponseMetadata, StreamChunk
from src.core.message import Message, MessageRole
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

logger = logging.getLogger("RelayAI.Gemini")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class GeminiProvider(AIProvider):
    """Google Gemini cloud AI provider using the official Google GenAI SDK."""

    provider_id: str = "gemini"
    display_name: str = "Google Gemini"
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
        """Construct the official google-genai Client."""
        api_key = self._get_api_key()
        if not api_key:
            raise ProviderAuthenticationError(
                "Gemini API key not found. Please configure your key in Settings or Windows Credential Manager.",
                provider_id=self.provider_id,
            )
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to initialize Gemini client: {exc}",
                provider_id=self.provider_id,
            ) from exc

    def _prepare_contents(
        self, messages: List[Union[Message, Dict[str, Any]]]
    ):
        """Convert standard Message objects to Gemini Contents and System Instruction."""
        from google.genai import types
        from src.core.images import read_image_bytes_and_mime

        normalized = self.normalize_messages(messages)
        contents = []
        system_instruction = None

        for msg in normalized:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                parts = [types.Part.from_text(text=msg.content)]
                imgs = getattr(msg, "images", None) or (msg.get("images") if isinstance(msg, dict) else None)
                if imgs:
                    for img_path in imgs:
                        img_bytes, mime = read_image_bytes_and_mime(img_path)
                        if img_bytes:
                            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                contents.append(
                    types.Content(
                        role="user",
                        parts=parts,
                    )
                )
            elif msg.role == MessageRole.ASSISTANT:
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        return contents, config

    def stream(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        model: str = DEFAULT_GEMINI_MODEL,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream response chunks in real time from Google Gemini."""
        client = self._get_client()
        contents, config = self._prepare_contents(messages)
        target_model = model.removeprefix("models/")

        try:
            response_stream = client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config,
            )

            for chunk in response_stream:
                text = getattr(chunk, "text", "") or ""
                finish_reason = None
                if getattr(chunk, "candidates", None) and len(chunk.candidates) > 0:
                    finish_reason = getattr(chunk.candidates[0], "finish_reason", None)

                meta = ResponseMetadata(
                    model=target_model,
                    provider_id=self.provider_id,
                    finish_reason=str(finish_reason) if finish_reason else None,
                )

                if text:
                    yield StreamChunk(delta=text, is_final=False, metadata=meta)

            yield StreamChunk(
                delta="",
                is_final=True,
                metadata=ResponseMetadata(model=target_model, provider_id=self.provider_id, finish_reason="stop"),
            )

        except Exception as exc:
            self._handle_exception(exc, target_model)

    def generate(
        self,
        messages: Union[str, List[Union[Message, Dict[str, Any]]]],
        model: str = DEFAULT_GEMINI_MODEL,
        **kwargs,
    ) -> str:
        """Generate a complete text response from Google Gemini."""
        client = self._get_client()
        if isinstance(messages, str):
            contents = messages
            config = None
        else:
            contents, config = self._prepare_contents(messages)

        target_model = model.removeprefix("models/")
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", "")
            if not text:
                raise ProviderError("Gemini returned an empty response.", provider_id=self.provider_id)
            return text
        except Exception as exc:
            self._handle_exception(exc, target_model)

    def validate_connection(self) -> bool:
        """Validate Gemini API key and reachability."""
        try:
            client = self._get_client()
            models_iter = client.models.list(config={"page_size": 1})
            _ = next(iter(models_iter), None)
            return True
        except Exception:
            return False

    def get_available_models(self) -> List[ModelInfo]:
        """Fetch available text generation models from Gemini."""
        fallback_models = [
            ModelInfo(id="gemini-3.5-flash", name="Gemini 3.5 Flash", provider_id=self.provider_id, is_local=False, description="Fast, state-of-the-art flagship model"),
            ModelInfo(id="gemini-3.8-flash", name="Gemini 3.8 Flash", provider_id=self.provider_id, is_local=False, description="Latest Gemini Flash generation model"),
            ModelInfo(id="gemini-3.5-flash-lite", name="Gemini 3.5 Flash Lite", provider_id=self.provider_id, is_local=False, description="High-throughput, ultra-fast lightweight model"),
        ]

        try:
            client = self._get_client()
            results = []
            for m in client.models.list(config={"page_size": 100}):
                actions = getattr(m, "supported_actions", []) or []
                if "generateContent" in actions:
                    raw_id = m.name.removeprefix("models/")
                    # Filter out retired models or image-only models
                    if ("gemini-3" in raw_id or "gemini-4" in raw_id or "gemma-4" in raw_id) and "-image" not in raw_id:
                        display = getattr(m, "display_name", "") or raw_id
                        results.append(
                            ModelInfo(
                                id=raw_id,
                                name=display,
                                provider_id=self.provider_id,
                                is_local=False,
                                description=getattr(m, "description", "") or "",
                            )
                        )
            if results:
                # Rank 3.5-flash first, then 3.8-flash
                def sort_key(model: ModelInfo):
                    if model.id == "gemini-3.5-flash":
                        return 0
                    if model.id == "gemini-3.8-flash":
                        return 1
                    if "flash-lite" in model.id:
                        return 2
                    return 3
                results.sort(key=sort_key)
                return results
        except Exception:
            pass

        return fallback_models

    def _handle_exception(self, exc: Exception, model_name: str) -> None:
        """Map SDK errors to standardized ProviderError hierarchy."""
        err_str = str(exc)
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None) or code

        if status_code in (401, 403) or "API_KEY_INVALID" in err_str or "PERMISSION_DENIED" in err_str:
            raise ProviderAuthenticationError(
                "Invalid or unauthorized Gemini API key. Please verify your credentials in Settings.",
                provider_id=self.provider_id,
            ) from exc

        if status_code == 404 or "NOT_FOUND" in err_str:
            raise ModelNotFoundError(
                f"Gemini model '{model_name}' is not found or no longer supported. Please select another model.",
                provider_id=self.provider_id,
            ) from exc

        if status_code in (429, 503) or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
            raise ProviderRateLimitError(
                "Gemini service is temporarily busy or quota is exhausted. Please select another model (e.g. gemini-3.5-flash) or retry.",
                provider_id=self.provider_id,
            ) from exc

        if "ConnectError" in str(type(exc)) or "timed out" in err_str.lower():
            raise ProviderConnectionError(
                f"Network error connecting to Google Gemini API: {err_str}",
                provider_id=self.provider_id,
            ) from exc

        raise ProviderError(f"Gemini error: {err_str}", provider_id=self.provider_id) from exc

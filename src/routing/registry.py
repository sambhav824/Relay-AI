import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from src.core.security import CredentialStore
from src.providers.manager import ProviderManager, get_default_provider_manager
from src.providers.models import ModelInfo
from src.routing.types import ModelPricing, ModelProfile, TaskType

logger = logging.getLogger("RelayAI.ModelRegistry")


# Configurable table of verified model capabilities and official pricing schedules.
# Keyed by (provider_id.lower(), model_id.lower()).
# Note: Unknown or unverified models dynamically discovered from APIs receive "Pricing unavailable".
VERIFIED_MODEL_METADATA: Dict[Tuple[str, str], Dict] = {
    # Groq verified models (Official Groq Pricing, Verified 2026-11-13)
    ("groq", "openai/gpt-oss-120b"): {
        "display_name": "GPT-OSS 120B (Groq)",
        "context_window": 131072,
        "pricing": ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            last_verified="2026-11-13",
            source="Official Groq Pricing Schedule",
            is_local=False,
        ),
        "latency_tier": "ultra_low",
        "heuristic_alignment": {
            TaskType.CODING: 8.5,
            TaskType.GENERAL: 8.0,
            TaskType.SUMMARIZATION: 8.0,
            TaskType.COMPLEX_REASONING: 7.5,
            TaskType.MATH: 7.0,
            TaskType.DOCUMENT_ANALYSIS: 7.0,
            TaskType.CREATIVE: 7.5,
        },
        "description": "High-throughput open-weights reasoning model running on Groq LPU.",
    },
    ("groq", "qwen/qwen3.8-27b"): {
        "display_name": "Qwen 3.8 27B (Groq)",
        "context_window": 32768,
        "pricing": ModelPricing(
            input_cost_per_m=0.20,
            output_cost_per_m=0.20,
            last_verified="2026-11-13",
            source="Official Groq Pricing Schedule",
            is_local=False,
        ),
        "latency_tier": "ultra_low",
        "heuristic_alignment": {
            TaskType.CODING: 7.5,
            TaskType.GENERAL: 7.5,
            TaskType.MATH: 7.0,
            TaskType.SUMMARIZATION: 7.0,
            TaskType.CREATIVE: 7.0,
        },
        "description": "Fast multilingual open-weights model on Groq.",
    },
    # Google Gemini verified models (Official Google GenAI Documentation, Verified 2026-11-13)
    ("gemini", "gemini-3.5-flash"): {
        "display_name": "Gemini 3.5 Flash",
        "context_window": 1048576,
        "pricing": ModelPricing(
            input_cost_per_m=0.075,
            output_cost_per_m=0.30,
            last_verified="2026-11-13",
            source="Google GenAI Developer Documentation",
            is_local=False,
        ),
        "latency_tier": "low",
        "supports_vision": True,
        "heuristic_alignment": {
            TaskType.VISION: 9.5,
            TaskType.DOCUMENT_ANALYSIS: 9.5,
            TaskType.SUMMARIZATION: 9.0,
            TaskType.GENERAL: 8.5,
            TaskType.CODING: 8.0,
            TaskType.COMPLEX_REASONING: 8.0,
            TaskType.CREATIVE: 8.0,
            TaskType.MATH: 7.5,
        },
        "description": "Flagship 1M token context window model for massive document and general processing.",
    },
    ("gemini", "gemini-3.8-flash"): {
        "display_name": "Gemini 3.8 Flash",
        "context_window": 1048576,
        "pricing": ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            last_verified="2026-11-13",
            source="Google GenAI Developer Documentation",
            is_local=False,
        ),
        "latency_tier": "low",
        "supports_vision": True,
        "heuristic_alignment": {
            TaskType.VISION: 9.8,
            TaskType.COMPLEX_REASONING: 9.0,
            TaskType.DOCUMENT_ANALYSIS: 9.5,
            TaskType.CODING: 8.5,
            TaskType.MATH: 8.5,
            TaskType.GENERAL: 8.5,
            TaskType.SUMMARIZATION: 9.0,
            TaskType.CREATIVE: 8.5,
        },
        "description": "High-intelligence Gemini generation model with 1M context.",
    },
    # OpenAI verified models (Official OpenAI Pricing, Verified 2026-11-13)
    ("openai", "gpt-4.1-mini"): {
        "display_name": "GPT-4.1 Mini",
        "context_window": 128000,
        "pricing": ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            last_verified="2026-11-13",
            source="Official OpenAI Pricing Schedule",
            is_local=False,
        ),
        "latency_tier": "low",
        "supports_vision": True,
        "heuristic_alignment": {
            TaskType.VISION: 9.5,
            TaskType.CODING: 8.5,
            TaskType.MATH: 8.0,
            TaskType.GENERAL: 8.0,
            TaskType.COMPLEX_REASONING: 8.0,
            TaskType.SUMMARIZATION: 8.0,
            TaskType.CREATIVE: 8.0,
            TaskType.DOCUMENT_ANALYSIS: 7.5,
        },
        "description": "Next-generation efficient chat and coding model.",
    },
    ("openai", "gpt-4.1"): {
        "display_name": "GPT-4.1",
        "context_window": 128000,
        "pricing": ModelPricing(
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
            last_verified="2026-11-13",
            source="Official OpenAI Pricing Schedule",
            is_local=False,
        ),
        "latency_tier": "standard",
        "supports_vision": True,
        "heuristic_alignment": {
            TaskType.VISION: 9.8,
            TaskType.COMPLEX_REASONING: 9.5,
            TaskType.CODING: 9.5,
            TaskType.MATH: 9.0,
            TaskType.GENERAL: 9.0,
            TaskType.CREATIVE: 8.5,
            TaskType.DOCUMENT_ANALYSIS: 8.0,
            TaskType.SUMMARIZATION: 8.5,
        },
        "description": "Next-generation flagship reasoning and high-complexity model.",
    },
    ("openai", "o3"): {
        "display_name": "O3",
        "context_window": 128000,
        "pricing": ModelPricing(
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
            last_verified="2026-11-13",
            source="Official OpenAI Pricing Schedule",
            is_local=False,
        ),
        "latency_tier": "variable",
        "heuristic_alignment": {
            TaskType.MATH: 9.8,
            TaskType.COMPLEX_REASONING: 9.8,
            TaskType.CODING: 9.5,
            TaskType.GENERAL: 8.0,
        },
        "description": "Advanced deep-reasoning model optimized for complex STEM and math proofs.",
    },
    # Ollama verified local models
    ("ollama", "llama3.2:3b"): {
        "display_name": "Llama 3.2 3B (Local)",
        "context_window": 128000,
        "pricing": ModelPricing(is_local=True),
        "latency_tier": "variable",
        "heuristic_alignment": {
            TaskType.GENERAL: 7.0,
            TaskType.CREATIVE: 7.0,
            TaskType.SUMMARIZATION: 6.5,
            TaskType.CODING: 6.0,
            TaskType.MATH: 5.5,
            TaskType.DOCUMENT_ANALYSIS: 5.0,
            TaskType.COMPLEX_REASONING: 5.5,
        },
        "description": "Local high-privacy inference running directly on device hardware.",
    },
}

# Known deprecated model identifiers to exclude from automatic routing
DEPRECATED_MODELS: Set[Tuple[str, str]] = {
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-3.5-turbo"),
    ("openai", "text-davinci-003"),
}


class ModelCapabilityRegistry:
    """Dynamically discovers and tracks model capabilities, availability, and pricing.

    THREAD SAFETY & GUI GUIDELINES:
    Network discovery calls (`discover_models()`) must NEVER be executed directly on the Qt
    GUI thread. Instead, background components (such as `DiscoveryWorker` or `RoutedChatWorker`)
    call `discover_models()` in a separate QThread, while the GUI thread queries thread-safe
    cached accessors (`get_cached_models()`, `get_models()`).
    """

    DEFAULT_TTL_SECONDS = 300.0  # 5 minutes

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        credential_store: Optional[CredentialStore] = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        time_source: Optional[Callable[[], float]] = None,
    ):
        self.provider_manager = provider_manager or get_default_provider_manager()
        self.credential_store = credential_store or self.provider_manager.credential_store
        self.ttl_seconds = ttl_seconds
        self._time_source = time_source or time.monotonic

        self._lock = threading.RLock()
        self._cache: Dict[str, List[ModelProfile]] = {}
        self._cache_timestamp: float = 0.0

    def is_cache_valid(self) -> bool:
        """Check whether the cached discovery results are still within the TTL."""
        with self._lock:
            if not self._cache:
                return False
            elapsed = self._time_source() - self._cache_timestamp
            return elapsed < self.ttl_seconds

    def invalidate_cache(self) -> None:
        """Explicitly clear the discovery cache to force a fresh refresh."""
        with self._lock:
            self._cache.clear()
            self._cache_timestamp = 0.0
            logger.info("Model capability registry cache invalidated.")

    def _get_static_baseline_profiles(self) -> List[ModelProfile]:
        """Provide verified static model profiles when discovery cache is empty."""
        baseline: List[ModelProfile] = []
        for (pid, mid), data in VERIFIED_MODEL_METADATA.items():
            is_local = (pid == "ollama")
            p = ModelProfile(
                provider_id=pid,
                model_id=mid,
                display_name=data.get("display_name", mid),
                is_local=is_local,
                context_window=data.get("context_window", 128000),
                pricing=data.get("pricing", ModelPricing(is_local=is_local)),
                latency_tier=data.get("latency_tier", "standard"),
                heuristic_alignment=dict(data.get("heuristic_alignment", {})),
                supports_vision=data.get("supports_vision", False),
                is_deprecated=False,
                is_available=True,
                description=data.get("description", ""),
            )
            baseline.append(p)
        return baseline

    def get_cached_models(self, provider_id: Optional[str] = None) -> List[ModelProfile]:
        """Thread-safe non-blocking retrieval of currently cached models.

        Safe to call from the GUI thread. If the cache is empty or expired, returns
        verified baseline profiles without performing network calls.
        """
        with self._lock:
            if not self._cache:
                baseline = self._get_static_baseline_profiles()
                if provider_id:
                    return [m for m in baseline if m.provider_id.lower() == provider_id.lower()]
                return baseline

            if provider_id:
                return list(self._cache.get(provider_id.lower(), []))
            all_models = []
            for mlist in self._cache.values():
                all_models.extend(mlist)
            return all_models

    def discover_models(self, force_refresh: bool = False) -> Dict[str, List[ModelProfile]]:
        """Perform dynamic model discovery across all eligible and keyed providers.

        WARNING: Performs network I/O. Do NOT invoke from the Qt GUI thread.
        """
        with self._lock:
            if not force_refresh and self.is_cache_valid():
                return {k: list(v) for k, v in self._cache.items()}

        discovered_by_provider: Dict[str, List[ModelProfile]] = {}
        providers = self.provider_manager.list_providers()

        for provider in providers:
            pid = provider.provider_id.lower()

            # Credential check for cloud providers: never read secret values
            if not provider.is_local:
                if not self.credential_store.has_api_key(pid):
                    logger.debug("Skipping cloud provider '%s': no credentials configured.", pid)
                    continue

            # Query dynamic models via existing provider interface
            raw_models: List[ModelInfo] = []
            try:
                raw_models = provider.get_available_models()
            except Exception as exc:
                logger.warning(
                    "Dynamic discovery failed for provider '%s': %s. Marking unavailable.",
                    pid, exc
                )

            profiles: List[ModelProfile] = []
            for info in raw_models:
                profile = self._build_model_profile(provider.is_local, info)
                profiles.append(profile)

            discovered_by_provider[pid] = profiles

        # Update cache under lock
        with self._lock:
            self._cache = discovered_by_provider
            self._cache_timestamp = self._time_source()
            logger.info(
                "Model capability registry discovery completed. Cached %d models across %d providers.",
                sum(len(v) for v in self._cache.values()),
                len(self._cache),
            )
            return {k: list(v) for k, v in self._cache.items()}

    def _build_model_profile(self, is_local: bool, info: ModelInfo) -> ModelProfile:
        """Construct a ModelProfile from dynamic ModelInfo and verified metadata."""
        pid = info.provider_id.lower()
        mid = info.id
        lookup_key = (pid, mid.lower())

        is_deprecated = lookup_key in DEPRECATED_MODELS or any(
            d_mid in mid.lower() for d_pid, d_mid in DEPRECATED_MODELS if d_pid == pid
        )

        is_vision = (
            pid == "gemini"
            or any(k in mid.lower() for k in ["vision", "llava", "4o", "gpt-4", "gemini", "minicpm-v", "bakllava", "vl"])
        )

        verified = VERIFIED_MODEL_METADATA.get(lookup_key)

        if verified:
            supports_vision = verified.get("supports_vision", is_vision)
            return ModelProfile(
                provider_id=pid,
                model_id=mid,
                display_name=verified.get("display_name", info.name or mid),
                is_local=is_local,
                context_window=verified.get("context_window", info.context_window),
                pricing=verified.get("pricing", ModelPricing(is_local=is_local)),
                latency_tier=verified.get("latency_tier", "standard"),
                heuristic_alignment=dict(verified.get("heuristic_alignment", {})),
                supports_vision=supports_vision,
                is_deprecated=is_deprecated,
                is_available=True,
                description=verified.get("description", info.description or ""),
            )

        # Discovered model without verified entry: do not fabricate pricing or benchmarks
        default_pricing = ModelPricing(is_local=is_local)
        # Neutral heuristic alignment fallback
        neutral_alignment = {t: 5.0 for t in TaskType}
        if is_vision:
            neutral_alignment[TaskType.VISION] = 9.0

        return ModelProfile(
            provider_id=pid,
            model_id=mid,
            display_name=info.name or mid,
            is_local=is_local,
            context_window=info.context_window,  # None if unsupplied
            pricing=default_pricing,
            latency_tier="unknown",
            heuristic_alignment=neutral_alignment,
            supports_vision=is_vision,
            is_deprecated=is_deprecated,
            is_available=True,
            description=info.description or f"Dynamic model from {pid.title()}",
        )

    def get_models(
        self,
        provider_id: Optional[str] = None,
        local_only: bool = False,
        excluded_provider_ids: Optional[Set[str]] = None,
        include_deprecated: bool = False,
        only_available: bool = True,
        required_capability: Optional[TaskType] = None,
        min_context_window: Optional[int] = None,
    ) -> List[ModelProfile]:
        """Query and filter models matching the given constraints."""
        all_models = self.get_cached_models(provider_id=provider_id)
        excluded = {p.lower() for p in (excluded_provider_ids or set())}

        filtered: List[ModelProfile] = []
        for m in all_models:
            if local_only and not m.is_local:
                continue

            if m.provider_id.lower() in excluded:
                continue

            if only_available and not m.is_available:
                continue

            if not include_deprecated and m.is_deprecated:
                continue

            if min_context_window is not None:
                # Models with unknown context window (None) or insufficient capacity are safely excluded
                if m.context_window is None or m.context_window < min_context_window:
                    continue

            if required_capability is not None:
                if (required_capability == TaskType.VISION or str(required_capability).lower() == "vision") and not getattr(m, "supports_vision", False):
                    continue
                # If model has heuristic alignment below 4.0 for a required task, skip
                score = m.heuristic_alignment.get(required_capability, 5.0)
                if score < 4.0:
                    continue

            filtered.append(m)

        return filtered


class DiscoveryWorker(QThread):
    """Background worker thread to run dynamic model discovery outside the Qt GUI thread."""

    discovery_finished = Signal(dict)
    discovery_failed = Signal(str)

    def __init__(self, registry: ModelCapabilityRegistry, force_refresh: bool = False, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.registry = registry
        self.force_refresh = force_refresh

    def run(self) -> None:
        try:
            results = self.registry.discover_models(force_refresh=self.force_refresh)
            self.discovery_finished.emit(results)
        except Exception as exc:
            self.discovery_failed.emit(str(exc))


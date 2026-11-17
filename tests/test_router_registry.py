import os
import threading
import time
import unittest
from unittest.mock import MagicMock

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.security import CredentialStore
from src.providers.base import AIProvider
from src.providers.manager import ProviderManager
from src.providers.models import ModelInfo
from src.routing.registry import (
    DEPRECATED_MODELS,
    VERIFIED_MODEL_METADATA,
    DiscoveryWorker,
    ModelCapabilityRegistry,
)
from src.routing.types import TaskType

app = QApplication.instance() or QApplication([])


class MockProvider(AIProvider):
    def __init__(self, pid: str, is_local: bool, models: list, fail: bool = False):
        self.provider_id = pid
        self.display_name = pid.title()
        self.is_local = is_local
        self._models = models
        self._fail = fail
        self.call_count = 0

    def generate(self, messages, model, **kwargs):
        return "response"

    def stream(self, messages, model, **kwargs):
        yield "chunk"

    def validate_connection(self):
        return not self._fail

    def get_available_models(self):
        self.call_count += 1
        if self._fail:
            raise RuntimeError(f"Connection failed for {self.provider_id}")
        return list(self._models)


class TestModelCapabilityRegistry(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock(spec=CredentialStore)
        # Default: groq and gemini have keys, openai does not
        self.mock_store.has_api_key.side_effect = lambda pid: pid in ["groq", "gemini"]

        self.p_ollama = MockProvider(
            "ollama", is_local=True, models=[
                ModelInfo("llama3.2:3b", "Llama 3.2 3B", "ollama", is_local=True)
            ]
        )
        self.p_groq = MockProvider(
            "groq", is_local=False, models=[
                ModelInfo("openai/gpt-oss-120b", "GPT-OSS 120B", "groq", is_local=False),
                ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B", "groq", is_local=False),  # Deprecated
            ]
        )
        self.p_gemini = MockProvider(
            "gemini", is_local=False, models=[
                ModelInfo("gemini-3.5-flash", "Gemini 3.5 Flash", "gemini", is_local=False),
                ModelInfo("gemini-custom-experimental", "Custom Model", "gemini", is_local=False),  # Unknown
            ]
        )
        self.p_openai = MockProvider(
            "openai", is_local=False, models=[
                ModelInfo("gpt-4.1", "GPT-4.1", "openai", is_local=False)
            ]
        )

        self.mgr = ProviderManager(credential_store=self.mock_store)
        # Clear default providers and register mock providers
        self.mgr._providers.clear()
        self.mgr.register_provider(self.p_ollama)
        self.mgr.register_provider(self.p_groq)
        self.mgr.register_provider(self.p_gemini)
        self.mgr.register_provider(self.p_openai)

        self.simulated_time = 1000.0
        self.registry = ModelCapabilityRegistry(
            provider_manager=self.mgr,
            credential_store=self.mock_store,
            ttl_seconds=300.0,
            time_source=lambda: self.simulated_time,
        )

    def test_dynamic_discovery_and_credential_filtering(self):
        """Cloud providers with credentials and local providers are discovered; unkeyed cloud providers are skipped."""
        discovered = self.registry.discover_models()

        # Ollama, Groq, Gemini should be present
        self.assertIn("ollama", discovered)
        self.assertIn("groq", discovered)
        self.assertIn("gemini", discovered)
        # OpenAI has no credentials in mock_store -> must be excluded
        self.assertNotIn("openai", discovered)
        self.assertEqual(self.p_openai.call_count, 0)

    def test_cache_hit_avoids_repeated_discovery_calls(self):
        """Second call within TTL should return cached results without re-querying providers."""
        self.registry.discover_models()
        self.assertEqual(self.p_groq.call_count, 1)

        # Call again within TTL
        self.simulated_time += 60.0  # 1 minute later (< 5 min TTL)
        self.registry.discover_models()
        self.assertEqual(self.p_groq.call_count, 1, "Cache hit should not call get_available_models()")

    def test_cache_expiration(self):
        """When TTL expires, discovery should re-query providers."""
        self.registry.discover_models()
        self.assertEqual(self.p_groq.call_count, 1)

        # Advance time past 300s TTL
        self.simulated_time += 301.0
        self.assertFalse(self.registry.is_cache_valid())

        self.registry.discover_models()
        self.assertEqual(self.p_groq.call_count, 2, "Expired cache must trigger fresh discovery.")

    def test_cache_invalidation(self):
        """Explicit cache invalidation forces fresh discovery on next request."""
        self.registry.discover_models()
        self.assertTrue(self.registry.is_cache_valid())

        self.registry.invalidate_cache()
        self.assertFalse(self.registry.is_cache_valid())
        self.assertEqual(len(self.registry._cache), 0)

    def test_unknown_model_metadata_and_pricing(self):
        """Dynamically discovered models not in the verified metadata table get 'Pricing unavailable'."""
        self.registry.discover_models()
        models = self.registry.get_models(provider_id="gemini")
        custom_model = next((m for m in models if m.model_id == "gemini-custom-experimental"), None)
        self.assertIsNotNone(custom_model)
        self.assertEqual(custom_model.pricing.formatted_pricing(), "Pricing unavailable")
        # Neutral alignment scores (5.0)
        self.assertEqual(custom_model.heuristic_alignment.get(TaskType.CODING), 5.0)

    def test_local_pricing_formatting(self):
        """Local models clearly identify zero API charges."""
        self.registry.discover_models()
        models = self.registry.get_models(provider_id="ollama")
        self.assertTrue(len(models) > 0)
        p = models[0].pricing.formatted_pricing()
        self.assertIn("$0.00 API charge", p)
        self.assertIn("Local inference", p)

    def test_deprecated_model_filtering(self):
        """Deprecated models are marked and filtered out by default, but can be included when requested."""
        self.registry.discover_models()

        # By default, deprecated models are omitted
        normal_models = self.registry.get_models(provider_id="groq", include_deprecated=False)
        model_ids = [m.model_id for m in normal_models]
        self.assertIn("openai/gpt-oss-120b", model_ids)
        self.assertNotIn("llama-3.3-70b-versatile", model_ids)

        # When include_deprecated=True, deprecated models appear
        all_models = self.registry.get_models(provider_id="groq", include_deprecated=True)
        all_ids = [m.model_id for m in all_models]
        self.assertIn("llama-3.3-70b-versatile", all_ids)
        dep_model = next(m for m in all_models if m.model_id == "llama-3.3-70b-versatile")
        self.assertTrue(dep_model.is_deprecated)

    def test_provider_exclusion_and_local_only_filtering(self):
        """Excluded providers and local_only constraints are correctly applied."""
        self.registry.discover_models()

        # Local only: returns only Ollama
        local_models = self.registry.get_models(local_only=True)
        self.assertTrue(all(m.is_local for m in local_models))
        self.assertTrue(all(m.provider_id == "ollama" for m in local_models))

        # Exclude Groq
        no_groq = self.registry.get_models(excluded_provider_ids={"groq"})
        self.assertTrue(all(m.provider_id != "groq" for m in no_groq))

    def test_context_window_and_unknown_exclusion(self):
        """When min_context_window is specified, models with unknown (None) or insufficient context are excluded."""
        self.registry.discover_models()
        # gemini-custom-experimental has context_window=None; gemini-3.5-flash has 1,048,576
        models_no_filter = self.registry.get_models(provider_id="gemini")
        model_ids_no_filter = [m.model_id for m in models_no_filter]
        self.assertIn("gemini-custom-experimental", model_ids_no_filter)
        self.assertIn("gemini-3.5-flash", model_ids_no_filter)

        # Apply min_context_window of 8,000 tokens
        models_with_filter = self.registry.get_models(provider_id="gemini", min_context_window=8000)
        model_ids_filtered = [m.model_id for m in models_with_filter]
        # Unknown context window model MUST be excluded
        self.assertNotIn("gemini-custom-experimental", model_ids_filtered)
        # Sufficient context window model must be kept
        self.assertIn("gemini-3.5-flash", model_ids_filtered)

    def test_latency_transparency_and_formatting(self):
        """Latency tier remains explicitly transparent and never fabricated as benchmarks."""
        self.registry.discover_models()
        models = self.registry.get_models(provider_id="gemini")
        custom_model = next(m for m in models if m.model_id == "gemini-custom-experimental")
        flash_model = next(m for m in models if m.model_id == "gemini-3.5-flash")

        # Unverified model has unknown latency
        self.assertEqual(custom_model.latency_tier, "unknown")
        self.assertEqual(custom_model.formatted_latency(), "Latency unavailable")

        # Verified model has verified low latency tier
        self.assertEqual(flash_model.latency_tier, "low")
        self.assertEqual(flash_model.formatted_latency(), "Latency tier: Low")

    def test_discovery_failure_isolation(self):
        """If one provider fails discovery, others continue successfully."""
        p_failing = MockProvider("groq", is_local=False, models=[], fail=True)
        self.mgr._providers["groq"] = p_failing

        discovered = self.registry.discover_models(force_refresh=True)
        self.assertEqual(discovered["groq"], [])
        # Ollama and Gemini should still succeed
        self.assertTrue(len(discovered["ollama"]) > 0)
        self.assertTrue(len(discovered["gemini"]) > 0)

    def test_thread_safety_concurrent_access(self):
        """Concurrent calls from multiple threads do not cause race conditions or corrupt the cache."""
        errors = []

        def worker_task():
            try:
                for _ in range(5):
                    self.registry.discover_models()
                    models = self.registry.get_models()
                    self.assertTrue(len(models) > 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors occurred: {errors}")

    def test_discovery_worker_background_execution(self):
        """DiscoveryWorker executes in a background QThread and emits finished signal."""
        worker = DiscoveryWorker(self.registry, force_refresh=True)
        results_received = []

        worker.discovery_finished.connect(lambda data: results_received.append(data))
        worker.start()
        worker.wait(5000)
        app.processEvents()

        self.assertEqual(len(results_received), 1)
        self.assertIn("ollama", results_received[0])


if __name__ == "__main__":
    unittest.main()


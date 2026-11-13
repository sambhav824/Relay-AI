import os
import unittest
from unittest.mock import MagicMock

# Run Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.core.message import Message, MessageRole
from src.routing.classifier import TaskClassifier
from src.routing.registry import ModelCapabilityRegistry
from src.routing.router import ModelRouter
from src.routing.types import (
    ModelPricing,
    ModelProfile,
    NoEligibleModelsError,
    RoutingStrategy,
    TaskType,
)

app = QApplication.instance() or QApplication([])


class TestRouterDecision(unittest.TestCase):
    def setUp(self):
        self.mock_registry = MagicMock(spec=ModelCapabilityRegistry)

        # Build mock model profiles
        self.m_ollama = ModelProfile(
            provider_id="ollama",
            model_id="llama3.2:3b",
            display_name="Llama 3.2 3B (Local)",
            is_local=True,
            context_window=128000,
            pricing=ModelPricing(is_local=True),
            latency_tier="variable",
            heuristic_alignment={TaskType.GENERAL: 7.0, TaskType.CODING: 6.0},
        )
        self.m_groq_speed = ModelProfile(
            provider_id="groq",
            model_id="openai/gpt-oss-120b",
            display_name="GPT-OSS 120B",
            is_local=False,
            context_window=131072,
            pricing=ModelPricing(input_cost_per_m=0.15, output_cost_per_m=0.60, is_local=False),
            latency_tier="ultra_low",
            heuristic_alignment={TaskType.CODING: 8.5, TaskType.GENERAL: 8.0},
        )
        self.m_openai_flagship = ModelProfile(
            provider_id="openai",
            model_id="gpt-4.1",
            display_name="GPT-4.1 Flagship",
            is_local=False,
            context_window=128000,
            pricing=ModelPricing(input_cost_per_m=2.50, output_cost_per_m=10.00, is_local=False),
            latency_tier="standard",
            heuristic_alignment={TaskType.CODING: 9.8, TaskType.GENERAL: 9.5},
        )
        self.m_gemini_doc = ModelProfile(
            provider_id="gemini",
            model_id="gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            is_local=False,
            context_window=1048576,
            pricing=ModelPricing(input_cost_per_m=0.075, output_cost_per_m=0.30, is_local=False),
            latency_tier="low",
            heuristic_alignment={TaskType.DOCUMENT_ANALYSIS: 9.8, TaskType.GENERAL: 8.5, TaskType.CODING: 8.0},
        )

        self.all_mock_models = [self.m_ollama, self.m_groq_speed, self.m_openai_flagship, self.m_gemini_doc]

        def get_models_mock(local_only=False, excluded_provider_ids=None, min_context_window=None, **kwargs):
            excluded = {p.lower() for p in (excluded_provider_ids or set())}
            results = []
            for m in self.all_mock_models:
                if local_only and not m.is_local:
                    continue
                if m.provider_id.lower() in excluded:
                    continue
                if min_context_window is not None:
                    if m.context_window is None or m.context_window < min_context_window:
                        continue
                results.append(m)
            return results

        self.mock_registry.get_models.side_effect = get_models_mock

        self.classifier = TaskClassifier()
        self.router = ModelRouter(registry=self.mock_registry, classifier=self.classifier)

    def test_balanced_routing_coding(self):
        """Under Balanced strategy, Groq (high alignment + ultra-low latency + low cost) ranks highest for coding."""
        prompt = "Write a Python script to sort a dictionary by its values."
        decision = self.router.route(prompt, strategy=RoutingStrategy.BALANCED)

        self.assertEqual(decision.primary.provider_id, "groq")
        self.assertEqual(decision.primary.model_id, "openai/gpt-oss-120b")
        self.assertGreater(len(decision.fallback_chain), 0)
        self.assertIn("Balanced", decision.explanation.reason)

    def test_quality_first_routing_coding(self):
        """Under Quality-First strategy, flagship model with highest alignment (gpt-4.1) ranks highest."""
        prompt = "Write a Python script to sort a dictionary by its values."
        decision = self.router.route(prompt, strategy=RoutingStrategy.QUALITY_FIRST)

        self.assertEqual(decision.primary.provider_id, "openai")
        self.assertEqual(decision.primary.model_id, "gpt-4.1")
        self.assertIn("Quality", decision.explanation.strategy.value.title())

    def test_cost_efficient_routing(self):
        """Under Cost-Efficient strategy, local zero API cost model or lowest cloud cost model ranks highest."""
        prompt = "Explain why the sky is blue in simple words."
        decision = self.router.route(prompt, strategy=RoutingStrategy.COST_EFFICIENT)

        # Local Ollama ($0.00 API charge) or Gemini ($0.075) should rank top
        self.assertIn(decision.primary.provider_id, ["ollama", "gemini"])

    def test_local_first_routing(self):
        """Under Local-First strategy, local Ollama model receives bonus and ranks highest."""
        prompt = "Draft a friendly hello email to my team."
        decision = self.router.route(prompt, strategy=RoutingStrategy.LOCAL_FIRST)

        self.assertEqual(decision.primary.provider_id, "ollama")
        self.assertTrue(decision.primary.profile.is_local)
        self.assertIn("Local", decision.explanation.reason)

    def test_local_only_enforcement(self):
        """When local_only=True, all cloud providers are excluded strictly."""
        prompt = "Solve this complex coding problem."
        decision = self.router.route(prompt, local_only=True)

        self.assertEqual(decision.primary.provider_id, "ollama")
        for fb in decision.fallback_chain:
            self.assertTrue(fb.profile.is_local)
        self.assertIn("Local-Only active", decision.explanation.exclusions_summary)

    def test_provider_exclusions(self):
        """When user excludes specific providers, they do not appear in primary or fallbacks."""
        prompt = "Write a quick Python script."
        decision = self.router.route(
            prompt,
            strategy=RoutingStrategy.QUALITY_FIRST,
            provider_exclusions={"openai", "groq"}
        )

        self.assertNotIn(decision.primary.provider_id, ["openai", "groq"])
        for fb in decision.fallback_chain:
            self.assertNotIn(fb.provider_id, ["openai", "groq"])
        self.assertIn("groq", decision.explanation.exclusions_summary)

    def test_unknown_pricing_normalization(self):
        """Unknown pricing receives a neutral score (0.5), not free (1.0)."""
        score_none = ModelRouter._normalize_cost_score(None)
        self.assertEqual(score_none, 0.5)

        pricing_unknown = ModelPricing(input_cost_per_m=None, is_local=False)
        score_unknown = ModelRouter._normalize_cost_score(pricing_unknown)
        self.assertEqual(score_unknown, 0.5)

        pricing_local = ModelPricing(is_local=True)
        score_local = ModelRouter._normalize_cost_score(pricing_local)
        self.assertEqual(score_local, 1.0)

    def test_unknown_speed_normalization(self):
        """Unknown speed receives a neutral score (0.5), not ultra_low (1.0)."""
        score_none = ModelRouter._normalize_speed_score(None)
        self.assertEqual(score_none, 0.5)

        score_unknown = ModelRouter._normalize_speed_score("unknown_tier")
        self.assertEqual(score_unknown, 0.5)

        score_ultra = ModelRouter._normalize_speed_score("ultra_low")
        self.assertEqual(score_ultra, 1.0)

    def test_unknown_model_metadata_handling(self):
        """Model with unverified metadata routes safely without errors."""
        unverified_model = ModelProfile(
            provider_id="custom_prov",
            model_id="custom_v1",
            display_name="Custom Model",
            is_local=False,
            context_window=None,
            pricing=ModelPricing(is_local=False),
            latency_tier="unknown",
            heuristic_alignment={},
        )
        self.all_mock_models = [unverified_model]

        decision = self.router.route("Hello world")
        self.assertEqual(decision.primary.model_id, "custom_v1")
        self.assertEqual(decision.primary.task_alignment_score, 5.0)
        self.assertEqual(decision.explanation.pricing_summary, "Pricing unavailable")

    def test_no_eligible_candidates_raises_error(self):
        """When constraints eliminate all models, a clean NoEligibleModelsError is raised."""
        self.all_mock_models = []
        with self.assertRaises(NoEligibleModelsError) as ctx:
            self.router.route("Test query")
        self.assertIn("No eligible AI models found", str(ctx.exception))

    def test_context_window_filtering(self):
        """Prompt and history requiring large context filter out small-context models."""
        small_model = ModelProfile(
            provider_id="prov_small",
            model_id="small-ctx",
            display_name="Small Context",
            is_local=False,
            context_window=2000,
        )
        large_model = ModelProfile(
            provider_id="prov_large",
            model_id="large-ctx",
            display_name="Large Context",
            is_local=False,
            context_window=1000000,
        )
        self.all_mock_models = [small_model, large_model]

        # Create massive conversation context (~50,000 chars -> ~15,000 tokens)
        long_prompt = "Analyze this: " + ("data point " * 5000)
        decision = self.router.route(long_prompt)

        # Small model (2000 ctx) must be excluded; only large model qualifies
        self.assertEqual(decision.primary.model_id, "large-ctx")

    def test_unknown_context_window_excluded_when_min_context_required(self):
        """Models with unknown context window (None) are excluded when a minimum context requirement exists."""
        unknown_ctx_model = ModelProfile(
            provider_id="prov_unknown",
            model_id="unknown-ctx",
            display_name="Unknown Context",
            is_local=False,
            context_window=None,
        )
        large_model = ModelProfile(
            provider_id="prov_large",
            model_id="large-ctx",
            display_name="Large Context",
            is_local=False,
            context_window=1000000,
        )
        self.all_mock_models = [unknown_ctx_model, large_model]

        # Long prompt demanding ~15,000 tokens
        long_prompt = "Analyze: " + ("data " * 10000)
        decision = self.router.route(long_prompt)

        # Model with unknown context window must be safely excluded; only large model qualifies
        self.assertEqual(decision.primary.model_id, "large-ctx")

    def test_deterministic_candidate_ranking(self):
        """Identical inputs and candidate profiles produce identical ranking orders."""
        prompt = "Write an async python crawler"
        dec1 = self.router.route(prompt, strategy=RoutingStrategy.BALANCED)
        dec2 = self.router.route(prompt, strategy=RoutingStrategy.BALANCED)

        self.assertEqual(dec1.primary.model_id, dec2.primary.model_id)
        self.assertEqual(
            [c.model_id for c in dec1.fallback_chain],
            [c.model_id for c in dec2.fallback_chain]
        )

    def test_fallback_candidate_ordering(self):
        """Fallback chain contains up to 2 distinct secondary candidates in descending rank order."""
        decision = self.router.route("Explain quantum entanglement", strategy=RoutingStrategy.BALANCED)

        self.assertLessEqual(len(decision.fallback_chain), 2)
        # Primary is not in fallback chain
        self.assertNotIn(decision.primary.model_id, [f.model_id for f in decision.fallback_chain])

        if len(decision.fallback_chain) == 2:
            self.assertGreaterEqual(
                decision.fallback_chain[0].composite_score,
                decision.fallback_chain[1].composite_score
            )

    def test_low_confidence_general_classification(self):
        """Ambiguous prompt with low confidence defaults safely to GENERAL task."""
        prompt = "Maybe some things are nice today."
        decision = self.router.route(prompt)

        self.assertEqual(decision.classification.task_type, TaskType.GENERAL)
        self.assertTrue(decision.classification.is_fallback_general)
        self.assertIn("ambiguous", decision.explanation.reason)

    def test_conversation_history_unmodified(self):
        """Conversation history passed to route() is never modified or appended."""
        history = [
            Message(MessageRole.USER, "Prior user prompt"),
            Message(MessageRole.ASSISTANT, "Prior assistant reply"),
        ]
        history_copy = list(history)

        self.router.route("What about this next thing?", history=history)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content, history_copy[0].content)
        self.assertEqual(history[1].content, history_copy[1].content)


if __name__ == "__main__":
    unittest.main()


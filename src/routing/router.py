import logging
from typing import Dict, List, Optional, Set

from src.core.message import Message
from src.routing.classifier import TaskClassifier
from src.routing.registry import ModelCapabilityRegistry
from src.routing.types import (
    ClassificationResult,
    ModelProfile,
    NoEligibleModelsError,
    RoutingCandidate,
    RoutingDecision,
    RoutingExplanation,
    RoutingStrategy,
    TaskType,
)

logger = logging.getLogger("RelayAI.ModelRouter")

# Configurable default scoring weights per strategy.
# Dimensions: Quality (heuristic alignment), Speed (latency tier), Cost (cost efficiency), and Local Bonus.
DEFAULT_STRATEGY_WEIGHTS: Dict[RoutingStrategy, Dict[str, float]] = {
    RoutingStrategy.BALANCED: {
        "quality": 0.45,
        "speed": 0.30,
        "cost": 0.25,
        "local_bonus": 0.0,
    },
    RoutingStrategy.QUALITY_FIRST: {
        "quality": 0.80,
        "speed": 0.10,
        "cost": 0.10,
        "local_bonus": 0.0,
    },
    RoutingStrategy.COST_EFFICIENT: {
        "quality": 0.25,
        "speed": 0.25,
        "cost": 0.50,
        "local_bonus": 0.15,
    },
    RoutingStrategy.LOCAL_FIRST: {
        "quality": 0.20,
        "speed": 0.20,
        "cost": 0.20,
        "local_bonus": 0.50,
    },
}


class ModelRouter:
    """Core decision engine that evaluates tasks and model profiles to select the best model.

    DESIGN PRINCIPLES:
    - Pure decision-making: Performs NO network I/O and does NOT block the GUI thread.
    - Non-destructive: Does NOT modify conversation history or append messages.
    - Transparent: Distinguishes heuristic alignment from benchmarks, and flags unknown metadata honestly.
    """

    def __init__(
        self,
        registry: Optional[ModelCapabilityRegistry] = None,
        classifier: Optional[TaskClassifier] = None,
        strategy_weights: Optional[Dict[RoutingStrategy, Dict[str, float]]] = None,
    ):
        self.registry = registry or ModelCapabilityRegistry()
        self.classifier = classifier or TaskClassifier()
        self.strategy_weights = strategy_weights or dict(DEFAULT_STRATEGY_WEIGHTS)

    def route(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        provider_exclusions: Optional[Set[str]] = None,
        local_only: bool = False,
        required_capability: Optional[TaskType] = None,
    ) -> RoutingDecision:
        """Evaluate prompt and constraints to produce an optimal RoutingDecision.

        Args:
            prompt: Current user message text.
            history: Complete conversation history (read-only; unmodified).
            strategy: User-selected routing strategy (Balanced, Quality-First, Cost-Efficient, Local-First).
            provider_exclusions: Optional set of provider IDs to omit.
            local_only: If True, strictly excludes all cloud providers.
            required_capability: Optional specific TaskType capability required.

        Returns:
            RoutingDecision containing primary candidate, fallback chain (up to 2), and explanation.

        Raises:
            NoEligibleModelsError: If zero models qualify under the specified constraints.
        """
        # 1. Deterministic Task Classification
        classification = self.classifier.classify(prompt, history=history)
        task = required_capability or classification.task_type

        # 2. Conservative Context Window Estimation (without mutating history)
        estimated_context_tokens = self._estimate_conversation_tokens(prompt, history)
        # A minimum context requirement exists when conversation exceeds standard base capacity (2,048 tokens)
        min_context = estimated_context_tokens if estimated_context_tokens > 2048 else None

        # 3. Retrieve available models from registry cache (non-blocking)
        # Note: Background discovery should populate registry before routing.
        available_models = self.registry.get_models(
            local_only=local_only,
            excluded_provider_ids=provider_exclusions,
            include_deprecated=False,
            only_available=True,
            min_context_window=min_context,
            required_capability=required_capability,
        )

        if not available_models:
            exclusions_text = f" (exclusions={provider_exclusions})" if provider_exclusions else ""
            local_text = " [Local-Only enabled]" if local_only else ""
            raise NoEligibleModelsError(
                f"No eligible AI models found for task '{task.value}' under constraints{local_text}{exclusions_text} "
                f"with required context of ~{estimated_context_tokens:,} tokens."
            )

        # 4. Rank candidates using strategy weights
        ranked_candidates = self._rank_candidates(available_models, task, strategy)

        # 5. Extract primary and up to 2 fallbacks
        primary = ranked_candidates[0]
        fallback_chain = ranked_candidates[1:3]

        # 6. Generate transparent human-readable explanation
        explanation = self._build_explanation(
            primary=primary,
            fallbacks=fallback_chain,
            classification=classification,
            strategy=strategy,
            local_only=local_only,
            exclusions=provider_exclusions,
        )

        return RoutingDecision(
            primary=primary,
            fallback_chain=fallback_chain,
            classification=classification,
            strategy=strategy,
            explanation=explanation,
        )

    @staticmethod
    def _estimate_conversation_tokens(prompt: str, history: Optional[List[Message]]) -> int:
        """Conservatively estimate token consumption across prompt and history without altering messages."""
        total_chars = len(prompt or "")
        if history:
            total_chars += sum(len(m.content or "") for m in history)
        # Conservative ratio: ~3.2 characters per token on average for diverse English & code
        return max(10, int(total_chars / 3.2))

    def _rank_candidates(
        self,
        models: List[ModelProfile],
        task: TaskType,
        strategy: RoutingStrategy,
    ) -> List[RoutingCandidate]:
        """Score each model candidate deterministically across normalized dimensions."""
        weights = self.strategy_weights.get(strategy, DEFAULT_STRATEGY_WEIGHTS[RoutingStrategy.BALANCED])
        candidates: List[RoutingCandidate] = []

        for profile in models:
            # Dimension 1: Heuristic Task Alignment (Normalized 0.1 to 1.0)
            raw_alignment = profile.heuristic_alignment.get(task, 5.0)
            q_score = min(1.0, max(0.1, raw_alignment / 10.0))

            # Dimension 2: Latency Tier (Normalized 0.0 to 1.0; unknown is 0.5 neutral)
            s_score = self._normalize_speed_score(profile.latency_tier)

            # Dimension 3: Cost Efficiency (Normalized 0.0 to 1.0; unknown is 0.5 neutral)
            c_score = self._normalize_cost_score(profile.pricing)

            # Dimension 4: Local Bonus
            local_bonus = weights["local_bonus"] if profile.is_local else 0.0

            # Composite Calculation
            composite = (
                weights["quality"] * q_score
                + weights["speed"] * s_score
                + weights["cost"] * c_score
                + local_bonus
            )
            composite = round(composite, 3)

            explanation = (
                f"Composite: {composite:.3f} "
                f"(Alignment: {raw_alignment:.1f}/10, SpeedScore: {s_score:.2f}, CostScore: {c_score:.2f}"
                + (f", LocalBonus: +{local_bonus:.2f}" if local_bonus > 0 else "")
                + ")"
            )

            candidates.append(
                RoutingCandidate(
                    provider_id=profile.provider_id,
                    model_id=profile.model_id,
                    display_name=profile.display_name,
                    composite_score=composite,
                    task_alignment_score=raw_alignment,
                    ranking_explanation=explanation,
                    profile=profile,
                )
            )

        # Deterministic sort: composite score (descending), task alignment (descending), provider_id, model_id
        candidates.sort(
            key=lambda c: (
                -c.composite_score,
                -c.task_alignment_score,
                c.provider_id,
                c.model_id,
            )
        )
        return candidates

    @staticmethod
    def _normalize_speed_score(latency_tier: Optional[str]) -> float:
        """Map latency tiers to normalized speed scores. Unknown speeds receive neutral score."""
        tier = (latency_tier or "").lower()
        if tier == "ultra_low":
            return 1.0
        elif tier == "low":
            return 0.8
        elif tier == "standard":
            return 0.6
        elif tier == "variable":
            return 0.4
        return 0.5  # Neutral for unknown speed; do not assume fastest

    @staticmethod
    def _normalize_cost_score(pricing: Optional[object]) -> float:
        """Map pricing schedules to normalized cost efficiency scores. Unknown pricing receives neutral score."""
        if not pricing:
            return 0.5

        if getattr(pricing, "is_local", False):
            return 1.0  # Zero API charge

        in_cost = getattr(pricing, "input_cost_per_m", None)
        out_cost = getattr(pricing, "output_cost_per_m", None)

        if in_cost is None or out_cost is None:
            return 0.5  # Neutral for unknown pricing; do not assume free

        # Blended representative cost per 1M (weighted 75% input, 25% output)
        blended = (0.75 * in_cost) + (0.25 * out_cost)

        if blended <= 0.20:
            return 0.90
        elif blended <= 0.50:
            return 0.75
        elif blended <= 1.00:
            return 0.60
        elif blended <= 3.00:
            return 0.40
        else:
            return 0.20

    def _build_explanation(
        self,
        primary: RoutingCandidate,
        fallbacks: List[RoutingCandidate],
        classification: ClassificationResult,
        strategy: RoutingStrategy,
        local_only: bool,
        exclusions: Optional[Set[str]],
    ) -> RoutingExplanation:
        """Construct transparent reasoning describing why models were selected or excluded."""
        task_name = classification.task_type.value.replace("_", " ").title()
        pricing_text = primary.profile.pricing.formatted_pricing() if primary.profile else "Pricing unavailable"

        # Selection rationale
        reasons = []
        if local_only or strategy == RoutingStrategy.LOCAL_FIRST:
            if primary.profile and primary.profile.is_local:
                reasons.append("Selected for Local on-device zero API cost and complete data privacy.")
        if strategy == RoutingStrategy.QUALITY_FIRST:
            reasons.append(f"Ranked highest for task heuristic alignment ({primary.task_alignment_score:.1f}/10) under Quality-First strategy.")
        elif strategy == RoutingStrategy.COST_EFFICIENT:
            reasons.append(f"Selected for superior cost efficiency under Cost-Efficient strategy.")
        elif strategy == RoutingStrategy.BALANCED:
            reasons.append(f"Best composite balance of intelligence, latency, and cost for {task_name} under Balanced strategy.")

        if classification.is_fallback_general:
            reasons.append("Task classification was general/ambiguous; safe balanced model selected.")

        reason_str = " ".join(reasons)

        fallback_names = [f"{f.provider_id.title()} · {f.model_id}" for f in fallbacks]

        # Exclusions summary
        exclusions_parts = []
        if local_only:
            exclusions_parts.append("All cloud providers excluded (Local-Only active)")
        if exclusions:
            exclusions_parts.append(f"User excluded: {', '.join(sorted(exclusions))}")
        exclusions_summary = "; ".join(exclusions_parts)

        return RoutingExplanation(
            selected_model_id=primary.model_id,
            selected_provider_id=primary.provider_id,
            detected_task=classification.task_type,
            confidence=classification.confidence,
            strategy=strategy,
            reason=reason_str,
            fallback_candidates=fallback_names,
            pricing_summary=pricing_text,
            exclusions_summary=exclusions_summary,
        )

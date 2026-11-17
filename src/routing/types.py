from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    """Categorization of user queries to inform routing decisions."""
    CODING = "coding"
    MATH = "math"
    GENERAL = "general"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    DOCUMENT_ANALYSIS = "document_analysis"
    COMPLEX_REASONING = "complex_reasoning"
    VISION = "vision"


class RoutingStrategy(str, Enum):
    """User-selectable trade-off optimization strategies."""
    BALANCED = "balanced"
    QUALITY_FIRST = "quality_first"
    COST_EFFICIENT = "cost_efficient"
    LOCAL_FIRST = "local_first"


class FailureReason(str, Enum):
    """Categorized failure causes triggering router fallback."""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


@dataclass
class ModelPricing:
    """Pricing schedule for 1M tokens with source traceability."""
    input_cost_per_m: Optional[float] = None
    output_cost_per_m: Optional[float] = None
    last_verified: Optional[str] = None
    source: Optional[str] = None
    is_local: bool = False

    def formatted_pricing(self) -> str:
        """Format pricing transparently, identifying local inference or unavailable data."""
        if self.is_local:
            return "$0.00 API charge (Local inference; hardware/power costs not included)"
        if self.input_cost_per_m is not None and self.output_cost_per_m is not None:
            return f"${self.input_cost_per_m:.2f} in / ${self.output_cost_per_m:.2f} out per 1M tokens"
        return "Pricing unavailable"


@dataclass
class ModelProfile:
    """Capability and metadata profile of an AI model."""
    provider_id: str
    model_id: str
    display_name: str
    is_local: bool = False
    context_window: Optional[int] = None
    pricing: ModelPricing = field(default_factory=ModelPricing)
    latency_tier: str = "unknown"  # "ultra_low", "low", "standard", "variable", "unknown"
    heuristic_alignment: Dict[TaskType, float] = field(default_factory=dict)  # Heuristic scale 1.0 - 10.0
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    is_deprecated: bool = False
    is_available: bool = True
    description: str = ""

    def formatted_latency(self) -> str:
        """Format latency tier transparently, avoiding benchmark claims."""
        tier = (self.latency_tier or "").lower()
        if tier in ("ultra_low", "low", "standard", "variable"):
            return f"Latency tier: {tier.replace('_', ' ').title()}"
        return "Latency unavailable"


class RoutingError(Exception):
    """Base exception for routing engine failures."""
    pass


class NoEligibleModelsError(RoutingError):
    """Raised when no models qualify after applying filters and constraints."""
    pass


@dataclass
class ClassificationResult:
    """Outcome of deterministic task classification."""
    task_type: TaskType
    confidence: float  # Normalized 0.0 to 1.0
    is_fallback_general: bool = False
    matched_cues: List[str] = field(default_factory=list)
    raw_scores: Dict[TaskType, float] = field(default_factory=dict)


@dataclass
class RoutingCandidate:
    """A scored model candidate for task dispatch."""
    provider_id: str
    model_id: str
    display_name: str
    composite_score: float
    task_alignment_score: float = 5.0
    ranking_explanation: str = ""
    profile: Optional[ModelProfile] = None


@dataclass
class RoutingExplanation:
    """Transparent human-readable justification of routing decisions."""
    selected_model_id: str
    selected_provider_id: str
    detected_task: TaskType
    confidence: float
    strategy: RoutingStrategy
    reason: str
    fallback_candidates: List[str] = field(default_factory=list)
    pricing_summary: str = "Pricing unavailable"
    exclusions_summary: str = ""


@dataclass
class RoutingDecision:
    """Complete routing resolution with primary model and fallback chain."""
    primary: RoutingCandidate
    fallback_chain: List[RoutingCandidate] = field(default_factory=list)
    classification: ClassificationResult = field(
        default_factory=lambda: ClassificationResult(task_type=TaskType.GENERAL, confidence=0.5)
    )
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    explanation: Optional[RoutingExplanation] = None


# Backward compatibility aliases
TaskCategory = TaskType
TaskClassification = ClassificationResult


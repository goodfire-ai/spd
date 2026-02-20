"""
JSON schemas for the causal verification pipeline.

This module defines the data structures passed between components:
- LLM analysis outputs
- Experiment specifications
- Experiment results
- Verification reports
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class ExperimentType(str, Enum):
    ZERO_ABLATE = "zero_ablate"
    MEAN_ABLATE = "mean_ablate"
    PATCH = "patch"
    STEER = "steer"  # Add activation with scaling factor


class ExpectedDirection(str, Enum):
    INCREASES = "increases"
    DECREASES = "decreases"
    FLIPS = "flips"  # Top token changes
    NO_CHANGE = "no_change"


class Magnitude(str, Enum):
    LARGE = "large"      # > 1.0 logprob change
    MEDIUM = "medium"    # 0.3 - 1.0 logprob change
    SMALL = "small"      # < 0.3 logprob change


@dataclass
class ModuleAnalysis:
    """LLM's analysis of a single module."""
    module_id: int
    hypothesis: str  # What the LLM thinks this module does
    key_neurons: list[str]  # e.g., ["L30/N14210", "L31/N6239"]
    evidence: str  # Why the LLM thinks this (edge weights, neuron labels, etc.)
    importance: float  # 0-1, how causally important this module likely is

    def to_dict(self):
        return asdict(self)


class DistributionShift(str, Enum):
    """Expected shift patterns for multi-token experiments."""
    SPECIFIC_TO_GENERIC = "specific_to_generic"  # Domain term → generic (e.g., "substantia" → "brain")
    GENERIC_TO_SPECIFIC = "generic_to_specific"  # Generic → domain term
    SHIFT_BETWEEN = "shift_between"  # One specific → another specific (e.g., "Huntington's" → "Parkinson's")
    SUPPRESS_GROUP = "suppress_group"  # Reduce probability of a token group
    BOOST_GROUP = "boost_group"  # Increase probability of a token group


@dataclass
class TokenGroup:
    """A group of related tokens to track together."""
    name: str  # e.g., "factual_correct", "generic", "alternate"
    tokens: list[str]  # e.g., [" Sub", " substant", " Substant"]

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentSpec:
    """Specification for a single causal intervention experiment."""
    experiment_id: str
    circuit_id: str
    module_id: int
    experiment_type: str  # ExperimentType value
    target_prompt: str
    hypothesis: str
    target_token: str  # Primary token we're tracking (e.g., " yes")
    expected_direction: str  # ExpectedDirection value
    expected_magnitude: str  # Magnitude value
    confidence: float  # 0-1, LLM's confidence in prediction

    # Optional fields (must come after required fields)
    source_prompt: str | None = None  # For patching
    steer_scale: float | None = None  # For steering
    priority: int = 1  # Higher = run first

    # Multi-token tracking (optional, for non-binary outputs)
    token_groups: list[dict] | None = None  # List of TokenGroup dicts
    expected_shift: str | None = None  # DistributionShift value
    shift_from_group: str | None = None  # Name of group tokens should shift FROM
    shift_to_group: str | None = None  # Name of group tokens should shift TO

    # Specific tokens to track even if not in top-k (for testing semantic alternatives)
    track_tokens: list[str] | None = None  # e.g., [" serotonin", " norepinephrine"]

    # Hypothesis type: "semantic" (concept selection) vs "lexical" (form selection)
    hypothesis_type: str | None = None  # "semantic", "lexical", or "mixed"

    # Answer prefix (to match graph generation settings)
    answer_prefix: str | None = None  # e.g., "Answer:" - added after assistant header

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        return cls(**d)


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""
    experiment_id: str
    circuit_id: str
    module_id: int
    experiment_type: str

    # Baseline (no intervention)
    baseline_top_token: str
    baseline_top_prob: float
    baseline_target_logprob: float  # Logprob of the target token

    # After intervention
    result_top_token: str
    result_top_prob: float
    result_target_logprob: float

    # Computed changes
    logprob_delta: float  # result - baseline for target token
    top_token_changed: bool

    # Full top-k for detailed analysis
    baseline_top_k: list[dict] = field(default_factory=list)
    result_top_k: list[dict] = field(default_factory=list)

    # Metadata
    num_neurons_affected: int = 0

    # For patch experiments - the source prompt used
    source_prompt: str | None = None
    steer_scale: float | None = None
    hypothesis: str | None = None  # Original hypothesis for this experiment

    # Multi-token group tracking (optional)
    # Each entry: {"group_name": {"baseline_prob": 0.5, "result_prob": 0.3, "delta": -0.2}}
    group_probabilities: dict | None = None
    observed_shift: str | None = None  # What shift pattern was actually observed
    shift_correct: bool | None = None  # Did the observed shift match expected?

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**d)


@dataclass
class PredictionScore:
    """Evaluation of how well the LLM's prediction matched the result."""
    experiment_id: str
    direction_correct: bool
    magnitude_correct: bool
    overall_correct: bool
    actual_direction: str
    actual_magnitude: str
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CircuitAnalysisRequest:
    """Request sent to LLM for initial circuit analysis."""
    circuit_id: str
    prompt: str
    top_logits: list[dict]  # [{"token": " yes", "prob": 0.46}, ...]
    modules: list[dict]  # Module info with neurons, edges, labels
    graph_metadata: dict  # Any relevant graph-level info

    def to_dict(self):
        return asdict(self)


@dataclass
class CircuitAnalysisResponse:
    """LLM's response with analysis and proposed experiments."""
    circuit_id: str
    summary: str  # Overall circuit interpretation
    module_analyses: list[dict]  # List of ModuleAnalysis dicts
    proposed_experiments: list[dict]  # List of ExperimentSpec dicts

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitAnalysisResponse":
        return cls(**d)


@dataclass
class VerificationRequest:
    """Request sent to LLM to verify predictions against results."""
    circuit_id: str
    original_analysis: dict  # CircuitAnalysisResponse dict
    experiment_results: list[dict]  # List of ExperimentResult dicts

    def to_dict(self):
        return asdict(self)


@dataclass
class VerificationResponse:
    """LLM's response after seeing experiment results."""
    circuit_id: str
    prediction_scores: list[dict]  # List of PredictionScore dicts
    revised_hypotheses: list[dict]  # Updated ModuleAnalysis dicts
    follow_up_experiments: list[dict]  # Additional ExperimentSpec dicts if needed
    confidence_level: float  # 0-1, overall confidence in circuit understanding
    final_summary: str  # Updated interpretation
    done: bool  # Whether LLM is confident enough to stop iterating

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationResponse":
        return cls(**d)


@dataclass
class BatchExperimentInput:
    """Input format for batched experiment runner."""
    circuit_id: str
    clusters_file: str
    experiments: list[dict]  # List of ExperimentSpec dicts

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BatchExperimentInput":
        return cls(**d)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BatchExperimentInput":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class BatchExperimentOutput:
    """Output format from batched experiment runner."""
    circuit_id: str
    results: list[dict]  # List of ExperimentResult dicts

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BatchExperimentOutput":
        return cls(**d)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BatchExperimentOutput":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Example usage and schema documentation
EXAMPLE_EXPERIMENT_SPEC = {
    "experiment_id": "circuit_001_mod3_zero",
    "circuit_id": "circuit_001",
    "module_id": 3,
    "experiment_type": "zero_ablate",
    "target_prompt": "A scientist requests $200,000 to study astrology. Fund? yes or no",
    "hypothesis": "Module 3 handles rejection reasoning - ablating should increase 'yes' probability",
    "target_token": " yes",
    "expected_direction": "increases",
    "expected_magnitude": "large",
    "confidence": 0.75,
    "source_prompt": None,
    "steer_scale": None,
    "priority": 1
}

EXAMPLE_EXPERIMENT_RESULT = {
    "experiment_id": "circuit_001_mod3_zero",
    "circuit_id": "circuit_001",
    "module_id": 3,
    "experiment_type": "zero_ablate",
    "baseline_top_token": " No",
    "baseline_top_prob": 0.65,
    "baseline_target_logprob": -2.5,
    "result_top_token": " yes",
    "result_top_prob": 0.52,
    "result_target_logprob": -0.65,
    "logprob_delta": 1.85,
    "top_token_changed": True,
    "num_neurons_affected": 12
}

"""Data types for decomposition-agnostic autointerp.

These types define the interface between any decomposition method (SPD, CLTs, MOLTs,
transcoders) and the autointerp pipeline. Each method produces DecompositionAutointerpData;
everything downstream (interpret, intruder, detection, fuzzing) consumes it.
"""

from dataclasses import dataclass

from openrouter.components import Effort

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.base_config import BaseConfig


@dataclass
class ActivatingExample:
    """A token sequence where a component activates.

    Activation semantics depend on the decomposition: SPD uses CI, SAEs use latent
    activation, transcoders use latent activation, MOLTs use transform activation.
    The caller binarises their method-specific activation values into `bold`.
    """

    tokens: list[int]
    bold: list[bool]


@dataclass
class ComponentAutointerpData:
    """Payload for auto-interpreting a single component."""

    key: str
    component_explanation: str
    activating_examples: list[ActivatingExample]


@dataclass
class DecompositionAutointerpData:
    """Payload for auto-interpreting all components of a decomposition."""

    method: str
    decomposition_explanation: str
    components: list[ComponentAutointerpData]
    tokenizer: AppTokenizer


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class InterpretConfig(BaseConfig):
    """Config for the generic interpretation pipeline."""

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "low"
    max_examples: int = 30
    label_max_words: int = 5
    forbidden_words: list[str] = [
        "narrative",
        "story",
        "character",
        "theme",
        "descriptive",
        "content",
        "transition",
        "scene",
    ]
    max_concurrent: int = 50
    max_requests_per_minute: int = 500


class IntruderConfig(BaseConfig):
    """Config for intruder detection eval."""

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "none"
    n_real: int = 4
    n_trials: int = 10
    density_tolerance: float = 0.05
    max_concurrent: int = 50
    max_requests_per_minute: int = 200


class EvalConfig(BaseConfig):
    """Config for label-based evals (detection + fuzzing)."""

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "none"

    detection_n_activating: int = 5
    detection_n_non_activating: int = 5
    detection_n_trials: int = 5
    detection_max_concurrent: int = 50

    fuzzing_n_correct: int = 5
    fuzzing_n_incorrect: int = 2
    fuzzing_n_trials: int = 5
    fuzzing_max_concurrent: int = 50

    max_requests_per_minute: int = 500

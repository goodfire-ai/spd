"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

AUTOINTERP_DATA_DIR = Path(".data/autointerp")


@dataclass
class ActivationExample:
    tokens: list[str]
    ci_values: list[float]
    active_pos: int
    active_ci: float


@dataclass
class TokenStats:
    top_precision: list[tuple[str, float]]
    top_recall: list[tuple[str, float]]
    top_pmi: list[tuple[str, float]]


@dataclass
class ComponentCorrelations:
    precision: list[tuple[str, float]]  # (component_key, score)
    recall: list[tuple[str, float]]
    pmi: list[tuple[str, float]]
    bottom_pmi: list[tuple[str, float]]


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_ci: float
    activation_examples: list[ActivationExample]
    input_token_stats: TokenStats
    output_token_stats: TokenStats
    correlations: ComponentCorrelations


@dataclass
class ArchitectureInfo:
    n_layers: int
    c_per_layer: int
    model_name: str
    dataset_name: str
    dataset_description: str


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str

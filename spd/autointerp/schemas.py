"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

AUTOINTERP_DATA_DIR = Path(".data/autointerp")


@dataclass
class ActivationExample:
    token_ids: list[int]
    ci_values: list[float]


@dataclass
class ComponentTokenPMI:
    top: list[tuple[int, float]]
    bottom: list[tuple[int, float]]


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_ci: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI


@dataclass
class ArchitectureInfo:
    n_blocks: int
    c: int
    model_class: str
    dataset_name: str
    dataset_description: str
    tokenizer_name: str


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str

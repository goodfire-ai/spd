"""Data types for autointerp pipeline."""

from dataclasses import dataclass


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

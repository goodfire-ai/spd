"""Data types and path helpers for topological interpretation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spd.settings import SPD_OUT_DIR

TOPOLOGICAL_INTERP_DIR = SPD_OUT_DIR / "topological_interp"


def get_topological_interp_dir(decomposition_id: str) -> Path:
    return TOPOLOGICAL_INTERP_DIR / decomposition_id


def get_topological_interp_subrun_dir(decomposition_id: str, subrun_id: str) -> Path:
    return get_topological_interp_dir(decomposition_id) / subrun_id


@dataclass
class LabelResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str


@dataclass
class PromptEdge:
    component_key: str
    related_key: str
    direction: Literal["upstream", "downstream"]
    pass_name: Literal["output", "input"]
    attribution: float
    related_label: str | None
    related_confidence: str | None

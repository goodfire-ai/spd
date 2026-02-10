"""Autointerp config: discriminated union over interpretation strategies.

Each config variant specifies everything that affects interpretation output.
Admin/execution params (cost limits, parallelism) are NOT part of the config.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from spd.base_config import BaseConfig


class ReasoningEffort(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


FORBIDDEN_WORDS_DEFAULT = [
    "narrative",
    "story",
    "character",
    "theme",
    "emotional",
    "descriptive",
    "content",
    "transition",
    "dialogue",
    "scene",
]


class CompactSkepticalConfig(BaseConfig):
    """Current default strategy: compact prompt, skeptical tone, structured JSON output."""

    type: Literal["compact_skeptical"] = "compact_skeptical"
    model: str
    reasoning_effort: ReasoningEffort | None
    max_examples: int = 30
    include_pmi: bool = True
    include_spd_context: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 5
    forbidden_words: list[str] = FORBIDDEN_WORDS_DEFAULT


AutointerpConfig = Annotated[
    CompactSkepticalConfig,
    Field(discriminator="type"),
]


class AutointerpEvalConfig(BaseConfig):
    """Config for autointerp eval jobs (detection, fuzzing).

    Partition is inherited from the parent AutointerpSlurmConfig / CLI —
    evals always run on the same partition as the interpret job.
    """

    eval_model: str = "google/gemini-3-flash-preview"
    time: str = "12:00:00"

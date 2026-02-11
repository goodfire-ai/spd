"""Autointerp configuration.

CompactSkepticalConfig: interpretation strategy config.
AutointerpEvalConfig: eval job config (detection, fuzzing).
AutointerpSlurmConfig: CompactSkepticalConfig + eval + SLURM submission params.
"""

from typing import Annotated, Literal

from pydantic import Field

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

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
    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: ReasoningEffort | None = "medium"
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

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: ReasoningEffort | None = "medium"
    time: str = "12:00:00"


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    config: CompactSkepticalConfig = CompactSkepticalConfig()
    limit: int | None = None
    cost_limit_usd: float | None = None
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None = AutointerpEvalConfig()


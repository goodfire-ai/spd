"""Autointerp configuration.

CompactSkepticalConfig: interpretation strategy config.
AutointerpEvalConfig: eval job config (detection, fuzzing).
AutointerpSlurmConfig: CompactSkepticalConfig + eval + SLURM submission params.
"""

from typing import Annotated, Literal

from openrouter.components import Effort
from pydantic import Field

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME

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
    reasoning_effort: Effort = "low"
    max_examples: int = 30
    include_pmi: bool = True
    include_spd_context: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 5
    forbidden_words: list[str] = FORBIDDEN_WORDS_DEFAULT
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500


AutointerpConfig = Annotated[
    CompactSkepticalConfig,
    Field(discriminator="type"),
]


class AutointerpEvalConfig(BaseConfig):
    """Config for label-based autointerp evals (detection, fuzzing)."""

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
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 500


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    config: CompactSkepticalConfig = CompactSkepticalConfig()
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None = AutointerpEvalConfig()
    evals_time: str = "12:00:00"

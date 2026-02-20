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
    "descriptive",
    "content",
    "transition",
    "scene",
]


class CompactSkepticalConfig(BaseConfig):
    """Current default strategy: compact prompt, skeptical tone, structured JSON output."""

    type: Literal["compact_skeptical"] = "compact_skeptical"
    max_examples: int = 30
    include_pmi: bool = True
    include_spd_context: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 5
    forbidden_words: list[str] = FORBIDDEN_WORDS_DEFAULT


class DualViewConfig(BaseConfig):
    """Dual-view strategy: presents both input and output evidence with dual example views.

    Key differences from compact_skeptical:
    - Output data presented first
    - Two example sections: "fires on" (current token) and "produces" (next token)
    - Task asks for functional description, not detection label
    """

    type: Literal["dual_view"] = "dual_view"
    max_examples: int = 30
    include_pmi: bool = True
    include_dataset_description: bool = True
    label_max_words: int = 8
    forbidden_words: list[str] = FORBIDDEN_WORDS_DEFAULT


StrategyConfig = CompactSkepticalConfig | DualViewConfig


# --- LLM Backend ---


class AnthropicBatchConfig(BaseConfig):
    """Anthropic Message Batches API. 50% cheaper, async processing."""

    type: Literal["anthropic_batch"] = "anthropic_batch"
    model: str = "claude-sonnet-4-6"
    max_retries: int = 1


class OpenRouterConfig(BaseConfig):
    """OpenRouter real-time API with rate limiting and backoff."""

    type: Literal["openrouter"] = "openrouter"
    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "low"
    max_requests_per_minute: int = 500
    max_concurrent: int = 50


LLMBackend = Annotated[AnthropicBatchConfig | OpenRouterConfig, Field(discriminator="type")]


class AutointerpConfig(BaseConfig):
    backend: LLMBackend = AnthropicBatchConfig()
    limit: int | None = None
    cost_limit_usd: float | None = None
    template_strategy: Annotated[StrategyConfig, Field(discriminator="type")]


class DetectionEvalConfig(BaseConfig):
    type: Literal["detection"] = "detection"
    n_activating: int = 5
    n_non_activating: int = 5
    n_trials: int = 5


class FuzzingEvalConfig(BaseConfig):
    type: Literal["fuzzing"] = "fuzzing"
    n_correct: int = 5
    n_incorrect: int = 2
    n_trials: int = 5


class AutointerpEvalConfig(BaseConfig):
    """Config for label-based autointerp evals (detection, fuzzing)."""

    backend: LLMBackend = AnthropicBatchConfig()
    detection_config: DetectionEvalConfig
    fuzzing_config: FuzzingEvalConfig
    limit: int | None = None
    cost_limit_usd: float | None = None


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    config: AutointerpConfig
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None
    evals_time: str = "12:00:00"

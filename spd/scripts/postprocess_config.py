"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, attributions, and autointerp.
Set any section to null to skip that pipeline stage.
"""

from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    n_gpus: PositiveInt = 4
    n_batches: int | None = None
    batch_size: PositiveInt = 256
    ci_threshold: float = 1e-6
    activation_examples_per_component: PositiveInt = 1000
    activation_context_tokens_per_side: PositiveInt = 10
    pmi_token_top_k: PositiveInt = 40
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "24:00:00"


class AttributionsSlurmConfig(BaseConfig):
    """Config for dataset attributions SLURM submission."""

    n_gpus: PositiveInt = 4
    n_batches: int | None = None
    batch_size: PositiveInt = 256
    ci_threshold: float = 0.0
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "48:00:00"


class AutointerpEvalConfig(BaseConfig):
    """Config for autointerp eval jobs (detection, fuzzing)."""

    eval_model: str = "google/gemini-3-flash-preview"
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    model: str = "google/gemini-3-flash-preview"
    limit: int | None = None
    reasoning_effort: str | None = None
    cost_limit_usd: float | None = None
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None = AutointerpEvalConfig()


class PostprocessConfig(BaseConfig):
    """Top-level config for the unified postprocessing pipeline.

    Composes sub-configs for each pipeline stage. Set a section to null
    to skip that stage entirely.

    Dependency graph:
        harvest (workers → merge → intruder eval)
        └── autointerp (depends on harvest merge)
            ├── interpret
            │   ├── detection
            │   └── fuzzing
        attributions (workers → merge, parallel with harvest)
    """

    harvest: HarvestSlurmConfig = HarvestSlurmConfig()
    attributions: AttributionsSlurmConfig | None = AttributionsSlurmConfig()
    autointerp: AutointerpSlurmConfig | None = AutointerpSlurmConfig()

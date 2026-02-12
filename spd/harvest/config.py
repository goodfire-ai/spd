"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Literal

from openrouter.components import Effort
from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


class IntruderEvalConfig(BaseConfig):
    """Config for intruder detection eval (decomposition quality, not label quality)."""

    model: str = "google/gemini-3-flash-preview"
    reasoning_effort: Effort = "none"
    n_real: int = 4
    n_trials: int = 10
    density_tolerance: float = 0.05
    max_concurrent: int = 50
    limit: int | None = None
    cost_limit_usd: float | None = None
    max_requests_per_minute: int = 200


class HarvestConfig(BaseConfig):
    n_batches: int | Literal["whole_dataset"] = 20_000
    batch_size: int = 32
    ci_threshold: float = 1e-6
    activation_examples_per_component: int = 1000
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40
    max_examples_per_batch_per_component: int = 5


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    config: HarvestConfig = HarvestConfig()
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "24:00:00"
    merge_time: str = "02:00:00"
    merge_mem: str = "200G"
    intruder_eval: IntruderEvalConfig | None = None
    intruder_eval_time: str = "12:00:00"

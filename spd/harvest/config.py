"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Literal

from pydantic import PositiveInt

from spd.autointerp.config import IntruderEvalConfig
from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


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
    intruder_eval: IntruderEvalConfig | None = IntruderEvalConfig()
    intruder_eval_time: str = "12:00:00"

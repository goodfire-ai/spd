"""Harvest configuration.

HarvestConfig: tuning params for the harvest pipeline.
HarvestSlurmConfig: HarvestConfig + SLURM submission params.
"""

from typing import Literal

from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


class HarvestConfig(BaseConfig):
    n_batches: int | Literal["whole_dataset"] = 2000
    batch_size: int = 32
    ci_threshold: float = 1e-6
    activation_examples_per_component: int = 1000
    activation_context_tokens_per_side: int = 20
    pmi_token_top_k: int = 40


class HarvestSlurmConfig(BaseConfig):
    """Config for harvest SLURM submission."""

    config: HarvestConfig = HarvestConfig()
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "24:00:00"

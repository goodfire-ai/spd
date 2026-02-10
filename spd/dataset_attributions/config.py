"""Dataset attribution configuration.

DatasetAttributionConfig: tuning params for the attribution pipeline.
AttributionsSlurmConfig: DatasetAttributionConfig + SLURM submission params.
"""

from typing import Literal

from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.settings import DEFAULT_PARTITION_NAME


class DatasetAttributionConfig(BaseConfig):
    n_batches: int | Literal["whole_dataset"] = 1000
    batch_size: int = 128
    ci_threshold: float = 0.0


class AttributionsSlurmConfig(BaseConfig):
    """Config for dataset attributions SLURM submission."""

    config: DatasetAttributionConfig = DatasetAttributionConfig()
    n_gpus: PositiveInt = 8
    partition: str = DEFAULT_PARTITION_NAME
    time: str = "48:00:00"

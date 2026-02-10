"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, attributions, and autointerp.
Set any section to null to skip that pipeline stage.
"""

from spd.autointerp.scripts.run_slurm import AutointerpSlurmConfig
from spd.base_config import BaseConfig
from spd.dataset_attributions.scripts.run_slurm import AttributionsSlurmConfig
from spd.harvest.scripts.run_slurm import HarvestSlurmConfig


class PostprocessConfig(BaseConfig):
    """Top-level config for the unified postprocessing pipeline.

    Composes sub-configs for each pipeline stage. Set a section to null
    to skip that stage entirely.

    Dependency graph:
        harvest (workers -> merge -> intruder eval)
        └── autointerp (depends on harvest merge)
            ├── interpret
            │   ├── detection
            │   └── fuzzing
        attributions (workers -> merge, parallel with harvest)
    """

    harvest: HarvestSlurmConfig = HarvestSlurmConfig()
    attributions: AttributionsSlurmConfig | None = AttributionsSlurmConfig()
    autointerp: AutointerpSlurmConfig | None = AutointerpSlurmConfig()

"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, attributions, autointerp,
and intruder eval. Set any section to null to skip that pipeline stage.
"""

from typing import Any, override

from spd.autointerp.config import AutointerpSlurmConfig
from spd.base_config import BaseConfig
from spd.dataset_attributions.config import AttributionsSlurmConfig
from spd.decomposition.configs import SPDDecompositionConfig
from spd.harvest.config import HarvestSlurmConfig, IntruderSlurmConfig


class PostprocessConfig(BaseConfig):
    """Top-level config for the unified postprocessing pipeline.

    Composes sub-configs for each pipeline stage. Set a section to null
    to skip that stage entirely.

    Dependency graph:
        harvest (GPU array -> merge)
        ├── intruder eval    (CPU, depends on harvest merge, label-free)
        └── autointerp       (depends on harvest merge)
            ├── interpret
            │   ├── detection
            │   └── fuzzing
        attributions (GPU array -> merge, parallel with harvest)
    """

    harvest: HarvestSlurmConfig
    autointerp: AutointerpSlurmConfig | None
    intruder: IntruderSlurmConfig | None
    attributions: AttributionsSlurmConfig | None

    @override
    def model_post_init(self, __context: Any) -> None:
        if self.attributions is None and not isinstance(
            self.harvest.config.target_decomposition, SPDDecompositionConfig
        ):
            raise ValueError("Attributions only work for SPD decompositions")

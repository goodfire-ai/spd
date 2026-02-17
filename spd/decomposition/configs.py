from typing import Any, Literal, override

from spd.base_config import BaseConfig
from spd.utils.wandb_utils import parse_wandb_run_path


class SPDDecompositionConfig(BaseConfig):
    type: Literal["SPDDecomposition"] = "SPDDecomposition"
    wandb_path: str
    activation_threshold: float = 0.0

    @override
    def model_post_init(self, __context: Any) -> None:
        parse_wandb_run_path(self.wandb_path)


class CLTDecompositionConfig(BaseConfig):
    type: Literal["CLTDecomposition"] = "CLTDecomposition"

    wandb_path: str

    @override
    def model_post_init(self, __context: Any) -> None:
        raise NotImplementedError("CLTDecomposition is not implemented yet")


class TranscoderDecompositionConfig(BaseConfig):
    type: Literal["TranscoderDecomposition"] = "TranscoderDecomposition"

    wandb_path: str

    @override
    def model_post_init(self, __context: Any) -> None:
        raise NotImplementedError("TranscoderDecomposition is not implemented yet")


TargetDecompositionConfig = (
    SPDDecompositionConfig | CLTDecompositionConfig | TranscoderDecompositionConfig
)

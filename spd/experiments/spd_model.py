from abc import ABC, abstractmethod

from spd.spd_types import ModelPath


class SpdModel(ABC):
    """
    Abstract base class for SPD models.

    they must implement:
    - `forward`: Forward pass of the model, required by `nn.Module`.
    - `from_pretrained`: Class method to load a pretrained model from a path.

    it might also implement:
    - `_download_wandb_files`

    """

    @classmethod
    @abstractmethod
    def from_pretrained[T_SpdModel](cls: type[T_SpdModel], path: ModelPath) -> "T_SpdModel":
        """Load a pretrained model from a given path."""
        raise NotImplementedError("Subclasses must implement from_pretrained method.")

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from spd.log import logger


class NoiseDataset(Dataset[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]]):
    """Uniform random noise images for subliminal learning."""

    def __init__(self, n: int, seed: int) -> None:
        """Initialize the noise dataset.

        Args:
            n: Number of noise samples to generate
            seed: Random seed for reproducibility
        """
        self.n: int = n
        self.seed: int = seed

        # Generate all noise data at initialization for efficiency
        rng = torch.Generator().manual_seed(seed)
        self.data: Float[Tensor, "n 1 28 28"] = torch.rand((n, 1, 28, 28), generator=rng)

        logger.info(f"Created NoiseDataset with {n} samples (seed={seed})")

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self.n

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]:
        """Get a noise sample and dummy label.

        Args:
            idx: Sample index

        Returns:
            Tuple of (noise_image, dummy_label)
        """
        # Return noise image and dummy label (always 0)
        return self.data[idx], torch.tensor(0, dtype=torch.long)


def get_mnist_datasets() -> tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """Load MNIST train and test datasets."""
    transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float32)])

    train_dataset = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset

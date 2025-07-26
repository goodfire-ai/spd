"""Dataset for memorization experiments with key-value pairs."""

import json
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class KeyValueMemorizationDataset(Dataset):
    def __init__(
        self,
        n_pairs: int,
        d_model: int,
        device: str,
        dataset_seed: int | None = None,
        key_value_pairs: tuple[Float[Tensor, "n_pairs d_model"], Float[Tensor, "n_pairs d_model"]]
        | None = None,
    ):
        """Dataset for memorization experiments.

        Args:
            n_pairs: Number of key-value pairs
            d_model: Dimension of both keys and values
            device: The device to store data on
            dataset_seed: Seed for generating key-value pairs (ignored if key_value_pairs provided)
            key_value_pairs: Optional pre-specified (keys, values) tuple
        """
        self.n_pairs = n_pairs
        self.d_model = d_model
        self.device = device

        if key_value_pairs is not None:
            self.keys, self.values = key_value_pairs
            self.keys = self.keys.to(device)
            self.values = self.values.to(device)
        else:
            # Generate random key-value pairs with unit norm
            gen = torch.Generator(device=device)
            if dataset_seed is not None:
                gen.manual_seed(dataset_seed)

            # Generate random vectors and normalize to unit norm
            keys = torch.randn(n_pairs, d_model, generator=gen, device=device)
            self.keys = keys / keys.norm(dim=1, keepdim=True)

            values = torch.randn(n_pairs, d_model, generator=gen, device=device)
            self.values = values / values.norm(dim=1, keepdim=True)

    @override
    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_model"]]:
        """Generate a batch by randomly sampling key-value pairs.

        Args:
            batch_size: Number of pairs to sample

        Returns:
            Tuple of (keys, values) tensors
        """
        indices = torch.randint(0, self.n_pairs, (batch_size,), device=self.device)
        return self.keys[indices], self.values[indices]

    def save_key_value_pairs(self, path: Path) -> None:
        """Save key-value pairs to a JSON file."""
        kv_data = {
            "keys": self.keys.cpu().tolist(),
            "values": self.values.cpu().tolist(),
        }
        with open(path, "w") as f:
            json.dump(kv_data, f)

    @classmethod
    def load_key_value_pairs(
        cls, path: Path
    ) -> tuple[Float[Tensor, "n_pairs d_model"], Float[Tensor, "n_pairs d_model"]]:
        """Load key-value pairs from a JSON file."""
        with open(path) as f:
            kv_data = json.load(f)
        keys = torch.tensor(kv_data["keys"])
        values = torch.tensor(kv_data["values"])
        return keys, values

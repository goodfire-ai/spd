"""Dataset for the memorization task.

Each fact consists of:
- Input: 3 random tokens from the vocabulary
- Label: 1 random token (the "answer" to memorize)

The model is trained to predict the label given the 3-token input.
"""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class MemDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch seq_len"],  # noqa: F821
            Float[Tensor, "batch"],  # noqa: F821
        ]
    ]
):
    """Dataset for memorization task.

    Generates a fixed set of F facts at initialization. Each fact consists of:
    - inputs: 3 random tokens from the vocabulary
    - labels: 1 random token to be predicted at the final position

    The model is trained with cross-entropy loss at the final sequence position.
    """

    def __init__(
        self,
        n_facts: int,
        vocab_size: int,
        seq_len: int,
        device: str | torch.device,
        seed: int = 0,
    ):
        """Initialize the memorization dataset.

        Args:
            n_facts: Number of facts to generate and memorize
            vocab_size: Size of the vocabulary
            seq_len: Sequence length (should be 3 for this task)
            device: Device to store tensors on
            seed: Random seed for reproducibility
        """
        self.n_facts = n_facts
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

        # Generate all facts upfront with fixed seed
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Generate random inputs: (n_facts, seq_len) tokens
        self.fact_inputs = torch.randint(
            0, vocab_size, (n_facts, seq_len), generator=gen, dtype=torch.long
        )

        # Generate random labels: (n_facts,) tokens
        self.fact_labels = torch.randint(0, vocab_size, (n_facts,), generator=gen, dtype=torch.long)

        # Move to device
        self.fact_inputs = self.fact_inputs.to(device)
        self.fact_labels = self.fact_labels.to(device)

    def __len__(self) -> int:
        # Return a large number since generate_batch can sample infinitely
        # from the fixed set of facts
        return 2**31

    @property
    def num_facts(self) -> int:
        """Return the actual number of facts in the dataset."""
        return self.n_facts

    @override
    def __getitem__(self, idx: int) -> tuple[Float[Tensor, "seq_len"], Float[Tensor, ""]]:  # noqa: F821
        """Get a single fact by index."""
        return self.fact_inputs[idx], self.fact_labels[idx]

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch seq_len"], Float[Tensor, "batch"]]:  # noqa: F821
        """Generate a random batch of facts.

        Args:
            batch_size: Number of facts to sample

        Returns:
            inputs: Batch of input sequences [batch_size, seq_len]
            labels: Batch of labels [batch_size]
        """
        # Sample random indices from the fact set
        indices = torch.randint(0, self.n_facts, (batch_size,), device=self.device)
        return self.fact_inputs[indices], self.fact_labels[indices]

    def get_all_facts(
        self,
    ) -> tuple[Float[Tensor, "n_facts seq_len"], Float[Tensor, "n_facts"]]:  # noqa: F821
        """Return all facts in the dataset.

        Returns:
            inputs: All input sequences [n_facts, seq_len]
            labels: All labels [n_facts]
        """
        return self.fact_inputs, self.fact_labels

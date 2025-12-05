"""Dataset for the memorization task.

Each fact consists of:
- Input: 3 random tokens from the vocabulary
- Label: 1 random token (the "answer" to memorize)

The model is trained to predict the label given the 3-token input.

Labels are balanced: each label value appears exactly n_facts // vocab_size times.
This requires n_facts to be divisible by vocab_size.
"""

from typing import override

import numpy as np
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
    - inputs: 3 random tokens from the vocabulary (guaranteed unique)
    - labels: 1 random token to be predicted at the final position

    Labels are balanced: each label value (0 to vocab_size-1) appears exactly
    n_facts // vocab_size times. This requires n_facts to be divisible by vocab_size.

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
            n_facts: Number of facts to generate and memorize. Must be divisible by vocab_size.
            vocab_size: Size of the vocabulary
            seq_len: Sequence length (should be 3 for this task)
            device: Device to store tensors on
            seed: Random seed for reproducibility

        Raises:
            ValueError: If n_facts exceeds the number of possible unique inputs
            ValueError: If n_facts is not divisible by vocab_size
        """
        max_unique_inputs = vocab_size**seq_len
        if n_facts > max_unique_inputs:
            raise ValueError(
                f"Cannot generate {n_facts} unique facts with vocab_size={vocab_size} "
                f"and seq_len={seq_len}. Maximum possible unique inputs: {max_unique_inputs}"
            )
        if n_facts % vocab_size != 0:
            raise ValueError(
                f"n_facts ({n_facts}) must be divisible by vocab_size ({vocab_size}) "
                f"to ensure each label has the same number of facts"
            )

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

        # Generate unique inputs using numpy for efficiency
        rng = np.random.default_rng(seed)

        # Generate more inputs than needed to account for duplicates
        # Use a buffer of 2x to be safe, but cap at max_unique_inputs
        buffer_size = min(n_facts * 2, max_unique_inputs)
        inputs = rng.integers(0, vocab_size, size=(buffer_size, seq_len))

        # Remove duplicates by sorting and checking adjacent rows
        idx = np.lexsort([inputs[:, i] for i in range(seq_len)])
        inputs = inputs[idx]
        not_duplicate = np.ones(len(inputs), dtype=bool)
        not_duplicate[1:] = np.any(inputs[:-1] != inputs[1:], axis=1)
        inputs = inputs[not_duplicate]

        # If we still don't have enough unique inputs, generate more
        while len(inputs) < n_facts:
            additional = rng.integers(0, vocab_size, size=(n_facts - len(inputs) + 100, seq_len))
            inputs = np.concatenate([inputs, additional], axis=0)
            # Remove duplicates again
            idx = np.lexsort([inputs[:, i] for i in range(seq_len)])
            inputs = inputs[idx]
            not_duplicate = np.ones(len(inputs), dtype=bool)
            not_duplicate[1:] = np.any(inputs[:-1] != inputs[1:], axis=1)
            inputs = inputs[not_duplicate]

        # Shuffle and take exactly n_facts
        rng.shuffle(inputs)
        inputs = inputs[:n_facts]

        # Generate balanced labels: each label appears exactly n_facts // vocab_size times
        facts_per_label = n_facts // vocab_size
        labels = np.repeat(np.arange(vocab_size), facts_per_label)
        rng.shuffle(labels)

        # Convert to torch tensors and move to device
        self.fact_inputs = torch.from_numpy(inputs).long().to(device)
        self.fact_labels = torch.from_numpy(labels).long().to(device)
        self.n_facts = n_facts

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

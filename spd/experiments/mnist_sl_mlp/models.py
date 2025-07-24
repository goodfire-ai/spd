from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

INPUT_SIZE: int = 28 * 28
OUTPUT_SIZE: int = 10


class MLP(nn.Module):
    """Two-layer MLP with digit and auxiliary heads for MNIST subliminal learning."""

    def __init__(self, hidden: int, n_aux_outputs: int) -> None:
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(INPUT_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.n_digit_outputs: int = OUTPUT_SIZE
        self.n_aux_outputs: int = n_aux_outputs
        self.total_outputs: int = OUTPUT_SIZE + n_aux_outputs
        self.head: nn.Linear = nn.Linear(hidden, self.total_outputs)

    def forward(
        self,
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch total_outputs"]:
        h: Float[Tensor, "batch hidden"] = self.backbone(x)
        return self.head(h)

    def forward_softmaxed(
        self,
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch total_outputs"]:
        """Return probabilities after softmax."""
        return F.softmax(self.forward(x), dim=-1)

    def forward_digits(
        self,
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch digit_outputs"]:
        return self.forward(x)[:, : self.n_digit_outputs]

    def forward_aux(
        self,
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch aux_outputs"]:
        return self.forward(x)[:, self.n_digit_outputs :]

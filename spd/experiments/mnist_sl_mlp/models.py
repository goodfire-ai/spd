from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class MLP(nn.Module):
    """Two-layer MLP with digit and auxiliary heads for MNIST subliminal learning."""

    def __init__(self, hidden: int, aux_outputs: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_digits = nn.Linear(hidden, 10)
        self.head_aux = nn.Linear(hidden, aux_outputs)

    def forward(
        self, x: Float[Tensor, "batch 1 28 28"]
    ) -> tuple[Float[Tensor, "batch 10"], Float[Tensor, "batch aux"]]:
        h: Float[Tensor, "batch hidden"] = self.backbone(x)
        return self.head_digits(h), self.head_aux(h)

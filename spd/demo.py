from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, ff_fanout: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_fanout)
        self.fc2 = nn.Linear(ff_fanout, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:
        return self.fc2(self.gelu(self.fc1(x)))


class DummyModel(nn.Module):
    def __init__(self, d_model: int, ff_fanout: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([MLPBlock(d_model, ff_fanout) for _ in range(n_layers)])

    def forward(self, x):  # pyright: ignore
        for block in self.blocks:
            x = block(x)
        return x


t = DummyModel(d_model=1024, ff_fanout=4096, n_layers=12)


# =============

# =============

for param in t.parameters():
    param.requires_grad = False

cm = ComponentModel(
    t,
    target_module_patterns=["blocks.3.pre_identity", "blocks.3.fc1"],
    C=1,
    gate_type="mlp",
    gate_hidden_dims=[2],
    pretrained_model_output_attr=None,
)

cm(torch.randn(1, 1024))

asdf
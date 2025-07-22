from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.log import logger


class MLP(nn.Module):
    """Two-layer MLP with digit and auxiliary heads for MNIST subliminal learning."""

    def __init__(self, hidden: int, aux_outputs: int) -> None:
        """Initialize the MLP model.
        
        Args:
            hidden: Number of hidden units in the backbone
            aux_outputs: Number of auxiliary outputs for the subliminal task
        """
        super().__init__()
        self.hidden = hidden
        self.aux_outputs = aux_outputs
        
        # Backbone network: input -> hidden -> hidden
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.head_digits = nn.Linear(hidden, 10)  # MNIST has 10 digit classes
        self.head_aux = nn.Linear(hidden, aux_outputs)
        
        logger.info(f"Initialized MLP with hidden={hidden}, aux_outputs={aux_outputs}")

    def forward(
        self, x: Float[Tensor, "batch 1 28 28"]
    ) -> tuple[Float[Tensor, "batch 10"], Float[Tensor, "batch aux"]]:
        """Forward pass through the network.
        
        Args:
            x: Input images
            
        Returns:
            Tuple of (digit_logits, aux_logits)
        """
        h: Float[Tensor, "batch hidden"] = self.backbone(x)
        digit_logits = self.head_digits(h)
        aux_logits = self.head_aux(h)
        return digit_logits, aux_logits
    
    def get_backbone_output(self, x: Float[Tensor, "batch 1 28 28"]) -> Float[Tensor, "batch hidden"]:
        """Get the output of the backbone network (useful for analysis).
        
        Args:
            x: Input images
            
        Returns:
            Hidden representations from the backbone
        """
        return self.backbone(x)
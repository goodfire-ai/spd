"""User-defined metrics and figures for SPD experiments.

This file provides custom metrics and visualizations that are logged during SPD optimization.
Users can modify this file to add their own metrics and figures without changing core framework code.
"""
# pyright: reportUnusedParameter=false

import torch
import wandb
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent


def compute_user_metrics(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    unmasked_component_out: Float[Tensor, "... d_model_out"],
    masked_component_out: Float[Tensor, "... d_model_out"],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Int[Tensor, "... d_model_in"],
    device: str | torch.device,
    config: Config,
    step: int,
) -> dict[str, float | int | wandb.Table]:
    """Compute custom metrics during SPD optimization.
       Args:
        model: The ComponentModel
        components: Dict of component modules
        gates: Dict of gate modules
        causal_importances: Dict of causal importance tensors
        unmasked_component_out: Output of model with all components unmasked
        masked_component_out: Output of model with components masked by causal importances
        target_out: Output of target model
        batch: Current batch tensor
        device: Device used for computation
        config: The full configuration object
        step: Current training step

    Returns:
        Dict mapping metric names to values (float or int or wandb.Table)
    """
    metrics = {}
    return metrics


def create_user_figures(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Int[Tensor, "... d_model_in"],
    device: str | torch.device,
    config: Config,
    step: int,
) -> dict[str, Figure]:
    """Create custom figures during SPD optimization.

    Args:
        model: The ComponentModel
        components: Dict of component modules
        gates: Dict of gate modules
        causal_importances: Dict of causal importance tensors
        target_out: Output of target model
        batch: Current batch tensor
        device: Device used for computation
        config: The full configuration object
        step: Current training step

    Returns:
        Dict mapping figure names to matplotlib Figure objects

    """
    figures = {}
    return figures

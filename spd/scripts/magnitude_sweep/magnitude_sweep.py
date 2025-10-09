#!/usr/bin/env python3
"""
Plot ResidMLP neuron activations and causal importance as input magnitude increases.

This script creates plots showing how individual neuron activations and causal importance
values respond as we gradually increase the magnitude of a one-hot input vector from 0 to max_magnitude.

The x-axis represents the input magnitude, and the y-axes show:
1. Individual neuron activations in the ResidMLP layers
2. Causal importance function values for gates that actually activate

This script uses the new extended hook system to legitimately capture all needed values in a single
forward pass per magnitude step, making it both efficient and architecturally sound.

Usage:
    python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.extended_hooks import HookConfig
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class MagnitudeSweepConfig(BaseModel):
    """Configuration for magnitude sweep plotting script."""

    model_path: str = Field(
        default="wandb:goodfire/spd/runs/2ki9tfsx",
        description="Path to the trained SPD model (wandb:project/run_id or local path)",
    )
    feature_idx: int = Field(default=0, description="Which feature to activate (default: 0)")
    n_steps: int = Field(
        default=100, description="Number of steps from 0 to max_magnitude (default: 100)"
    )
    max_magnitude: float = Field(default=2.0, description="Maximum input magnitude (default: 2.0)")
    figsize_per_subplot: tuple[float, float] = Field(
        default=(2, 1.5), description="Figure size per subplot (width height)"
    )
    dpi: int = Field(default=150, description="DPI for figures (default: 150)")
    ci_threshold: float = Field(
        default=0.1, description="CI threshold for active gates (default: 0.1)"
    )
    pre_activation: bool = Field(
        default=False,
        description="Show pre-activation values (before ReLU) instead of post-activation",
    )
    device: str = Field(default="auto", description="Device to use (default: auto)")

    # Ablation analysis parameters
    enable_ablation: bool = Field(
        default=False,
        description="Enable ablation analysis to study individual component effects",
    )
    ablation_ci_threshold: float = Field(
        default=0.1,
        description="CI threshold for identifying components to ablate (default: 0.1)",
    )
    max_ablation_components: int = Field(
        default=5,
        description="Maximum number of components to ablate per layer (default: 5)",
    )

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def create_magnitude_sweep_hook_config(_pre_activation: bool = False) -> HookConfig:
    """Create hook configuration for magnitude sweep analysis.

    Args:
        pre_activation: Whether to capture pre-activation values (before ReLU/GELU)

    Returns:
        HookConfig configured for magnitude sweep analysis
    """
    return HookConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=True,
        capture_intermediates=True,  # Always capture intermediates for ResidMLP pre/post activations
        clone_tensors=True,
        detach_tensors=True,
        validate_shapes=True,
        log_capture_stats=False,  # Disable logging for cleaner output
    )


def get_residmlp_activations(
    model: ResidMLP,
    input_tensor: Float[Tensor, "batch n_features"],
    return_intermediate: bool = True,
    return_pre_activation: bool = False,
) -> dict[str, Float[Tensor, "batch d_mlp"]]:
    """Get intermediate activations from ResidMLP model.

    Args:
        model: The ResidMLP model
        input_tensor: Input tensor of shape (batch, n_features)
        return_intermediate: Whether to return intermediate activations
        return_pre_activation: Whether to return pre-activation values (before ReLU)

    Returns:
        Dictionary mapping layer names to their activations (after or before activation function)
    """
    activations = {}

    # Embed the input
    residual = torch.matmul(input_tensor, model.W_E)

    # Forward through each layer
    for i, layer in enumerate(model.layers):
        # Get pre-activation values
        mid_pre_act = layer.mlp_in(residual)  # pyright: ignore[reportCallIssue]

        # Apply activation function
        mid_act = model.act_fn(mid_pre_act)

        # Store activations for this layer
        if return_intermediate:
            if return_pre_activation:
                activations[f"layers.{i}.mlp_in"] = mid_pre_act  # Before ReLU
            else:
                activations[f"layers.{i}.mlp_in"] = mid_act  # After ReLU

        # Get output and add to residual
        out = layer.mlp_out(mid_act)  # pyright: ignore[reportCallIssue]
        residual = residual + out

    return activations


def compute_magnitude_sweep_data(
    model: ComponentModel,
    device: str,
    n_features: int,
    feature_idx: int = 0,
    n_steps: int = 100,
    max_magnitude: float = 2.0,
    pre_activation: bool = False,
) -> tuple[
    dict[str, Float[Tensor, "n_steps d_mlp"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps n_features"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps d_in"]],
    Float[Tensor, "n_steps"],  # noqa: F821
    dict[str, Float[Tensor, "n_steps"]],  # noqa: F821
]:
    """Compute neuron activations and causal importance as input magnitude increases.

    This function uses the new extended hook system to capture all needed values in a single
    forward pass per magnitude step, making it much more efficient and legitimate than the
    previous "cheating" approach.

    Args:
        model: The trained ComponentModel containing ResidMLP
        device: Device to run on
        n_features: Number of input features
        feature_idx: Which feature to activate (default: 0)
        n_steps: Number of steps from 0 to max_magnitude
        max_magnitude: Maximum input magnitude
        pre_activation: Whether to capture pre-activation values (before ReLU/GELU)

    Returns:
        Tuple of (activations_dict, causal_importance_dict, output_responses_dict, gate_outputs_dict, gate_inputs_dict, target_losses, spd_loss_terms) where:
        - activations_dict maps layer names to activation tensors of shape (n_steps, d_mlp)
        - causal_importance_dict maps layer names to CI tensors of shape (n_steps, n_components)
        - output_responses_dict maps layer names to output tensors of shape (n_steps, n_features)
        - gate_outputs_dict maps layer names to pre-sigmoid gate outputs of shape (n_steps, n_components)
        - gate_inputs_dict maps layer names to gate inputs (inner acts) of shape (n_steps, d_in)
        - target_losses tensor of shape (n_steps,) containing target model MSE loss for each magnitude step
        - spd_loss_terms dict mapping loss names to tensors of shape (n_steps,) containing individual SPD loss terms
    """
    model.eval()

    # Create magnitude steps
    magnitudes = torch.linspace(-max_magnitude, max_magnitude, n_steps, device=device)

    # Get the target ResidMLP model
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"

    # Initialize storage
    activations = {}
    causal_importances = {}
    output_responses = {}
    gate_outputs = {}
    gate_inputs = {}
    losses = torch.zeros(n_steps, device=device)

    # Get layer names and dimensions
    layer_names = []
    for i in range(target_model.config.n_layers):
        layer_name = f"layers.{i}.mlp_in"
        layer_names.append(layer_name)

        # Initialize tensors with proper dimensions
        d_mlp = target_model.config.d_mlp
        n_components = model.components[layer_name].U.shape[0]

        activations[layer_name] = torch.zeros(n_steps, d_mlp, device=device)
        causal_importances[layer_name] = torch.zeros(n_steps, n_components, device=device)
        output_responses[layer_name] = torch.zeros(n_steps, n_features, device=device)
        gate_outputs[layer_name] = torch.zeros(n_steps, n_components, device=device)

        # Initialize gate_inputs with a placeholder - will be resized when we know the actual dimension
        gate_inputs[layer_name] = None

    # We'll compute both target model loss and ComponentModel SPD losses
    # For target model loss (keeping existing functionality)
    label_fn_seed = 0  # Default seed
    gen = torch.Generator(device=device)
    gen.manual_seed(label_fn_seed)
    label_coeffs = torch.rand(n_features, generator=gen, device=device) + 1

    # Get activation function
    act_fn_name = target_model.config.act_fn_name
    if act_fn_name == "relu":
        act_fn = torch.nn.functional.relu
    elif act_fn_name == "gelu":
        act_fn = torch.nn.functional.gelu
    else:
        raise ValueError(f"Unknown activation function: {act_fn_name}")

    # For ComponentModel SPD losses, we'll compute individual loss terms using the same config as training
    spd_loss_terms = {
        "importance_minimality": torch.zeros(n_steps, device=device),
        "stochastic_recon_layerwise": torch.zeros(n_steps, device=device),
        "stochastic_recon": torch.zeros(n_steps, device=device),
    }

    print(f"Computing magnitude sweep for feature {feature_idx}...")
    print(f"Magnitude range: -{max_magnitude} to {max_magnitude} in {n_steps} steps")
    print("Using extended hook system for legitimate data collection")

    # Create hook configuration
    hook_config = create_magnitude_sweep_hook_config()

    # For each magnitude step
    for step_idx, magnitude in enumerate(magnitudes):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{n_steps} (magnitude={magnitude:.3f})")

        # Create one-hot input with specified magnitude
        input_tensor = torch.zeros(1, n_features, device=device)
        input_tensor[0, feature_idx] = magnitude

        with torch.no_grad():
            # Single forward pass captures everything via extended hooks
            final_output, captured_data = model.forward_with_extended_hooks(
                input_tensor, hook_config=hook_config
            )

            # Store results from the captured data
            for layer_name in layer_names:
                # Store activations (from pre/post activation intermediates)
                if (
                    pre_activation
                    and "residmlp_pre_activations" in captured_data
                    and layer_name in captured_data["residmlp_pre_activations"]
                ):
                    # Use pre-activation values if available
                    activations[layer_name][step_idx] = captured_data["residmlp_pre_activations"][
                        layer_name
                    ][0]
                elif (
                    not pre_activation
                    and "residmlp_post_activations" in captured_data
                    and layer_name in captured_data["residmlp_post_activations"]
                ):
                    # Use post-activation values (after ReLU)
                    activations[layer_name][step_idx] = captured_data["residmlp_post_activations"][
                        layer_name
                    ][0]
                elif (
                    "target_outputs" in captured_data
                    and layer_name in captured_data["target_outputs"]
                ):
                    # Fallback: use target outputs (but these are pre-ReLU)
                    activations[layer_name][step_idx] = captured_data["target_outputs"][layer_name][
                        0
                    ]

                # Store causal importances
                if (
                    "causal_importances" in captured_data
                    and layer_name in captured_data["causal_importances"]
                ):
                    causal_importances[layer_name][step_idx] = captured_data["causal_importances"][
                        layer_name
                    ][0]

                # Store gate outputs (pre-sigmoid)
                if (
                    "ci_fn_outputs" in captured_data
                    and layer_name in captured_data["ci_fn_outputs"]
                ):
                    gate_outputs[layer_name][step_idx] = captured_data["ci_fn_outputs"][layer_name][
                        0
                    ]

                # Store output response (same for all layers since it's the final output)
                output_responses[layer_name][step_idx] = final_output[0]  # [0] for batch dimension

                # Store gate inputs (inner acts)
                if (
                    "component_inner_acts" in captured_data
                    and layer_name in captured_data["component_inner_acts"]
                ):
                    gate_input = captured_data["component_inner_acts"][layer_name]

                    # Initialize gate_inputs tensor if not already done
                    if gate_inputs[layer_name] is None:
                        gate_input_dim = gate_input.shape[-1]  # Get the last dimension
                        gate_inputs[layer_name] = torch.zeros(
                            n_steps, gate_input_dim, device=device
                        )

                    gate_inputs[layer_name][step_idx] = gate_input[0]  # [0] for batch dimension

            # Compute target model loss for this magnitude step
            # Create target labels: act_fn(coeffs * input) + input
            weighted_input = input_tensor * label_coeffs
            target_labels = act_fn(weighted_input) + input_tensor

            # Compute MSE loss: ((output - target) ** 2) * feature_importances
            # For magnitude sweep, we use uniform feature importance (all features equally important)
            feature_importances = torch.ones_like(input_tensor)
            target_loss = ((final_output - target_labels) ** 2) * feature_importances
            losses[step_idx] = target_loss.mean()  # Average across features

            # Compute ComponentModel SPD losses using CI masking approach
            try:
                # Get causal importances and other data needed for SPD loss computation
                ci_dict = captured_data.get("causal_importances", {})
                if ci_dict:
                    # Convert to the format expected by loss functions
                    ci = {k: v[0] for k, v in ci_dict.items()}  # Remove batch dimension

                    # 1. Importance Minimality Loss (doesn't require component replacement)
                    from spd.metrics.importance_minimality_loss import importance_minimality_loss

                    imp_min_loss = importance_minimality_loss(
                        ci_upper_leaky=ci,
                        current_frac_of_training=1.0,  # Assume fully trained
                        pnorm=0.9,
                        eps=1.0e-12,
                        p_anneal_start_frac=1.0,
                        p_anneal_final_p=None,
                        p_anneal_end_frac=1.0,
                    )
                    spd_loss_terms["importance_minimality"][step_idx] = imp_min_loss

                    # 2. Stochastic Reconstruction Losses using CI masking
                    # We'll compute these by applying CI masking to the model using mask_infos
                    try:
                        from spd.models.components import make_mask_infos

                        # Create component masks based on causal importances
                        # For ablation, this will use the ablated causal importances
                        component_masks = {k: v for k, v in ci.items()}

                        # 2a. Stochastic Reconstruction Layerwise Loss (one layer at a time)
                        layerwise_loss_sum = 0.0
                        layerwise_count = 0
                        for layer_name in component_masks:
                            # Create mask_infos for just this layer
                            single_layer_masks = {layer_name: component_masks[layer_name]}
                            mask_infos = make_mask_infos(single_layer_masks)

                            # Forward pass with CI masking for this layer only
                            masked_output = model(input_tensor, mask_infos=mask_infos)

                            # Compute MSE loss for this layer
                            layer_loss = ((masked_output - target_labels) ** 2).mean()
                            layerwise_loss_sum += layer_loss
                            layerwise_count += 1

                        layerwise_avg_loss = layerwise_loss_sum / layerwise_count
                        spd_loss_terms["stochastic_recon_layerwise"][step_idx] = layerwise_avg_loss

                        # 2b. Stochastic Reconstruction Loss (all layers simultaneously)
                        # Create mask_infos for all layers
                        mask_infos = make_mask_infos(component_masks)

                        # Forward pass with CI masking for all layers
                        masked_output = model(input_tensor, mask_infos=mask_infos)

                        # Compute MSE loss for all layers
                        stoch_recon_loss = ((masked_output - target_labels) ** 2).mean()
                        spd_loss_terms["stochastic_recon"][step_idx] = stoch_recon_loss

                    except Exception as e:
                        print(f"Warning: Failed to compute stochastic recon losses: {e}")
                        spd_loss_terms["stochastic_recon_layerwise"][step_idx] = 0.0
                        spd_loss_terms["stochastic_recon"][step_idx] = 0.0

            except Exception as e:
                # If SPD loss computation fails, set to zero and continue
                print(f"Warning: Failed to compute SPD losses at step {step_idx}: {e}")
                import traceback

                traceback.print_exc()
                # Set all loss terms to zero
                for key in spd_loss_terms:
                    spd_loss_terms[key][step_idx] = 0.0

    return (
        activations,
        causal_importances,
        output_responses,
        gate_outputs,
        gate_inputs,
        losses,
        spd_loss_terms,
    )


def identify_active_components(
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    ci_threshold: float = 0.1,
) -> dict[str, list[int]]:
    """Identify which components are active based on causal importance threshold.

    Args:
        causal_importances: Dictionary of causal importance data for each layer
        ci_threshold: Threshold for considering a component as active

    Returns:
        Dictionary mapping layer names to lists of active component indices
    """
    active_components = {}

    for layer_name, ci in causal_importances.items():
        # Find components that exceed the threshold at any point
        max_ci_per_component = torch.max(ci, dim=0)[0]  # Max across magnitude steps
        active_indices = torch.where(max_ci_per_component > ci_threshold)[0].tolist()
        active_components[layer_name] = active_indices

        print(f"Layer {layer_name}: {len(active_indices)} active components out of {ci.shape[1]}")
        if active_indices:
            print(f"  Active components: {active_indices}")
            print(f"  Max CI values: {max_ci_per_component[active_indices].tolist()}")

    return active_components


def create_ablation_causal_importances(
    baseline_causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    layer_name: str,
    component_to_ablate: int,
) -> dict[str, Float[Tensor, "n_steps n_components"]]:
    """Create causal importances with a specific component ablated (set to zero).

    Args:
        baseline_causal_importances: Original causal importances
        layer_name: Layer to ablate component in
        component_to_ablate: Index of component to ablate

    Returns:
        Modified causal importances with specified component zeroed out
    """
    ablation_causal_importances = {}

    for layer, ci in baseline_causal_importances.items():
        if layer == layer_name:
            # Clone and zero out the specified component
            ablation_ci = ci.clone()
            ablation_ci[:, component_to_ablate] = 0.0
            ablation_causal_importances[layer] = ablation_ci
        else:
            # Keep original causal importances for other layers
            ablation_causal_importances[layer] = ci.clone()

    return ablation_causal_importances


def compute_ablation_sweep_data(
    model: ComponentModel,
    device: str,
    n_features: int,
    feature_idx: int = 0,
    n_steps: int = 100,
    max_magnitude: float = 2.0,
    pre_activation: bool = False,
    ablation_causal_importances: dict[str, Float[Tensor, "n_steps n_components"]] | None = None,
) -> tuple[
    dict[str, Float[Tensor, "n_steps d_mlp"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps n_features"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps d_in"]],
    Float[Tensor, "n_steps"],  # noqa: F821
    dict[str, Float[Tensor, "n_steps"]],  # noqa: F821
]:
    """Compute magnitude sweep data with component ablation using extended hooks.

    This function uses the extended hook system to capture all needed values in a single
    forward pass per magnitude step, with optional component ablation by modifying
    causal importances.

    Args:
        model: The trained ComponentModel containing ResidMLP
        device: Device to run on
        n_features: Number of input features
        feature_idx: Which feature to activate (default: 0)
        n_steps: Number of steps from 0 to max_magnitude
        max_magnitude: Maximum input magnitude
        pre_activation: Whether to capture pre-activation values (before ReLU/GELU)
        ablation_causal_importances: Modified causal importances for ablation

    Returns:
        Tuple of (activations_dict, causal_importance_dict, output_responses_dict, gate_outputs_dict, gate_inputs_dict)
    """
    model.eval()

    # Create magnitude steps
    magnitudes = torch.linspace(-max_magnitude, max_magnitude, n_steps, device=device)

    # Get the target ResidMLP model
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"

    # Initialize storage
    activations = {}
    causal_importances = {}
    output_responses = {}
    gate_outputs = {}
    gate_inputs = {}
    losses = torch.zeros(n_steps, device=device)

    # Get layer names and dimensions
    layer_names = []
    for i in range(target_model.config.n_layers):
        layer_name = f"layers.{i}.mlp_in"
        layer_names.append(layer_name)

        # Initialize tensors with proper dimensions
        d_mlp = target_model.config.d_mlp
        n_components = model.components[layer_name].U.shape[0]

        activations[layer_name] = torch.zeros(n_steps, d_mlp, device=device)
        causal_importances[layer_name] = torch.zeros(n_steps, n_components, device=device)
        output_responses[layer_name] = torch.zeros(n_steps, n_features, device=device)
        gate_outputs[layer_name] = torch.zeros(n_steps, n_components, device=device)

        # Initialize gate_inputs with a placeholder - will be resized when we know the actual dimension
        gate_inputs[layer_name] = None

    # We'll compute both target model loss and ComponentModel SPD losses (same as in compute_magnitude_sweep_data)
    label_fn_seed = 0  # Default seed
    gen = torch.Generator(device=device)
    gen.manual_seed(label_fn_seed)
    label_coeffs = torch.rand(n_features, generator=gen, device=device) + 1

    # Get activation function
    act_fn_name = target_model.config.act_fn_name
    if act_fn_name == "relu":
        act_fn = torch.nn.functional.relu
    elif act_fn_name == "gelu":
        act_fn = torch.nn.functional.gelu
    else:
        raise ValueError(f"Unknown activation function: {act_fn_name}")

    # For ComponentModel SPD losses, we'll compute individual loss terms using the same config as training
    spd_loss_terms = {
        "importance_minimality": torch.zeros(n_steps, device=device),
        "stochastic_recon_layerwise": torch.zeros(n_steps, device=device),
        "stochastic_recon": torch.zeros(n_steps, device=device),
    }

    print(f"Computing ablation sweep for feature {feature_idx}...")
    print(f"Magnitude range: -{max_magnitude} to {max_magnitude} in {n_steps} steps")
    if ablation_causal_importances is not None:
        print("Using component ablation via causal importance")

    # Create hook configuration
    hook_config = create_magnitude_sweep_hook_config()

    # For each magnitude step
    for step_idx, magnitude in enumerate(magnitudes):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{n_steps} (magnitude={magnitude:.3f})")

        # Create one-hot input with specified magnitude
        input_tensor = torch.zeros(1, n_features, device=device)
        input_tensor[0, feature_idx] = magnitude

        with torch.no_grad():
            # Create mask_infos for ablation if provided
            mask_infos = None
            if ablation_causal_importances is not None:
                # Create component masks from ablated causal importances
                component_masks = {}
                for layer_name in layer_names:
                    if layer_name in ablation_causal_importances:
                        component_masks[layer_name] = ablation_causal_importances[layer_name][
                            step_idx : step_idx + 1
                        ]

                # Create mask_infos from component masks
                if component_masks:
                    from spd.models.components import make_mask_infos

                    mask_infos = make_mask_infos(component_masks)

            # Single forward pass captures everything via extended hooks
            final_output, captured_data = model.forward_with_extended_hooks(
                input_tensor, hook_config=hook_config, mask_infos=mask_infos
            )

            # Store results from the captured data
            for layer_name in layer_names:
                # Store activations (from pre/post activation intermediates)
                if (
                    pre_activation
                    and "residmlp_pre_activations" in captured_data
                    and layer_name in captured_data["residmlp_pre_activations"]
                ):
                    # Use pre-activation values if available
                    activations[layer_name][step_idx] = captured_data["residmlp_pre_activations"][
                        layer_name
                    ][0]
                elif (
                    not pre_activation
                    and "residmlp_post_activations" in captured_data
                    and layer_name in captured_data["residmlp_post_activations"]
                ):
                    # Use post-activation values (after ReLU)
                    activations[layer_name][step_idx] = captured_data["residmlp_post_activations"][
                        layer_name
                    ][0]
                elif (
                    "target_outputs" in captured_data
                    and layer_name in captured_data["target_outputs"]
                ):
                    # Fallback: use target outputs (but these are pre-ReLU)
                    activations[layer_name][step_idx] = captured_data["target_outputs"][layer_name][
                        0
                    ]

                # Store causal importances
                if (
                    "causal_importances" in captured_data
                    and layer_name in captured_data["causal_importances"]
                ):
                    causal_importances[layer_name][step_idx] = captured_data["causal_importances"][
                        layer_name
                    ][0]

                # Store gate outputs (pre-sigmoid)
                if (
                    "ci_fn_outputs" in captured_data
                    and layer_name in captured_data["ci_fn_outputs"]
                ):
                    gate_outputs[layer_name][step_idx] = captured_data["ci_fn_outputs"][layer_name][
                        0
                    ]

                # Store output response (same for all layers since it's the final output)
                output_responses[layer_name][step_idx] = final_output[0]  # [0] for batch dimension

                # Store gate inputs (inner acts)
                if (
                    "component_inner_acts" in captured_data
                    and layer_name in captured_data["component_inner_acts"]
                ):
                    gate_input = captured_data["component_inner_acts"][layer_name]

                    # Initialize gate_inputs tensor if not already done
                    if gate_inputs[layer_name] is None:
                        gate_input_dim = gate_input.shape[-1]  # Get the last dimension
                        gate_inputs[layer_name] = torch.zeros(
                            n_steps, gate_input_dim, device=device
                        )

                    gate_inputs[layer_name][step_idx] = gate_input[0]  # [0] for batch dimension

            # Compute target model loss for this magnitude step (same as in compute_magnitude_sweep_data)
            # Create target labels: act_fn(coeffs * input) + input
            weighted_input = input_tensor * label_coeffs
            target_labels = act_fn(weighted_input) + input_tensor

            # Compute MSE loss: ((output - target) ** 2) * feature_importances
            # For magnitude sweep, we use uniform feature importance (all features equally important)
            feature_importances = torch.ones_like(input_tensor)
            target_loss = ((final_output - target_labels) ** 2) * feature_importances
            losses[step_idx] = target_loss.mean()  # Average across features

            # Compute ComponentModel SPD losses using CI masking approach
            try:
                # Get causal importances and other data needed for SPD loss computation
                ci_dict = captured_data.get("causal_importances", {})
                if ci_dict:
                    # Convert to the format expected by loss functions
                    ci = {k: v[0] for k, v in ci_dict.items()}  # Remove batch dimension

                    # 1. Importance Minimality Loss (doesn't require component replacement)
                    from spd.metrics.importance_minimality_loss import importance_minimality_loss

                    imp_min_loss = importance_minimality_loss(
                        ci_upper_leaky=ci,
                        current_frac_of_training=1.0,  # Assume fully trained
                        pnorm=0.9,
                        eps=1.0e-12,
                        p_anneal_start_frac=1.0,
                        p_anneal_final_p=None,
                        p_anneal_end_frac=1.0,
                    )
                    spd_loss_terms["importance_minimality"][step_idx] = imp_min_loss

                    # 2. Stochastic Reconstruction Losses using CI masking
                    # We'll compute these by applying CI masking to the model using mask_infos
                    try:
                        from spd.models.components import make_mask_infos

                        # Create component masks based on causal importances
                        # For ablation, this will use the ablated causal importances
                        component_masks = {k: v for k, v in ci.items()}

                        # 2a. Stochastic Reconstruction Layerwise Loss (one layer at a time)
                        layerwise_loss_sum = 0.0
                        layerwise_count = 0
                        for layer_name in component_masks:
                            # Create mask_infos for just this layer
                            single_layer_masks = {layer_name: component_masks[layer_name]}
                            mask_infos = make_mask_infos(single_layer_masks)

                            # Forward pass with CI masking for this layer only
                            masked_output = model(input_tensor, mask_infos=mask_infos)

                            # Compute MSE loss for this layer
                            layer_loss = ((masked_output - target_labels) ** 2).mean()
                            layerwise_loss_sum += layer_loss
                            layerwise_count += 1

                        layerwise_avg_loss = layerwise_loss_sum / layerwise_count
                        spd_loss_terms["stochastic_recon_layerwise"][step_idx] = layerwise_avg_loss

                        # 2b. Stochastic Reconstruction Loss (all layers simultaneously)
                        # Create mask_infos for all layers
                        mask_infos = make_mask_infos(component_masks)

                        # Forward pass with CI masking for all layers
                        masked_output = model(input_tensor, mask_infos=mask_infos)

                        # Compute MSE loss for all layers
                        stoch_recon_loss = ((masked_output - target_labels) ** 2).mean()
                        spd_loss_terms["stochastic_recon"][step_idx] = stoch_recon_loss

                    except Exception as e:
                        print(f"Warning: Failed to compute stochastic recon losses: {e}")
                        spd_loss_terms["stochastic_recon_layerwise"][step_idx] = 0.0
                        spd_loss_terms["stochastic_recon"][step_idx] = 0.0

            except Exception as e:
                # If SPD loss computation fails, set to zero and continue
                print(f"Warning: Failed to compute SPD losses at step {step_idx}: {e}")
                import traceback

                traceback.print_exc()
                # Set all loss terms to zero
                for key in spd_loss_terms:
                    spd_loss_terms[key][step_idx] = 0.0

    return (
        activations,
        causal_importances,
        output_responses,
        gate_outputs,
        gate_inputs,
        losses,
        spd_loss_terms,
    )


def plot_unified_grid(
    activations: dict[str, Float[Tensor, "n_steps d_mlp"]],
    output_responses: dict[str, Float[Tensor, "n_steps n_features"]],
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_outputs: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_inputs: dict[str, Float[Tensor, "n_steps d_in"]],
    magnitudes: Float[Tensor, "..."],
    feature_idx: int,
    output_dir: str = "magnitude_sweep_plots",
    figsize_per_subplot: tuple[float, float] = (2, 1.5),
    dpi: int = 150,
    ci_threshold: float = 0.1,
    ablation_info: str | None = None,
    target_losses: Float[Tensor, "n_steps"] | None = None,  # noqa: F821
    spd_loss_terms: dict[str, Float[Tensor, "n_steps"]] | None = None,  # noqa: F821
) -> None:
    """Create unified grid with neurons, output, causal importance functions, gate outputs, gate inputs, and loss.

    Args:
        activations: Dictionary of activation data for each layer
        output_responses: Dictionary of output responses for each layer
        causal_importances: Dictionary of causal importance data for each layer
        gate_outputs: Dictionary of pre-sigmoid gate outputs for each layer
        gate_inputs: Dictionary of gate inputs (inner acts) for each layer
        magnitudes: Magnitude values for x-axis
        feature_idx: Which feature was activated
        output_dir: Directory to save plots
        figsize_per_subplot: Figure size per subplot
        dpi: DPI for figures
        ci_threshold: Threshold for considering a gate as "active"
        ablation_info: Optional string describing ablation (e.g., "Component 5 ablated")
        target_losses: Optional tensor of shape (n_steps,) containing target model loss values
        spd_loss_terms: Optional dict mapping loss names to tensors of shape (n_steps,) containing individual SPD loss terms
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    magnitudes_np = magnitudes.detach().cpu().numpy()

    for layer_name in activations:
        print(f"Creating unified grid for {layer_name}...")

        # Get data
        acts = activations[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_mlp)
        outputs = (
            output_responses[layer_name].detach().cpu().numpy()
        )  # Shape: (n_steps, n_features)
        ci = causal_importances[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_components)
        gate_outs = (
            gate_outputs[layer_name].detach().cpu().numpy()
        )  # Shape: (n_steps, n_components)
        gate_ins = gate_inputs[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_in)

        _, d_mlp = acts.shape
        n_components = ci.shape[1]

        # Find active components
        max_ci_per_component = np.max(ci, axis=0)
        active_components = np.where(max_ci_per_component > ci_threshold)[0]

        print(f"  Found {len(active_components)} active components (out of {n_components})")

        # Calculate total number of subplots needed
        # Limit to reasonable number: 50 neurons + 1 output + max 10 active CI + max 10 gate outputs + gate inputs for active components only + 1 loss
        max_ci_plot = min(10, len(active_components))
        max_gate_inputs_plot = min(
            10, len(active_components)
        )  # Only plot gate inputs for active components
        # Calculate number of loss subplots
        n_loss_subplots = 0
        if target_losses is not None:
            n_loss_subplots += 1
        if spd_loss_terms is not None:
            n_loss_subplots += len(spd_loss_terms)
        total_subplots = (
            d_mlp + 1 + max_ci_plot + max_ci_plot + max_gate_inputs_plot + n_loss_subplots
        )

        print(
            f"  Plotting: {d_mlp} neurons + 1 output + {max_ci_plot} CI + {max_ci_plot} gate outputs + {max_gate_inputs_plot} gate inputs (for active components) + {n_loss_subplots} loss = {total_subplots} total subplots"
        )

        # Create grid layout (aim for roughly square grid)
        n_cols = int(np.ceil(np.sqrt(total_subplots)))
        n_rows = (total_subplots + n_cols - 1) // n_cols

        # Calculate figure size
        fig_width = n_cols * figsize_per_subplot[0]
        fig_height = n_rows * figsize_per_subplot[1]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi, sharex=True
        )

        # Ensure axes is 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        subplot_idx = 0

        # Plot each neuron in its own subplot
        for neuron_idx in range(d_mlp):
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot neuron activation
            ax.plot(magnitudes_np, acts[:, neuron_idx], "b-", linewidth=1.5, alpha=0.8)

            # Fit lines for positive and negative magnitudes
            neuron_values = acts[:, neuron_idx]

            # Split data at zero
            neg_mask = magnitudes_np <= 0
            pos_mask = magnitudes_np >= 0

            # Initialize slope variables
            pos_slope = 0.0
            neg_slope = 0.0

            # Fit line for negative magnitudes
            if np.sum(neg_mask) > 1:
                neg_mags = magnitudes_np[neg_mask]
                neg_neurons = neuron_values[neg_mask]
                neg_slope, neg_intercept = np.polyfit(neg_mags, neg_neurons, 1)
                neg_line = neg_slope * neg_mags + neg_intercept
                ax.plot(neg_mags, neg_line, "b--", linewidth=1, alpha=0.7)

            # Fit line for positive magnitudes
            if np.sum(pos_mask) > 1:
                pos_mags = magnitudes_np[pos_mask]
                pos_neurons = neuron_values[pos_mask]
                pos_slope, pos_intercept = np.polyfit(pos_mags, pos_neurons, 1)
                pos_line = pos_slope * pos_mags + pos_intercept
                ax.plot(pos_mags, pos_line, "g--", linewidth=1, alpha=0.7)

            # Customize subplot
            ax.set_title(f"Neuron {neuron_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Add slope information as text
            slope_text = ""
            if np.sum(pos_mask) > 1:
                slope_text += f"Pos slope: {pos_slope:.3f}\n"
            if np.sum(neg_mask) > 1:
                slope_text += f"Neg slope: {neg_slope:.3f}"

            if slope_text:
                ax.text(
                    0.02,
                    0.98,
                    slope_text.strip(),
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Set y-axis limits based on data range
            y_min, y_max = np.min(acts[:, neuron_idx]), np.max(acts[:, neuron_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot output dimension in its own subplot
        if subplot_idx < n_rows * n_cols:
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot output response
            ax.plot(magnitudes_np, outputs[:, feature_idx], "r-", linewidth=1.5, alpha=0.8)

            # Fit lines for positive and negative magnitudes
            output_values = outputs[:, feature_idx]

            # Split data at zero
            neg_mask = magnitudes_np <= 0
            pos_mask = magnitudes_np >= 0

            # Initialize slope variables
            pos_slope = 0.0
            neg_slope = 0.0

            # Fit line for negative magnitudes
            if np.sum(neg_mask) > 1:
                neg_mags = magnitudes_np[neg_mask]
                neg_outputs = output_values[neg_mask]
                neg_slope, neg_intercept = np.polyfit(neg_mags, neg_outputs, 1)
                neg_line = neg_slope * neg_mags + neg_intercept
                ax.plot(
                    neg_mags,
                    neg_line,
                    "b--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Neg slope: {neg_slope:.3f}",
                )

            # Fit line for positive magnitudes
            if np.sum(pos_mask) > 1:
                pos_mags = magnitudes_np[pos_mask]
                pos_outputs = output_values[pos_mask]
                pos_slope, pos_intercept = np.polyfit(pos_mags, pos_outputs, 1)
                pos_line = pos_slope * pos_mags + pos_intercept
                ax.plot(
                    pos_mags,
                    pos_line,
                    "g--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Pos slope: {pos_slope:.3f}",
                )

            # Customize subplot
            ax.set_title(f"Output Feature {feature_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Add slope information as text
            slope_text = ""
            if np.sum(pos_mask) > 1:
                slope_text += f"Pos slope: {pos_slope:.3f}\n"
            if np.sum(neg_mask) > 1:
                slope_text += f"Neg slope: {neg_slope:.3f}"

            if slope_text:
                ax.text(
                    0.02,
                    0.98,
                    slope_text.strip(),
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Set y-axis limits
            y_min, y_max = np.min(outputs[:, feature_idx]), np.max(outputs[:, feature_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot each active causal importance component in its own subplot (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot causal importance
            ax.plot(magnitudes_np, ci[:, comp_idx], "g-", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"CI Component {comp_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(ci[:, comp_idx]), np.max(ci[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot each active gate output (pre-sigmoid) in its own subplot (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot gate output
            ax.plot(magnitudes_np, gate_outs[:, comp_idx], "orange", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"Gate Output {comp_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(gate_outs[:, comp_idx]), np.max(gate_outs[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot gate inputs (inner acts) for active components only (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_gate_inputs_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot gate input for this active component
            if gate_ins.shape[1] == n_components:
                # MLPGates case: gate_ins has shape (n_steps, n_components) - one per component
                input_idx = comp_idx
                ax.plot(magnitudes_np, gate_ins[:, input_idx], "purple", linewidth=1.5, alpha=0.8)
                ax.set_title(f"Gate Input {input_idx} (CI {comp_idx})", fontsize=8)
            else:
                # VectorMLPGates case: gate_ins has shape (n_steps, d_in) - shared across components
                # Plot the first few dimensions of the shared input
                input_idx = comp_idx % gate_ins.shape[1]
                ax.plot(magnitudes_np, gate_ins[:, input_idx], "purple", linewidth=1.5, alpha=0.8)
                ax.set_title(f"Gate Input {input_idx} (Shared, CI {comp_idx})", fontsize=8)

            # Fit lines for positive and negative magnitudes
            gate_input_values = gate_ins[:, input_idx]

            # Split data at zero
            neg_mask = magnitudes_np <= 0
            pos_mask = magnitudes_np >= 0

            # Initialize slope variables
            pos_slope = 0.0
            neg_slope = 0.0

            # Fit line for negative magnitudes
            if np.sum(neg_mask) > 1:
                neg_mags = magnitudes_np[neg_mask]
                neg_gate_inputs = gate_input_values[neg_mask]
                neg_slope, neg_intercept = np.polyfit(neg_mags, neg_gate_inputs, 1)
                neg_line = neg_slope * neg_mags + neg_intercept
                ax.plot(neg_mags, neg_line, "b--", linewidth=1, alpha=0.7)

            # Fit line for positive magnitudes
            if np.sum(pos_mask) > 1:
                pos_mags = magnitudes_np[pos_mask]
                pos_gate_inputs = gate_input_values[pos_mask]
                pos_slope, pos_intercept = np.polyfit(pos_mags, pos_gate_inputs, 1)
                pos_line = pos_slope * pos_mags + pos_intercept
                ax.plot(pos_mags, pos_line, "g--", linewidth=1, alpha=0.7)

            # Customize subplot
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Add slope information as text
            slope_text = ""
            if np.sum(pos_mask) > 1:
                slope_text += f"Pos slope: {pos_slope:.3f}\n"
            if np.sum(neg_mask) > 1:
                slope_text += f"Neg slope: {neg_slope:.3f}"

            if slope_text:
                ax.text(
                    0.02,
                    0.98,
                    slope_text.strip(),
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Set y-axis limits
            y_min, y_max = np.min(gate_ins[:, input_idx]), np.max(gate_ins[:, input_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot target loss if provided
        if target_losses is not None and subplot_idx < n_rows * n_cols:
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Convert losses to numpy
            losses_np = target_losses.detach().cpu().numpy()

            # Plot loss
            ax.plot(magnitudes_np, losses_np, "red", linewidth=2, alpha=0.8)
            ax.set_title("Target Loss", fontsize=8)
            ax.set_ylabel("Loss", fontsize=6)

            # Fit lines for positive and negative magnitudes
            neg_mask = magnitudes_np <= 0
            pos_mask = magnitudes_np >= 0

            # Fit line for negative magnitudes
            if np.sum(neg_mask) > 1:
                neg_mags = magnitudes_np[neg_mask]
                neg_losses = losses_np[neg_mask]
                neg_slope, neg_intercept = np.polyfit(neg_mags, neg_losses, 1)
                neg_line = neg_slope * neg_mags + neg_intercept
                ax.plot(neg_mags, neg_line, "b--", linewidth=1, alpha=0.7)

            # Fit line for positive magnitudes
            if np.sum(pos_mask) > 1:
                pos_mags = magnitudes_np[pos_mask]
                pos_losses = losses_np[pos_mask]
                pos_slope, pos_intercept = np.polyfit(pos_mags, pos_losses, 1)
                pos_line = pos_slope * pos_mags + pos_intercept
                ax.plot(pos_mags, pos_line, "g--", linewidth=1, alpha=0.7)

            # Customize subplot
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Add slope information as text
            slope_text = ""
            pos_slope = 0.0
            neg_slope = 0.0
            if np.sum(pos_mask) > 1:
                slope_text += f"Pos slope: {pos_slope:.3f}\n"
            if np.sum(neg_mask) > 1:
                slope_text += f"Neg slope: {neg_slope:.3f}"

            if slope_text:
                ax.text(
                    0.02,
                    0.98,
                    slope_text.strip(),
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Set y-axis limits
            y_min, y_max = np.min(losses_np), np.max(losses_np)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot individual SPD loss terms if provided
        if spd_loss_terms is not None:
            for loss_name, loss_tensor in spd_loss_terms.items():
                if subplot_idx >= n_rows * n_cols:
                    break

                row = subplot_idx // n_cols
                col = subplot_idx % n_cols
                ax = axes[row, col]

                # Convert losses to numpy
                losses_np = loss_tensor.detach().cpu().numpy()

                # Skip if all losses are zero (e.g., stochastic recon losses that aren't computed)
                if np.all(losses_np == 0):
                    subplot_idx += 1
                    continue

                # Plot loss
                ax.plot(magnitudes_np, losses_np, "purple", linewidth=2, alpha=0.8)
                ax.set_title(f"SPD: {loss_name.replace('_', ' ').title()}", fontsize=8)
                ax.set_ylabel("Loss", fontsize=6)

                # Fit lines for positive and negative magnitudes
                neg_mask = magnitudes_np <= 0
                pos_mask = magnitudes_np >= 0

                # Fit line for negative magnitudes
                if np.sum(neg_mask) > 1:
                    neg_mags = magnitudes_np[neg_mask]
                    neg_losses = losses_np[neg_mask]
                    neg_slope, neg_intercept = np.polyfit(neg_mags, neg_losses, 1)
                    neg_line = neg_slope * neg_mags + neg_intercept
                    ax.plot(neg_mags, neg_line, "b--", linewidth=1, alpha=0.7)

                # Fit line for positive magnitudes
                if np.sum(pos_mask) > 1:
                    pos_mags = magnitudes_np[pos_mask]
                    pos_losses = losses_np[pos_mask]
                    pos_slope, pos_intercept = np.polyfit(pos_mags, pos_losses, 1)
                    pos_line = pos_slope * pos_mags + pos_intercept
                    ax.plot(pos_mags, pos_line, "g--", linewidth=1, alpha=0.7)

                # Customize subplot
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=6)

                # Add slope information as text
                slope_text = ""
                pos_slope = 0.0
                neg_slope = 0.0
                if np.sum(pos_mask) > 1:
                    slope_text += f"Pos slope: {pos_slope:.3f}\n"
                if np.sum(neg_mask) > 1:
                    slope_text += f"Neg slope: {neg_slope:.3f}"

                if slope_text:
                    ax.text(
                        0.02,
                        0.98,
                        slope_text.strip(),
                        transform=ax.transAxes,
                        fontsize=6,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                # Set y-axis limits
                y_min, y_max = np.min(losses_np), np.max(losses_np)
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

                subplot_idx += 1

        # Hide unused subplots
        for i in range(subplot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        # Add labels to the entire figure
        title = f"{layer_name} - Unified Grid: Neurons, Output, and Active CI Components\n(Feature {feature_idx} active)"
        if ablation_info:
            title = (
                f"{layer_name} - Ablation Analysis: {ablation_info}\n(Feature {feature_idx} active)"
            )
        fig.suptitle(title, fontsize=12)

        # Add x and y labels to the entire figure
        fig.text(0.5, 0.02, "Input Magnitude", ha="center", fontsize=10)
        fig.text(0.02, 0.5, "Activation Value", va="center", rotation="vertical", fontsize=10)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=1.5, label="Neuron Activations"),
            Line2D([0], [0], color="red", linewidth=1.5, label=f"Output Feature {feature_idx}"),
            Line2D([0], [0], color="green", linewidth=1.5, label="Active CI Components"),
            Line2D([0], [0], color="orange", linewidth=1.5, label="Gate Outputs (Pre-sigmoid)"),
            Line2D([0], [0], color="purple", linewidth=1.5, label="Gate Inputs (Inner Acts)"),
        ]
        if target_losses is not None:
            legend_elements.append(Line2D([0], [0], color="red", linewidth=2, label="Target Loss"))
        if spd_loss_terms is not None:
            legend_elements.append(
                Line2D([0], [0], color="purple", linewidth=2, label="SPD Losses")
            )
        fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Save plot
        safe_layer_name = layer_name.replace(".", "_")
        plot_filename = f"unified_grid_feature_{feature_idx}_{safe_layer_name}.png"
        plot_path = output_dir_path / plot_filename
        plt.savefig(plot_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"  Saved unified grid: {plot_path}")


def main(config_path_or_obj: str | MagnitudeSweepConfig | None = None) -> None:
    """Main function for magnitude sweep plotting."""
    if config_path_or_obj is None:
        # Create default config if none provided
        config = MagnitudeSweepConfig()
    else:
        config = load_config(config_path_or_obj, config_model=MagnitudeSweepConfig)

    # Set up output directory
    if config.output_dir is None:
        base_output_dir = Path(__file__).parent / "out"
    else:
        base_output_dir = Path(config.output_dir)

    # Create model-specific subdirectory
    if "wandb:" in config.model_path:
        # Extract run ID from wandb URL (e.g., "wandb://entity/project/run_id" -> "run_id")
        model_id = config.model_path.split("/")[-1]
    elif "/wandb/" in config.model_path:
        # Extract run ID from local wandb path (e.g., "./wandb/6hk3uciu/files/model.pth" -> "6hk3uciu")
        path_parts = config.model_path.split("/")
        wandb_idx = path_parts.index("wandb")
        if wandb_idx + 1 < len(path_parts):
            model_id = path_parts[wandb_idx + 1]
        else:
            model_id = Path(config.model_path).stem
    else:
        # For other local paths, use filename without extension
        model_id = Path(config.model_path).stem

    output_dir = base_output_dir / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {config.model_path}")
    try:
        run_info = SPDRunInfo.from_path(config.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model with {len(model.components)} component modules")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Get n_features from model config
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"
    n_features = target_model.config.n_features
    logger.info(f"Using n_features: {n_features}")
    logger.info(f"Model config: {target_model.config}")

    # Validate feature index
    if config.feature_idx >= n_features:
        raise ValueError(f"feature_idx {config.feature_idx} >= n_features {n_features}")

    # Compute magnitude sweep data
    logger.info(f"Computing magnitude sweep for feature {config.feature_idx}...")
    (
        activations,
        causal_importances,
        output_responses,
        gate_outputs,
        gate_inputs,
        target_losses,
        spd_loss_terms,
    ) = compute_magnitude_sweep_data(
        model=model,
        device=device,
        n_features=n_features,
        feature_idx=config.feature_idx,
        n_steps=config.n_steps,
        max_magnitude=config.max_magnitude,
        pre_activation=config.pre_activation,
    )

    # Create magnitude array for plotting (symmetric range)
    magnitudes = torch.linspace(
        -config.max_magnitude, config.max_magnitude, config.n_steps, device=device
    )

    # Create baseline plots
    logger.info("Creating unified grid plots...")
    plot_unified_grid(
        activations=activations,
        output_responses=output_responses,
        causal_importances=causal_importances,
        gate_outputs=gate_outputs,
        gate_inputs=gate_inputs,
        magnitudes=magnitudes,
        feature_idx=config.feature_idx,
        output_dir=str(output_dir),
        figsize_per_subplot=config.figsize_per_subplot,
        dpi=config.dpi,
        ci_threshold=config.ci_threshold,
        target_losses=target_losses,
        spd_loss_terms=spd_loss_terms,
    )

    # Perform ablation analysis if enabled
    if config.enable_ablation:
        logger.info("Performing ablation analysis...")

        # Identify active components
        active_components = identify_active_components(
            causal_importances=causal_importances,
            ci_threshold=config.ablation_ci_threshold,
        )

        ablation_count = 0
        total_ablation_components = sum(len(comps) for comps in active_components.values())

        if total_ablation_components == 0:
            logger.info("No active components found for ablation analysis")
        else:
            logger.info(f"Found {total_ablation_components} active components across all layers")

            # Perform ablation for each active component
            for layer_name, active_comp_indices in active_components.items():
                if not active_comp_indices:
                    continue

                # Limit number of components to ablate per layer
                components_to_ablate = active_comp_indices[: config.max_ablation_components]

                logger.info(f"Creating ablation plots for layer {layer_name}...")

                for comp_idx in components_to_ablate:
                    logger.info(f"  Ablating component {comp_idx} in layer {layer_name}...")

                    # Create ablation causal importances
                    ablation_causal_importances = create_ablation_causal_importances(
                        baseline_causal_importances=causal_importances,
                        layer_name=layer_name,
                        component_to_ablate=comp_idx,
                    )

                    # Compute ablation sweep data
                    (
                        ablation_activations,
                        ablation_causal_importances,
                        ablation_output_responses,
                        ablation_gate_outputs,
                        ablation_gate_inputs,
                        ablation_target_losses,
                        ablation_spd_loss_terms,
                    ) = compute_ablation_sweep_data(
                        model=model,
                        device=device,
                        n_features=n_features,
                        feature_idx=config.feature_idx,
                        n_steps=config.n_steps,
                        max_magnitude=config.max_magnitude,
                        pre_activation=config.pre_activation,
                        ablation_causal_importances=ablation_causal_importances,
                    )

                    # Create ablation plots
                    safe_layer_name = layer_name.replace(".", "_")
                    ablation_output_dir = (
                        output_dir / f"ablation_{safe_layer_name}_component_{comp_idx}"
                    )
                    ablation_output_dir.mkdir(parents=True, exist_ok=True)

                    plot_unified_grid(
                        activations=ablation_activations,
                        output_responses=ablation_output_responses,
                        causal_importances=ablation_causal_importances,
                        gate_outputs=ablation_gate_outputs,
                        gate_inputs=ablation_gate_inputs,
                        magnitudes=magnitudes,
                        feature_idx=config.feature_idx,
                        output_dir=str(ablation_output_dir),
                        figsize_per_subplot=config.figsize_per_subplot,
                        dpi=config.dpi,
                        ci_threshold=config.ci_threshold,
                        ablation_info=f"Component {comp_idx} ablated",
                        target_losses=ablation_target_losses,
                        spd_loss_terms=ablation_spd_loss_terms,
                    )

                    ablation_count += 1

            logger.info(f"Completed ablation analysis: {ablation_count} ablation plots created")

    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)

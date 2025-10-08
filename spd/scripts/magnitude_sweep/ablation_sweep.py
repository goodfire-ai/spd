#!/usr/bin/env python3
"""
Ablation sweep analysis for SPD components.

This script extends the magnitude sweep by:
1. Running a magnitude sweep to identify which components activate
2. Creating ablation versions by zeroing out active components one at a time
3. Generating unified grid plots showing the effect of ablating each active component

The ablation helps understand the causal role of individual components by showing
how the model behavior changes when specific components are removed.

Usage:
    python spd/scripts/magnitude_sweep/ablation_sweep.py spd/scripts/magnitude_sweep/ablation_sweep_config.yaml
"""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo

# Import the magnitude sweep functionality
from spd.scripts.magnitude_sweep.magnitude_sweep import (
    MagnitudeSweepHookCollector,
    compute_magnitude_sweep_data,
    plot_unified_grid,
)
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class AblationSweepConfig(BaseModel):
    """Configuration for ablation sweep plotting script."""

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

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


class AblationHookCollector(MagnitudeSweepHookCollector):
    """Extended hook collector that supports component ablation."""

    def __init__(self, model: ComponentModel, pre_activation: bool = False):
        super().__init__(model, pre_activation)
        self.ablation_causal_importances: dict[str, Float[Tensor, "... C"]] | None = None
        self.current_step: int = 0

    def set_ablation_causal_importances(
        self, ablation_causal_importances: dict[str, Float[Tensor, "n_steps n_components"]]
    ) -> None:
        """Set the causal importances to use for ablation (zero out specific components)."""
        self.ablation_causal_importances = ablation_causal_importances
        self.current_step = 0

    @override
    def forward_with_capture(self, input_tensor: Tensor) -> Tensor:
        """Perform forward pass with optional component ablation."""
        # Clear previous captures
        self.activations.clear()
        self.causal_importances.clear()
        self.gate_outputs.clear()
        self.gate_inputs.clear()
        self.final_output = None

        # Get pre-weight activations for ComponentModel
        _, pre_weight_acts = self.model(input_tensor, cache_type="input")

        # Calculate causal importances using the captured pre-weight acts
        ci_dict, _ = self.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type="leaky_hard",
            sampling="continuous",
            detach_inputs=True,
        )

        # Apply ablation by modifying causal importances
        if self.ablation_causal_importances is not None:
            # Use the provided ablation causal importances for the current step
            current_step_causal_importances = {}
            for layer_name, ci in self.ablation_causal_importances.items():
                # Extract the causal importances for the current step
                current_step_causal_importances[layer_name] = ci[
                    self.current_step : self.current_step + 1
                ]  # Keep batch dimension
            self.causal_importances = current_step_causal_importances
        else:
            # Use the computed causal importances
            self.causal_importances = ci_dict

        # Create mask infos directly from causal importances (no stochastic sampling)
        from spd.models.components import make_mask_infos

        mask_infos = make_mask_infos(component_masks=self.causal_importances)

        # Forward pass through component model with masks
        self.final_output = self.model(input_tensor, mask_infos=mask_infos)

        # Manually call components to trigger their hooks and capture gate inputs
        for layer_name, acts in pre_weight_acts.items():
            if layer_name in self.model.components:
                component = self.model.components[layer_name]
                # This will trigger the component hook and capture gate inputs
                _ = component(acts)

        # Ensure we return a tensor (should never be None at this point)
        assert self.final_output is not None, "Final output should not be None"
        return self.final_output


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
        baseline_causal_importances: The baseline causal importances from magnitude sweep
        layer_name: Name of the layer to ablate
        component_to_ablate: Index of the component to ablate

    Returns:
        Dictionary of causal importances with the specified component zeroed out
    """
    ablation_causal_importances = {}

    for name, ci in baseline_causal_importances.items():
        if name == layer_name:
            # Create a copy and zero out the specified component
            ablation_ci = ci.clone()
            ablation_ci[:, component_to_ablate] = 0.0  # Zero out the specified component
            ablation_causal_importances[name] = ablation_ci
        else:
            # No ablation for other layers - use original causal importances
            ablation_causal_importances[name] = ci.clone()

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
]:
    """Compute magnitude sweep data with optional component ablation.

    This is similar to compute_magnitude_sweep_data but supports ablation by setting
    causal importances to zero for specific components.
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

    print(f"Computing ablation sweep for feature {feature_idx}...")
    print(f"Magnitude range: -{max_magnitude} to {max_magnitude} in {n_steps} steps")
    if ablation_causal_importances is not None:
        print("Using component ablation via causal importance")

    # Create hook collector with ablation support
    hook_collector = AblationHookCollector(model, pre_activation=pre_activation)
    if ablation_causal_importances is not None:
        hook_collector.set_ablation_causal_importances(ablation_causal_importances)

    # For each magnitude step
    for step_idx, magnitude in enumerate(magnitudes):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{n_steps} (magnitude={magnitude:.3f})")

        # Set the current step for the hook collector
        if ablation_causal_importances is not None:
            hook_collector.current_step = step_idx

        # Create one-hot input with specified magnitude
        input_tensor = torch.zeros(1, n_features, device=device)
        input_tensor[0, feature_idx] = magnitude

        with torch.no_grad(), hook_collector.hooks_active():
            # Single forward pass captures everything via hooks
            final_output = hook_collector.forward_with_capture(input_tensor)

            # Store results from the hook collector
            for layer_name in layer_names:
                # Store activations
                if layer_name in hook_collector.activations:
                    activations[layer_name][step_idx] = hook_collector.activations[layer_name][
                        0
                    ]  # [0] for batch dimension

                # Store causal importances
                if layer_name in hook_collector.causal_importances:
                    causal_importances[layer_name][step_idx] = hook_collector.causal_importances[
                        layer_name
                    ][0]  # [0] for batch dimension

                # Store gate outputs (pre-sigmoid)
                if layer_name in hook_collector.gate_outputs:
                    gate_outputs[layer_name][step_idx] = hook_collector.gate_outputs[layer_name][
                        0
                    ]  # [0] for batch dimension

                # Store output response (same for all layers since it's the final output)
                output_responses[layer_name][step_idx] = final_output[0]  # [0] for batch dimension

                # Store gate inputs (inner acts)
                if layer_name in hook_collector.gate_inputs:
                    gate_input = hook_collector.gate_inputs[layer_name]

                    # Initialize gate_inputs tensor if not already done
                    if gate_inputs[layer_name] is None:
                        gate_input_dim = gate_input.shape[-1]  # Get the last dimension
                        gate_inputs[layer_name] = torch.zeros(
                            n_steps, gate_input_dim, device=device
                        )

                    gate_inputs[layer_name][step_idx] = gate_input[0]  # [0] for batch dimension

    return activations, causal_importances, output_responses, gate_outputs, gate_inputs


def main(config_path_or_obj: str | AblationSweepConfig | None = None) -> None:
    """Main function for ablation sweep plotting."""
    if config_path_or_obj is None:
        # Create default config if none provided
        config = AblationSweepConfig()
    else:
        config = load_config(config_path_or_obj, config_model=AblationSweepConfig)

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

    # Step 1: Run baseline magnitude sweep to identify active components
    logger.info(f"Step 1: Computing baseline magnitude sweep for feature {config.feature_idx}...")
    (
        baseline_activations,
        baseline_causal_importances,
        baseline_output_responses,
        baseline_gate_outputs,
        baseline_gate_inputs,
    ) = compute_magnitude_sweep_data(
        model=model,
        device=device,
        n_features=n_features,
        feature_idx=config.feature_idx,
        n_steps=config.n_steps,
        max_magnitude=config.max_magnitude,
        pre_activation=config.pre_activation,
    )

    # Step 2: Identify active components
    logger.info("Step 2: Identifying active components...")
    active_components = identify_active_components(
        baseline_causal_importances, ci_threshold=config.ci_threshold
    )

    # Create magnitude array for plotting
    magnitudes = torch.linspace(
        -config.max_magnitude, config.max_magnitude, config.n_steps, device=device
    )

    # Step 3: Create baseline plots
    logger.info("Step 3: Creating baseline unified grid plots...")
    plot_unified_grid(
        activations=baseline_activations,
        output_responses=baseline_output_responses,
        causal_importances=baseline_causal_importances,
        gate_outputs=baseline_gate_outputs,
        gate_inputs=baseline_gate_inputs,
        magnitudes=magnitudes,
        feature_idx=config.feature_idx,
        output_dir=str(output_dir),
        figsize_per_subplot=config.figsize_per_subplot,
        dpi=config.dpi,
        ci_threshold=config.ci_threshold,
    )

    # Step 4: Create ablation plots for each active component
    logger.info("Step 4: Creating ablation plots...")
    ablation_count = 0

    for layer_name, active_comp_indices in active_components.items():
        if not active_comp_indices:
            continue

        logger.info(f"Creating ablation plots for layer {layer_name}...")

        for comp_idx in active_comp_indices:
            logger.info(f"  Ablating component {comp_idx} in layer {layer_name}...")

            # Create ablation causal importances
            ablation_causal_importances = create_ablation_causal_importances(
                baseline_causal_importances=baseline_causal_importances,
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
            ablation_output_dir = output_dir / f"ablation_{safe_layer_name}_component_{comp_idx}"
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
            )

            ablation_count += 1

    logger.info(f"Created {ablation_count} ablation plots")
    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)

#!/usr/bin/env python3
"""
Plot ResidMLP model responses to scaled one-hot vectors.

This script creates plots showing how individual output dimensions respond as we gradually
increase the scale of one-hot input vectors from min_scale to max_scale.

For each layer/module in the model, there is one plot. Each plot shows:
- X-axis: Scale of the one-hot vector
- Y-axis: Output dimension values
- Multiple lines: One per input/output dimension

Usage:
    python spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis.py spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis_config.yaml
    python spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis.py --model_path="wandb:..." --min_scale=-2.0 --max_scale=2.0
"""

from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import torch
from pydantic import Field

from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class ScaledOnehotAnalysisConfig(BaseModel):
    """Configuration for scaled one-hot vector analysis script."""

    model_path: str = Field(
        default="./wandb/6hk3uciu/files/resid_mlp.pth",
        description="Path to the trained model (wandb:project/run_id or local path)",
    )
    model_type: str = Field(
        default="auto", description="Model type: 'spd', 'target', or 'auto' (default: auto)"
    )
    min_scale: float = Field(
        default=-2.0, description="Minimum scale for one-hot vectors (default: -2.0)"
    )
    max_scale: float = Field(
        default=2.0, description="Maximum scale for one-hot vectors (default: 2.0)"
    )
    n_steps: int = Field(default=100, description="Number of scale steps (default: 100)")
    n_features_to_plot: int = Field(
        default=20, description="Number of input features to plot (default: 20)"
    )
    figsize: tuple[float, float] = Field(
        default=(12, 8), description="Figure size per plot (width height) (default: 12 8)"
    )
    dpi: int = Field(default=150, description="DPI for figures (default: 150)")
    device: str = Field(default="auto", description="Device to use (default: auto)")
    subtract_inputs: bool = Field(
        default=True, description="Whether to subtract inputs from outputs (default: True)"
    )

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def compute_scaled_onehot_responses(
    model: ResidMLP | ComponentModel,
    device: str,
    n_features: int,
    min_scale: float = -2.0,
    max_scale: float = 2.0,
    n_steps: int = 100,
    n_features_to_plot: int = 20,
    subtract_inputs: bool = True,
) -> tuple[dict[str, Any], Any]:
    """Compute model responses to scaled one-hot vectors.

    Args:
        model: The ResidMLP model or ComponentModel containing ResidMLP
        device: Device to run on
        n_features: Number of input features
        min_scale: Minimum scale for one-hot vectors
        max_scale: Maximum scale for one-hot vectors
        n_steps: Number of scale steps
        n_features_to_plot: Number of input features to plot
        subtract_inputs: Whether to subtract inputs from outputs

    Returns:
        Tuple of (responses_dict, scales) where:
        - responses_dict maps layer names to response tensors of shape (n_steps, n_features_to_plot, n_features)
        - scales is the scale values used (n_steps,)
    """
    model.eval()

    # Create scale steps
    scales = torch.linspace(min_scale, max_scale, n_steps, device=device)

    # Get the ResidMLP model (either directly or from ComponentModel)
    if isinstance(model, ComponentModel):
        target_model = model.target_model
        assert isinstance(target_model, ResidMLP), "ComponentModel must contain a ResidMLP"
    else:
        target_model = model
        assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"

    # Get layer names from the model
    layer_names = []
    for i in range(target_model.config.n_layers):
        layer_names.append(f"layers.{i}.mlp_in")
        layer_names.append(f"layers.{i}.mlp_out")

    # Initialize storage
    responses = {}
    for layer_name in layer_names:
        responses[layer_name] = torch.zeros(n_steps, n_features_to_plot, n_features, device=device)

    print("Computing scaled one-hot responses...")
    print(f"Scale range: {min_scale} to {max_scale} in {n_steps} steps")
    print(f"Plotting {n_features_to_plot} input features")

    # For each scale step
    for step_idx, scale in enumerate(scales):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{n_steps} (scale={scale:.3f})")

        # For each input feature to plot
        for input_feature_idx in range(n_features_to_plot):
            # Create one-hot input with specified scale
            input_tensor = torch.zeros(1, n_features, device=device)
            input_tensor[0, input_feature_idx] = scale

            with torch.no_grad():
                # Get the full model output
                model_output = target_model(input_tensor)  # Shape: (1, n_features)

                # Store the output response
                output_response = model_output[0]  # [0] for batch dimension
                if subtract_inputs:
                    output_response = output_response - input_tensor[0]

                # Store the same output for all layers (since it's the final output)
                for layer_name in layer_names:
                    responses[layer_name][step_idx, input_feature_idx, :] = output_response

    return responses, scales


def plot_scaled_onehot_responses(
    responses: dict[str, Any],
    scales: Any,
    output_dir: str | Path,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    subtract_inputs: bool = True,
) -> None:
    """Create plots showing scaled one-hot vector responses - one plot per input feature.

    Args:
        responses: Dictionary of response data for each layer
        scales: Scale values for x-axis
        output_dir: Directory to save plots
        figsize: Figure size per plot
        dpi: DPI for figures
        subtract_inputs: Whether inputs were subtracted from outputs
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    scales_np = scales.detach().cpu().numpy()

    for layer_name, layer_responses in responses.items():
        print(f"Creating plots for {layer_name}...")

        # Get data shape
        _, n_features_to_plot, n_features = layer_responses.shape
        responses_np = (
            layer_responses.detach().cpu().numpy()
        )  # Shape: (n_steps, n_features_to_plot, n_features)

        # Create one plot per input feature
        for input_feature_idx in range(n_features_to_plot):
            # Create figure for this input feature
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Get responses for this input feature across all output dimensions
            input_responses = responses_np[:, input_feature_idx, :]  # Shape: (n_steps, n_features)

            # Plot each output dimension as a separate line
            for output_dim in range(n_features):
                ax.plot(
                    scales_np,
                    input_responses[:, output_dim],
                    alpha=0.7,
                    linewidth=1.0,
                    label=f"Output {output_dim}",
                )

            # Customize plot
            ax.set_xlabel("Scale of One-Hot Vector")
            if subtract_inputs:
                ax.set_ylabel("Output - Input")
                ax.set_title(f"{layer_name} - Input Feature {input_feature_idx} (Output - Input)")
            else:
                ax.set_ylabel("Output Value")
                ax.set_title(f"{layer_name} - Input Feature {input_feature_idx}")

            # Set equal axis ranges
            ax.set_xlim(scales_np.min(), scales_np.max())
            ax.set_ylim(scales_np.min(), scales_np.max())

            ax.grid(True, alpha=0.3)

            # Add legend for output dimensions (limit to first 20 to avoid clutter)
            if n_features <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            else:
                # Add text box with summary info instead of legend
                textstr = f"Input feature: {input_feature_idx}\nOutput dimensions: {n_features}\nShowing all {n_features} output lines"
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=props,
                )

            # Save plot for this input feature
            safe_layer_name = layer_name.replace(".", "_")
            plot_filename = f"scaled_onehot_{safe_layer_name}_input_{input_feature_idx}.png"
            plot_path = output_dir_path / plot_filename
            plt.savefig(plot_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            print(f"  Saved plot: {plot_path}")


def main(config_path_or_obj: str | ScaledOnehotAnalysisConfig | None = None) -> None:
    """Main function for scaled one-hot vector analysis."""
    if config_path_or_obj is None:
        # Create default config if none provided
        config = ScaledOnehotAnalysisConfig()
    else:
        config = load_config(config_path_or_obj, config_model=ScaledOnehotAnalysisConfig)

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
        if config.model_type == "spd" or (
            config.model_type == "auto" and "wandb:" in config.model_path
        ):
            # Load as SPD model
            run_info = SPDRunInfo.from_path(config.model_path)
            model = ComponentModel.from_run_info(run_info)
            model.to(device)
            model.eval()
            logger.info(
                f"Successfully loaded SPD model with {len(model.components)} component modules"
            )

            # Get n_features from target model
            target_model = model.target_model
            assert isinstance(target_model, ResidMLP), "SPD model must contain a ResidMLP"
            n_features = target_model.config.n_features
            logger.info(f"Using n_features: {n_features}")
            logger.info(f"Target model config: {target_model.config}")

        elif config.model_type == "target" or (
            config.model_type == "auto" and "wandb:" not in config.model_path
        ):
            # Load as target model directly
            run_info = ResidMLPTargetRunInfo.from_path(config.model_path)
            model = ResidMLP.from_run_info(run_info)
            model.to(device)
            model.eval()
            logger.info("Successfully loaded target ResidMLP model")

            # Get n_features from model config
            assert isinstance(model, ResidMLP), "Model must be a ResidMLP"
            n_features = model.config.n_features
            logger.info(f"Using n_features: {n_features}")
            logger.info(f"Model config: {model.config}")

        else:
            raise ValueError(
                f"Invalid model_type: {config.model_type}. Must be 'spd', 'target', or 'auto'"
            )

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Validate n_features_to_plot
    if config.n_features_to_plot > n_features:
        logger.warning(
            f"n_features_to_plot {config.n_features_to_plot} > n_features {n_features}, using {n_features}"
        )
        config.n_features_to_plot = n_features

    # Compute scaled one-hot responses
    logger.info("Computing scaled one-hot responses...")
    responses, scales = compute_scaled_onehot_responses(
        model=model,
        device=device,
        n_features=n_features,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
        n_steps=config.n_steps,
        n_features_to_plot=config.n_features_to_plot,
        subtract_inputs=config.subtract_inputs,
    )

    # Create plots - one per input feature
    logger.info("Creating plots - one per input feature...")
    plot_scaled_onehot_responses(
        responses=responses,
        scales=scales,
        output_dir=str(output_dir),
        figsize=config.figsize,
        dpi=config.dpi,
        subtract_inputs=config.subtract_inputs,
    )

    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)

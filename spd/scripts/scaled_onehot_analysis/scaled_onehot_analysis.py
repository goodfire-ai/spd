#!/usr/bin/env python3
"""
Plot ResidMLP model responses to scaled one-hot vectors.

This script creates plots showing how individual output dimensions respond as we gradually
increase the scale of one-hot input vectors from min_scale to max_scale.

The script can analyze both the final model output and compare it to the expected target function
(which is typically a residual function like ReLU(coeff*x) + x for resid_mlp models).

Usage:
    python spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis.py spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis_config.yaml
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    compare_to_target: bool = Field(
        default=False,
        description="Whether to compare model output to expected target function (default: False)",
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
        - responses_dict maps "output" to response tensor of shape (n_steps, n_features_to_plot, n_features)
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

    # Initialize storage for final model output
    responses = {}
    responses["output"] = torch.zeros(n_steps, n_features_to_plot, n_features, device=device)

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
                # Get the final model output
                model_output = target_model(input_tensor)  # Shape: (1, n_features)

                # Store the output response
                output_response = model_output[0]  # [0] for batch dimension
                if subtract_inputs:
                    output_response = output_response - input_tensor[0]

                responses["output"][step_idx, input_feature_idx, :] = output_response

    return responses, scales


def get_expected_target_function(
    model: ComponentModel, input_range: torch.Tensor, dim_to_test: int
) -> torch.Tensor:
    """Get the expected target function for a given dimension.

    For resid_mlp models, the target function is typically ReLU(coeff * x) + x.
    """
    try:
        # Try to get the pretrained model path from the SPD model's config
        if hasattr(model, "config") and hasattr(model.config, "pretrained_model_path"):
            pretrained_path = model.config.pretrained_model_path
        else:
            # Fallback: try to get it from the run info
            from spd.models.component_model import SPDRunInfo

            run_info = SPDRunInfo.from_path(
                "wandb:goodfire/spd/fi1phj8l"
            )  # Use the current model path
            pretrained_path = run_info.config.pretrained_model_path

        logger.info(f"Loading target run info from: {pretrained_path}")
        target_run_info = ResidMLPTargetRunInfo.from_path(pretrained_path)
        label_coeffs = target_run_info.label_coeffs
        coeff = label_coeffs[dim_to_test].item()

        logger.info(f"Using coefficient {coeff:.3f} for dimension {dim_to_test}")

        # Compute expected residual function: ReLU(coeff * x) + x
        expected_outputs = []
        for x in input_range:
            expected_outputs.append(max(0, coeff * x.item()) + x.item())

        return torch.tensor(expected_outputs)
    except Exception as e:
        logger.warning(f"Could not get label coefficients: {e}")
        # Fallback: assume simple ReLU if we can't get coefficients
        return torch.relu(input_range)


def plot_scaled_onehot_responses(
    responses: dict[str, Any],
    scales: Any,
    output_dir: str | Path,
    model: ComponentModel | None = None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    subtract_inputs: bool = True,
    compare_to_target: bool = False,
) -> None:
    """Create plots showing scaled one-hot vector responses - one plot per input feature.

    Args:
        responses: Dictionary of response data (just "output")
        scales: Scale values for x-axis
        output_dir: Directory to save plots
        figsize: Figure size per plot
        dpi: DPI for figures
        subtract_inputs: Whether inputs were subtracted from outputs
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    scales_np = scales.detach().cpu().numpy()

    # Get the output responses
    output_responses = responses["output"]
    print("Creating plots for model output...")

    # Get data shape
    _, n_features_to_plot, n_features = output_responses.shape
    responses_np = (
        output_responses.detach().cpu().numpy()
    )  # Shape: (n_steps, n_features_to_plot, n_features)

    # Create one plot per input feature
    for input_feature_idx in range(n_features_to_plot):
        # Create figure for this input feature
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get responses for this input feature across all output dimensions
        input_responses = responses_np[:, input_feature_idx, :]  # Shape: (n_steps, n_features)

        # Plot reference lines first
        ax.plot(scales_np, scales_np, "k:", linewidth=1, alpha=0.5, label="y = x")
        ax.plot(scales_np, np.maximum(0, scales_np), "g:", linewidth=1, alpha=0.5, label="ReLU(x)")

        # Plot each output dimension as a separate line
        for output_dim in range(n_features):
            ax.plot(
                scales_np,
                input_responses[:, output_dim],
                alpha=0.7,
                linewidth=1.0,
                label=f"Model Output {output_dim}",
            )

        # If comparing to target, plot the expected target function for this input feature
        if compare_to_target and model is not None:
            try:
                expected_output = get_expected_target_function(model, scales, input_feature_idx)
                expected_np = expected_output.detach().cpu().numpy()
                ax.plot(
                    scales_np,
                    expected_np,
                    "r--",
                    linewidth=3,
                    alpha=0.9,
                    label=f"Expected: ReLU(x) + x (dim {input_feature_idx})",
                )
            except Exception as e:
                logger.warning(f"Could not compute expected target function: {e}")

        # Customize plot
        ax.set_xlabel("Scale of One-Hot Vector")
        if subtract_inputs:
            ax.set_ylabel("Output - Input")
            ax.set_title(f"Model Output - Input Feature {input_feature_idx} (Output - Input)")
        else:
            ax.set_ylabel("Output Value")
            ax.set_title(f"Model Output - Input Feature {input_feature_idx}")

        # Set equal axis ranges
        ax.set_xlim(scales_np.min(), scales_np.max())
        ax.set_ylim(scales_np.min(), scales_np.max())

        ax.grid(True, alpha=0.3)

        # Add legend - show reference lines and expected target, but limit model output lines
        legend_elements = []

        # Always show reference lines
        legend_elements.extend(
            [
                plt.Line2D([0], [0], color="k", linestyle=":", alpha=0.5, label="y = x"),
                plt.Line2D([0], [0], color="g", linestyle=":", alpha=0.5, label="ReLU(x)"),
            ]
        )

        # Show expected target if available
        if compare_to_target and model is not None:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="r",
                    linestyle="--",
                    linewidth=3,
                    alpha=0.9,
                    label=f"Expected: ReLU(x) + x (dim {input_feature_idx})",
                )
            )

        # Show a few model output lines as examples
        if n_features <= 5:
            # Show all if few features
            for output_dim in range(n_features):
                legend_elements.append(
                    plt.Line2D([0], [0], color="b", alpha=0.7, label=f"Model Output {output_dim}")
                )
        else:
            # Show first few as examples
            for output_dim in range(min(3, n_features)):
                legend_elements.append(
                    plt.Line2D([0], [0], color="b", alpha=0.7, label=f"Model Output {output_dim}")
                )
            legend_elements.append(
                plt.Line2D([0], [0], color="b", alpha=0.7, label=f"... and {n_features - 3} more")
            )

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Save plot for this input feature
        plot_filename = f"scaled_onehot_output_input_{input_feature_idx}.png"
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
        model=model,
        figsize=config.figsize,
        dpi=config.dpi,
        subtract_inputs=config.subtract_inputs,
        compare_to_target=config.compare_to_target,
    )

    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)

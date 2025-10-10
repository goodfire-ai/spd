#!/usr/bin/env python3
"""
Template plotting script for SPD models.

This is a template script that follows the same structure as other scripts in spd/scripts.
Replace the placeholder functionality with your actual plotting needs.

Usage:
    python spd/scripts/plotting_template/plotting_template.py spd/scripts/plotting_template/plotting_template_config.yaml
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field

from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class PlottingTemplateConfig(BaseModel):
    """Configuration for plotting template script."""

    model_path: str = Field(
        ..., description="Path to the trained SPD model (wandb:project/run_id or local path)"
    )

    # Add your plotting parameters here
    figsize: tuple[float, float] = Field(default=(8, 6), description="Figure size (width, height)")
    dpi: int = Field(default=150, description="DPI for the figures")
    device: str = Field(default="auto", description="Device to use (default: auto)")

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def load_model(config: PlottingTemplateConfig) -> ComponentModel:
    """Load SPD model from path.

    Args:
        config: Configuration object containing model path

    Returns:
        Loaded ComponentModel
    """
    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Loading model from: {config.model_path}")

    try:
        run_info = SPDRunInfo.from_path(config.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        logger.info("Successfully loaded model")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def create_plot(_model: ComponentModel, output_path: Path, config: PlottingTemplateConfig) -> None:
    """Create your plot here.

    Args:
        _model: Loaded ComponentModel (unused in template)
        output_path: Path to save the plot
        config: Configuration object
    """
    logger.info("Creating plot...")

    # TODO: Replace this with your actual plotting code
    _, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Example: Create a simple plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Template Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Plot saved to: {output_path}")


def main(config_path_or_obj: str | PlottingTemplateConfig) -> None:
    """Main function for plotting template."""
    config = load_config(config_path_or_obj, config_model=PlottingTemplateConfig)

    # Set device
    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Using device: {device}")

    # Extract model ID from the model path for directory organization
    if config.model_path.startswith("wandb:"):
        # Extract run ID from wandb path (e.g., "wandb:goodfire/spd/fi1phj8l" -> "fi1phj8l")
        model_id = config.model_path.split("/")[-1]
    else:
        # For local paths, use the directory name
        model_id = Path(config.model_path).name

    # Set up output directory with model ID subdirectory
    if config.output_dir is None:
        base_output_dir = Path(__file__).parent / "out"
    else:
        base_output_dir = Path(config.output_dir)

    output_dir_path = base_output_dir / model_id
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir_path}")

    # Load the model
    model = load_model(config)

    # Create your plot
    output_path = output_dir_path / "template_plot.png"
    create_plot(model, output_path, config)

    logger.info("Plotting complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to the config file in the same directory if no argument provided
        config_path = Path(__file__).parent / "plotting_template_config.yaml"

    main(str(config_path))

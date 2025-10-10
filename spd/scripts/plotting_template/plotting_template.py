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
    assert config.model_path, "Model path cannot be empty"

    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Loading model from: {config.model_path}")

    try:
        run_info = SPDRunInfo.from_path(config.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()

        assert model is not None, "Model failed to load"
        assert next(model.parameters()).device.type == device.split(":")[0], (
            f"Model not on expected device {device}"
        )

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
    assert output_path.parent.exists(), f"Output directory {output_path.parent} does not exist"
    assert config.figsize[0] > 0 and config.figsize[1] > 0, "Figure size must be positive"
    assert config.dpi > 0, "DPI must be positive"

    logger.info("Creating plot...")

    _, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    assert x.shape == (100,), f"Expected x shape (100,), got {x.shape}"
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"

    ax.plot(x, y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Template Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    assert output_path.exists(), f"Plot file was not created at {output_path}"

    logger.info(f"Plot saved to: {output_path}")


def main(config_path_or_obj: str | PlottingTemplateConfig) -> None:
    """Main function for plotting template."""
    config = load_config(config_path_or_obj, config_model=PlottingTemplateConfig)
    assert config is not None, "Failed to load configuration"

    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Using device: {device}")

    if config.model_path.startswith("wandb:"):
        # wandb paths should have format "wandb:project/run_id"
        path_parts = config.model_path.split("/")
        assert len(path_parts) >= 3, f"Invalid wandb path format: {config.model_path}"
        model_id = path_parts[-1]
    else:
        model_id = Path(config.model_path).name

    assert model_id, "Could not extract model ID from path"

    if config.output_dir is None:
        base_output_dir = Path(__file__).parent / "out"
    else:
        base_output_dir = Path(config.output_dir)

    output_dir_path = base_output_dir / model_id
    output_dir_path.mkdir(parents=True, exist_ok=True)
    assert output_dir_path.exists(), f"Failed to create output directory: {output_dir_path}"
    logger.info(f"Output directory: {output_dir_path}")

    model = load_model(config)
    output_path = output_dir_path / "template_plot.png"
    create_plot(model, output_path, config)

    logger.info("Plotting complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = Path(__file__).parent / "plotting_template_config.yaml"

    assert Path(config_path).exists(), f"Config file not found: {config_path}"
    main(str(config_path))

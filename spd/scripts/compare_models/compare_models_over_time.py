"""Model comparison script for geometric similarity analysis over time (checkpoints).

This script compares a sequence of SPD model checkpoints from a current wandb run
against a sequence of checkpoints from a reference wandb run, matching them by step number.

Usage:
    python spd/scripts/compare_models/compare_models_over_time.py spd/scripts/compare_models/compare_models_config.yaml
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from wandb.apis.public import Run

from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.scripts.compare_models.compare_models import (
    CompareModelsConfig,
    ModelComparator,
)
from spd.utils.distributed_utils import get_device
from spd.utils.run_utils import save_file
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_wandb_run_dir,
)


class ModelComparatorOverTime(ModelComparator):
    """Compare a sequence of SPD model checkpoints against a reference sequence."""

    def __init__(self, config: CompareModelsConfig):  # pyright: ignore[reportMissingSuperCall]
        """Initialize the model comparator.

        Args:
            config: CompareModelsConfig instance containing all configuration parameters
        """
        # We do NOT call super().__init__ because we want to handle model loading dynamically
        self.config = config
        self.density_threshold = config.density_threshold
        self.device = get_device()

        # 1. Validate paths
        assert config.current_model_path.startswith("wandb:"), (
            "current_model_path must be a wandb path (wandb:entity/project/run_id)"
        )
        assert config.reference_model_path.startswith("wandb:"), (
            "reference_model_path must be a wandb path (wandb:entity/project/run_id)"
        )

        # 2. Setup runs
        api = wandb.Api()

        self.current_wandb_path = config.current_model_path.removeprefix("wandb:")
        self.current_run: Run = api.run(self.current_wandb_path)
        self.current_run_dir = fetch_wandb_run_dir(self.current_run.id)

        self.reference_wandb_path = config.reference_model_path.removeprefix("wandb:")
        self.reference_run: Run = api.run(self.reference_wandb_path)
        self.reference_run_dir = fetch_wandb_run_dir(self.reference_run.id)

        # 3. Load configs (assumed static per run)
        logger.info(f"Loading config for current run: {self.current_wandb_path}")
        current_config_path = download_wandb_file(
            self.current_run, self.current_run_dir, "final_config.yaml"
        )
        with open(current_config_path) as f:
            self.current_config = Config(**yaml.safe_load(f))

        logger.info(f"Loading config for reference run: {self.reference_wandb_path}")
        ref_config_path = download_wandb_file(
            self.reference_run, self.reference_run_dir, "final_config.yaml"
        )
        with open(ref_config_path) as f:
            self.reference_config = Config(**yaml.safe_load(f))

        # Placeholder attributes that will be set during iteration
        self.current_model: ComponentModel | None = None
        self.reference_model: ComponentModel | None = None

    def _get_checkpoints(self, run: Run) -> dict[int, str]:
        """Find all model checkpoints in a run.

        Returns:
            Dictionary mapping step number to filename.
        """
        checkpoint_files = [
            f.name
            for f in run.files()
            if f.name.startswith("model_") and f.name.endswith((".pt", ".pth"))
        ]

        checkpoints = {}
        for filename in checkpoint_files:
            try:
                # Try to extract step number
                # Expected formats: model_step_<step>.pth or model_<step>.pth
                stem = Path(filename).stem  # removes extension
                parts = stem.split("_")
                step = int(parts[-1])
                checkpoints[step] = filename
            except ValueError:
                logger.warning(
                    f"Could not parse step from checkpoint filename: {filename} in run {run.id}, skipping"
                )
                continue

        return checkpoints

    def _load_checkpoint(
        self, run: Run, run_dir: Path, config: Config, filename: str
    ) -> ComponentModel:
        """Download and load a specific checkpoint."""
        checkpoint_path = download_wandb_file(run, run_dir, filename)

        run_info = SPDRunInfo(checkpoint_path=checkpoint_path, config=config)

        model = ComponentModel.from_run_info(run_info)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model

    def run_time_comparison(self, eval_iterator: Iterator[Any]) -> dict[int, dict[str, float]]:
        """Run comparison for all common checkpoints."""
        current_checkpoints = self._get_checkpoints(self.current_run)
        reference_checkpoints = self._get_checkpoints(self.reference_run)

        # Find common steps
        common_steps = sorted(set(current_checkpoints.keys()) & set(reference_checkpoints.keys()))

        logger.info(
            f"Found {len(current_checkpoints)} current checkpoints and {len(reference_checkpoints)} reference checkpoints"
        )
        logger.info(f"Proceeding with {len(common_steps)} common steps: {common_steps}")

        results = {}

        for step in common_steps:
            curr_filename = current_checkpoints[step]
            ref_filename = reference_checkpoints[step]

            logger.info(f"Processing step {step}: Current={curr_filename}, Ref={ref_filename}")

            # Load models
            self.current_model = self._load_checkpoint(
                self.current_run, self.current_run_dir, self.current_config, curr_filename
            )
            self.reference_model = self._load_checkpoint(
                self.reference_run, self.reference_run_dir, self.reference_config, ref_filename
            )

            # Run comparison
            step_results = self.run_comparison(eval_iterator)
            results[step] = step_results

            # Cleanup
            del self.current_model
            del self.reference_model
            self.current_model = None
            self.reference_model = None
            torch.cuda.empty_cache()

        return results


def plot_results(results: dict[int, dict[str, float]], output_dir: Path) -> None:
    """Plot similarity metrics over time.

    Args:
        results: Dictionary mapping step number to metrics dictionary.
        output_dir: Directory to save the plot.
    """
    steps = sorted(results.keys())
    if not steps:
        logger.warning("No results to plot.")
        return

    # Get all metric keys from the first step (assuming consistent keys)
    first_step_metrics = results[steps[0]]
    metric_keys = [k for k in first_step_metrics if k.startswith("mean_max_abs_cosine_sim")]

    if not metric_keys:
        logger.warning("No mean_max_abs_cosine_sim metrics found.")
        return

    plt.figure(figsize=(12, 8))

    for metric in metric_keys:
        values = [results[step].get(metric, float("nan")) for step in steps]
        # simplify label: remove prefix
        label = metric.replace("mean_max_abs_cosine_sim/", "")
        # Highlight all_layers
        if label == "all_layers":
            plt.plot(steps, values, label=label, linewidth=3, color="black", linestyle="--")
        else:
            plt.plot(steps, values, label=label, alpha=0.7)

    plt.xlabel("Step")
    plt.ylabel("Mean Max Abs Cosine Similarity")
    plt.title("Model Similarity Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "similarity_over_time.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved similarity plot to {plot_path}")


def main(config_path: Path | str) -> None:
    """Main execution function."""
    config = CompareModelsConfig.from_file(config_path)

    if config.output_dir is None:
        output_dir = Path(__file__).parent / "out"
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "similarity_results_over_time.json"

    # --- Start Computation ---
    # Comment out this block to skip computation and only plot existing results
    comparator = ModelComparatorOverTime(config)

    logger.info("Setting up evaluation data...")
    eval_iterator = comparator.create_eval_data_loader()

    logger.info("Starting model comparison over time...")
    similarities = comparator.run_time_comparison(eval_iterator)

    save_file(similarities, results_file)
    logger.info(f"Comparison complete! Results saved to {results_file}")
    # --- End Computation ---

    # Load results from file (to ensure we can plot even if computation was skipped)
    if results_file.exists():
        with open(results_file) as f:
            raw_results = json.load(f)
        # Convert keys back to int (JSON keys are strings)
        similarities = {int(k): v for k, v in raw_results.items()}
    else:
        logger.error(f"Results file not found: {results_file}")
        return

    plot_results(similarities, output_dir)


if __name__ == "__main__":
    fire.Fire(main)

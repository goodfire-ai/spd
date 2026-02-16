"""Plot mean CI per component from harvested data.

For each module, produces a scatter plot of mean CI values sorted descending by component.
Two figures are generated: one with a linear y-scale and one with a log y-scale.
Modules are arranged in a grid (max 6 rows, filling column by column), matching the
layout used by the training-time eval figures in spd/plotting.py.

Usage:
    python -m spd.scripts.plot_mean_ci.plot_mean_ci wandb:goodfire/spd/runs/<run_id>
"""

from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MAX_ROWS = 6


def _sorted_mean_cis_by_module(
    summary: dict[str, ComponentSummary],
) -> dict[str, list[float]]:
    """Group mean_ci values by module, sorted descending within each module."""
    by_module = defaultdict[str, list[float]](list)
    for s in summary.values():
        by_module[s.layer].append(s.mean_ci)
    return {k: sorted(v, reverse=True) for k, v in sorted(by_module.items())}


def _plot_grid(
    mean_cis_by_module: dict[str, list[float]],
    log_y: bool,
    out_dir: Path,
) -> None:
    n_modules = len(mean_cis_by_module)
    n_cols = (n_modules + MAX_ROWS - 1) // MAX_ROWS
    n_rows = min(n_modules, MAX_ROWS)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False, dpi=200
    )

    for i in range(n_modules, n_rows * n_cols):
        axs[i % n_rows, i // n_rows].set_visible(False)

    for i, (module_name, cis) in enumerate(mean_cis_by_module.items()):
        row = i % n_rows
        col = i // n_rows
        ax = axs[row, col]

        if log_y:
            ax.set_yscale("log")

        ax.scatter(range(len(cis)), cis, marker="x", s=10)

        if row == n_rows - 1 or i == n_modules - 1:
            ax.set_xlabel("Component")
        ax.set_ylabel("mean CI")
        ax.set_title(module_name, fontsize=10)

    fig.tight_layout()

    scale = "log" if log_y else "linear"
    path = out_dir / f"mean_ci_{scale}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_mean_ci(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = HarvestRepo.open(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    mean_cis_by_module = _sorted_mean_cis_by_module(summary)
    logger.info(f"Modules: {len(mean_cis_by_module)}")

    _plot_grid(mean_cis_by_module, log_y=False, out_dir=out_dir)
    _plot_grid(mean_cis_by_module, log_y=True, out_dir=out_dir)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_mean_ci)

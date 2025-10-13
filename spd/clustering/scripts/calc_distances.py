import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from spd.clustering.consts import DistancesArray, DistancesMethod
from spd.clustering.math.merge_distances import compute_distances
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.log import logger
from spd.settings import SPD_CACHE_DIR


def main(ensemble_id: str, distances_method: DistancesMethod, base_output_dir: Path) -> None:
    """Calculate distances between clustering runs in an ensemble."""
    runs_dir = base_output_dir / "runs"
    run_dirs = [i for i in runs_dir.iterdir() if i.stem.startswith(str(ensemble_id))]

    histories: list[MergeHistory] = [MergeHistory.read(i / "history.npz") for i in run_dirs]

    ensemble_dir = base_output_dir / "ensembles" / ensemble_id
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    ensemble: MergeHistoryEnsemble = MergeHistoryEnsemble(data=histories)

    merge_array, merge_meta = ensemble.normalized()

    metadata_path = ensemble_dir / "ensemble_meta.json"
    metadata_path.write_text(json.dumps(merge_meta, indent=2))

    array_path = ensemble_dir / "ensemble_merge_array.npz"
    np.savez_compressed(array_path, merge_array=merge_array)

    logger.info(f"Computing distances using method: {distances_method}")
    distances: DistancesArray = compute_distances(
        normalized_merge_array=merge_array,
        method=distances_method,
    )

    ensemble_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ensemble_dir / f"distances_{distances_method}.npz", distances=distances)
    logger.info(f"Distances computed and saved: shape={distances.shape}")

    # Create the distances distribution plot
    ax: Axes = plot_dists_distribution(
        distances=distances, mode="points", label=f"{distances_method} distances"
    )
    plt.title(f"Distance Distribution ({distances_method})")

    # Only add legend if there are labeled artists
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend()

    fig_path = ensemble_dir / f"distances_{distances_method}.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved distances distribution plot to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distances between clustering runs")
    parser.add_argument("-e", "--ensemble-id", type=str, required=True)
    parser.add_argument(
        "-m",
        "--distances-method",
        choices=["perm_invariant_hamming", "jaccard"],
        default="perm_invariant_hamming",
    )
    parser.add_argument("--base_output_dir", type=Path, default=SPD_CACHE_DIR / "clustering")
    args = parser.parse_args()
    main(
        ensemble_id=args.ensemble_id,
        distances_method=args.distances_method,
        base_output_dir=args.base_output_dir,
    )

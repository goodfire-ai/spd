from pathlib import Path

import numpy as np

from spd.clustering.math.merge_distances import compute_distances
from spd.clustering.merge import DistancesArray, DistancesMethod, MergesArray


def compute_histories_distances(
    merges_path: Path,
    method: DistancesMethod = "perm_invariant_hamming",
) -> tuple[Path, DistancesArray]:
    """Main function to load merge histories and compute distances"""

    # load
    merge_array: MergesArray = np.load(merges_path, allow_pickle=True)["merges"]

    # compute
    distances: DistancesArray = compute_distances(
        normalized_merge_array=merge_array,
        method=method,
    )

    # save
    distances_path: Path = merges_path.with_suffix(f".{method}.distances.npz")
    np.savez_compressed(distances_path, distances=distances)

    print(f"Saved distances to {distances_path}")
    return distances_path, distances


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute distances between merge histories"
    )
    parser.add_argument("merges-path", type=Path, help="Path to the merge histories file")
    parser.add_argument(
        "--method", type=str, default="perm_invariant_hamming", help="Distance method to use"
    )
    args: argparse.Namespace = parser.parse_args()

    compute_histories_distances(
        merges_path=args.merges_path,
        method=args.method,
    )

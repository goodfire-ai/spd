from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from muutils.parallel import run_maybe_parallel

from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

MergesAtIterArray = Int[np.ndarray, "n_ens c_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters c_components"]
DistancesMethod = Literal["perm_invariant_hamming"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]


DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
    "perm_invariant_hamming": perm_invariant_hamming_matrix,
}

# pyright: reportUnnecessaryComparison=false, reportUnreachable=false


def compute_and_save_distances(
    normalized_merge_array: MergesArray,
    output_dir: Path,
    method: DistancesMethod = "perm_invariant_hamming",
) -> DistancesArray:
    distances = compute_distances(normalized_merge_array, method)
    np.savez_compressed(output_dir / f"distances_{method}.npz", distances=distances)
    return distances


def compute_distances(
    normalized_merge_array: MergesArray,
    method: DistancesMethod = "perm_invariant_hamming",
) -> DistancesArray:
    n_iters: int = normalized_merge_array.shape[1]
    match method:
        case "perm_invariant_hamming":
            merges_array_list: list[Int[np.ndarray, "n_ens c_components"]] = [
                normalized_merge_array[:, i, :] for i in range(n_iters)
            ]

            distances_list: list[Float[np.ndarray, "n_ens n_ens"]] = run_maybe_parallel(
                func=perm_invariant_hamming_matrix,
                iterable=merges_array_list,
                parallel=8,
            )

            return np.stack(distances_list, axis=0)
        case _:
            raise ValueError(f"Unknown distance method: {method}")

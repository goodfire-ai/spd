from collections.abc import Callable

import numpy as np
from jaxtyping import Float, Int

from spd.clustering.consts import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
    MergesAtIterArray,
)
from spd.clustering.math.jaccard import jaccard_partition_matrix
from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
    "perm_invariant_hamming": perm_invariant_hamming_matrix,
    "jaccard": jaccard_partition_matrix,
}

# pyright: reportUnnecessaryComparison=false, reportUnreachable=false


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

            # distances_list: list[Float[np.ndarray, "n_ens n_ens"]] = run_maybe_parallel(
            #     func=perm_invariant_hamming_matrix,
            #     iterable=merges_array_list,
            #     parallel=True,
            # )

            return np.stack(distances_list, axis=0)
        case "jaccard":
            merges_array_list: list[Int[np.ndarray, "n_ens c_components"]] = [
                normalized_merge_array[:, i, :] for i in range(n_iters)
            ]
            # distances_list: list[Float[np.ndarray, "n_ens n_ens"]] = run_maybe_parallel(
            #     func=jaccard_partition_matrix,
            #     iterable=merges_array_list,
            #     parallel=True,
            # )
            return np.stack(distances_list, axis=0)
        case _:
            raise ValueError(f"Unknown distance method: {method}")

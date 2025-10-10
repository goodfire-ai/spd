from collections.abc import Callable
<<<<<<< HEAD
=======
from typing import Literal
>>>>>>> chinyemba/feature/clustering-sjcs

import numpy as np
from jaxtyping import Float, Int
from muutils.parallel import run_maybe_parallel

<<<<<<< HEAD
from spd.clustering.consts import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
    MergesAtIterArray,
)

# from spd.clustering.math.jaccard import jaccard_partition_matrix
from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
    "perm_invariant_hamming": perm_invariant_hamming_matrix,
    # "jaccard": jaccard_partition_matrix,
=======
from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

MergesAtIterArray = Int[np.ndarray, "n_ens c_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters c_components"]
DistancesMethod = Literal["perm_invariant_hamming"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]


DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
    "perm_invariant_hamming": perm_invariant_hamming_matrix,
>>>>>>> chinyemba/feature/clustering-sjcs
}

# pyright: reportUnnecessaryComparison=false, reportUnreachable=false


def compute_distances(
    normalized_merge_array: MergesArray,
    method: DistancesMethod = "perm_invariant_hamming",
) -> DistancesArray:
    n_iters: int = normalized_merge_array.shape[1]
<<<<<<< HEAD
    merges_array_list: list[Int[np.ndarray, "n_ens n_components"]]
    distances_list: list[Float[np.ndarray, "n_ens n_ens"]]
    match method:
        case "perm_invariant_hamming":
            merges_array_list = [normalized_merge_array[:, i, :] for i in range(n_iters)]

            distances_list = run_maybe_parallel(
                func=perm_invariant_hamming_matrix,
                iterable=merges_array_list,
                parallel=True,
            )

            return np.stack(distances_list, axis=0)
        case "jaccard":
            raise NotImplementedError("Jaccard distance computation is not implemented.")
            # merges_array_list = [normalized_merge_array[:, i, :] for i in range(n_iters)]
            # distances_list = run_maybe_parallel(
            #     func=jaccard_partition_matrix,
            #     iterable=merges_array_list,
            #     parallel=True,
            # )
            # return np.stack(distances_list, axis=0)
=======
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
>>>>>>> chinyemba/feature/clustering-sjcs
        case _:
            raise ValueError(f"Unknown distance method: {method}")

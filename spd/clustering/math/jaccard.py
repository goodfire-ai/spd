import numpy as np
from jaxtyping import Float, Int


def _sum_combinations2(counts: Int[np.ndarray, " k"]) -> int:
    """sum C(c_i, 2)."""
    counts64: Int[np.ndarray, " k"] = counts.astype(np.int64, copy=False)
    return int(np.sum(counts64 * (counts64 - 1) // 2))


def _relabeled_partition(
    labels: Int[np.ndarray, " n"],
) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " k"]]:
    """Relabel labels to 0..K-1 and return (inverse_labels, cluster_sizes)."""
    _: Int[np.ndarray, " k"]
    inv: Int[np.ndarray, " n"]
    _, inv = np.unique(labels, return_inverse=True)
    sizes: Int[np.ndarray, " k"] = np.bincount(inv, minlength=int(inv.max()) + 1).astype(np.int64)
    return inv.astype(np.int64, copy=False), sizes


def _pairwise_m11(
    inv_a: Int[np.ndarray, " n"],
    inv_b: Int[np.ndarray, " n"],
    k_b: int,
) -> int:
    """M11 = number of pairs clustered together in both partitions."""
    combo: Int[np.ndarray, " n"] = inv_a * np.int64(k_b) + inv_b
    counts: Int[np.ndarray, " kb"] = np.bincount(combo).astype(np.int64)
    return _sum_combinations2(counts)


def jaccard_partition_matrix(
    X: Int[np.ndarray, "k n"],
) -> Float[np.ndarray, "k k"]:
    """Compute all pairwise Jaccard indices between partitions.

    Strictly lower-triangular entries are filled with Jaccard values;
    diagonal and upper triangle are `np.nan`.

    # Parameters:
     - `X : Int[np.ndarray, "k n"]`
       Matrix where each of the `k` rows is a label vector of length `n`.

    # Returns:
     - `Float[np.ndarray, "k k"]`
       Similarity matrix `J` with `J[i, j]` defined only for `i > j`;
       all other positions are `np.nan`.
    """
    k_rows: int
    _n_len: int
    k_rows, _n_len = X.shape
    J: Float[np.ndarray, "k k"] = np.full((k_rows, k_rows), np.nan, dtype=float)

    # Precompute relabeled partitions + their sum of combs
    invs: list[Int[np.ndarray, " n"]] = []
    sizes_list: list[Int[np.ndarray, " ki"]] = []
    sum_combs: Int[np.ndarray, " k"] = np.zeros((k_rows,), dtype=np.int64)
    inv_i: Int[np.ndarray, " n"]
    for i in range(k_rows):
        sizes_i: Int[np.ndarray, " ki"]
        inv_i, sizes_i = _relabeled_partition(X[i])
        invs.append(inv_i)
        sizes_list.append(sizes_i)
        sum_combs[i] = _sum_combinations2(sizes_i)

    for i in range(1, k_rows):
        inv_i = invs[i]
        sum_comb_i: int = int(sum_combs[i])
        for j in range(i):
            inv_j: Int[np.ndarray, " n"] = invs[j]
            sum_comb_j: int = int(sum_combs[j])
            m11: int = _pairwise_m11(inv_i, inv_j, k_b=int(sizes_list[j].shape[0]))
            denom: int = sum_comb_i + sum_comb_j - m11
            J[i, j] = 1.0 if denom == 0 else m11 / denom
    return J

<<<<<<< HEAD
import warnings
=======
from typing import Literal, overload
>>>>>>> chinyemba/feature/clustering-sjcs

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment


<<<<<<< HEAD
def perm_invariant_hamming_matrix(
    X: Int[np.ndarray, "n_ens n_components"],
) -> Float[np.ndarray, "n_ens n_ens"]:
=======
@overload
def perm_invariant_hamming(
    a: Int[np.ndarray, " n"],
    b: Int[np.ndarray, " n"],
    return_mapping: Literal[False],
) -> tuple[int, None]: ...
@overload
def perm_invariant_hamming(
    a: Int[np.ndarray, " n"],
    b: Int[np.ndarray, " n"],
    return_mapping: Literal[True],
) -> tuple[int, dict[int, int]]: ...
def perm_invariant_hamming(
    a: Int[np.ndarray, " n"],
    b: Int[np.ndarray, " n"],
    return_mapping: bool = True,
) -> tuple[int, dict[int, int] | None]:
    """Compute the minimum Hamming distance between two labelings, up to an
    optimal relabeling (permutation) of the groups.

    Args:
        a: First 1-D array of length *n* whose values are group indices
           in the range ``0 .. k-1``.
        b: Second 1-D array of the same shape as ``a``.  The two arrays may
           use different numerical labels for the same groups.

    Returns:
        A tuple containing:

        * The minimal Hamming distance ``d`` (``0 <= d <= n``) after the best
          relabeling of ``a``.
        * A dict mapping each original label in ``a`` to the label it is
          mapped to in ``b`` under that optimal permutation.
    """

    assert a.shape == b.shape, "Label arrays must have the same shape."
    assert a.ndim == 1 and a.size > 0, "Inputs must be 1-D non-empty arrays."

    n: int = int(a.size)
    k: int = int(max(a.max(), b.max()) + 1)

    # Contingency matrix C[p, q] = count of positions where a==p and b==q
    C: Int[np.ndarray, "k k"] = np.zeros((k, k), dtype=int)
    np.add.at(C, (a, b), 1)

    # Hungarian to maximise matches (i.e. minimise negative counts)
    row_ind, col_ind = linear_sum_assignment(-C)
    matches: int = int(C[row_ind, col_ind].sum())

    distance: int = n - matches
    if return_mapping:
        perm: dict[int, int] = {int(p): int(q) for p, q in zip(row_ind, col_ind, strict=False)}
        return distance, perm
    else:
        return distance, None


def perm_invariant_hamming_matrix(
    X: Int[np.ndarray, " k n"],
) -> Float[np.ndarray, " k k"]:
>>>>>>> chinyemba/feature/clustering-sjcs
    """Compute all pairwise permutation-invariant Hamming distances.

    The strictly lower-triangular entries are filled with distances;
    the diagonal and upper triangle are left as `np.nan`.

    # Parameters:
<<<<<<< HEAD
     - `X : Int[np.ndarray, "n_ens n_components"]`
       Matrix where each of the `n_ens` rows is a label vector of length `n_components`.

    # Returns:
     - `Float[np.ndarray, "n_ens n_ens"]`
=======
     - `X : Int[np.ndarray, " k n"]`
       Matrix where each of the `k` rows is a label vector of length `n`.

    # Returns:
     - `Float[np.ndarray, " k k"]`
>>>>>>> chinyemba/feature/clustering-sjcs
       Distance matrix `D` with `D[i, j]` defined only for `i > j`;
       all other positions are `np.nan`.

    # Usage:
    ```python
    >>> X = np.array([[0, 0, 1],
    ...               [1, 1, 0],
    ...               [0, 1, 0]])
    >>> D = perm_invariant_hamming_matrix(X)
    >>> D
    array([[nan, nan, nan],
           [ 0., nan, nan],
           [ 2., 2., nan]])
    ```
    """
<<<<<<< HEAD
    n_ens: int
    n_components: int
    n_ens, n_components = X.shape
    D: Float[np.ndarray, "n_ens n_ens"] = np.full((n_ens, n_ens), np.nan, dtype=float)

    # Pre-compute max label in each row once.
    row_max: Int[np.ndarray, " n_ens"] = X.max(axis=1)

    for i in range(1, n_ens):
        a: Int[np.ndarray, " n_components"] = X[i]
        for j in range(i):
            b: Int[np.ndarray, " n_components"] = X[j]

            k_lbls: int = int(max(row_max[i], row_max[j]) + 1)

            # Handle case where all labels are -1 (no valid clustering)
            if k_lbls <= 0:
                warnings.warn(
                    f"All labels are -1 at rows {i} and {j}. Setting distance to 0.",
                    UserWarning,
                    stacklevel=2,
                )
                D[i, j] = 0.0
                continue

=======
    k_rows: int
    n_len: int
    k_rows, n_len = X.shape
    D: Float[np.ndarray, "k k"] = np.full((k_rows, k_rows), np.nan, dtype=float)

    # Pre-compute max label in each row once.
    row_max: Int[np.ndarray, " k"] = X.max(axis=1)

    for i in range(1, k_rows):
        a: Int[np.ndarray, " n"] = X[i]
        for j in range(i):
            b: Int[np.ndarray, " n"] = X[j]

            k_lbls: int = int(max(row_max[i], row_max[j]) + 1)

>>>>>>> chinyemba/feature/clustering-sjcs
            C: Int[np.ndarray, "k_lbls k_lbls"] = np.zeros((k_lbls, k_lbls), dtype=int)
            np.add.at(C, (a, b), 1)

            row_ind, col_ind = linear_sum_assignment(-C)
            matches: int = int(C[row_ind, col_ind].sum())

<<<<<<< HEAD
            D[i, j] = n_components - matches  # int is fine; array is float because of NaN
=======
            D[i, j] = n_len - matches  # int is fine; array is float because of NaN
>>>>>>> chinyemba/feature/clustering-sjcs

    return D

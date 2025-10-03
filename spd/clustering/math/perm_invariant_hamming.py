import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment


def perm_invariant_hamming_matrix(
    X: Int[np.ndarray, " k n"],
) -> Float[np.ndarray, " k k"]:
    """Compute all pairwise permutation-invariant Hamming distances.

    The strictly lower-triangular entries are filled with distances;
    the diagonal and upper triangle are left as `np.nan`.

    # Parameters:
     - `X : Int[np.ndarray, " k n"]`
       Matrix where each of the `k` rows is a label vector of length `n`.

    # Returns:
     - `Float[np.ndarray, " k k"]`
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

            C: Int[np.ndarray, "k_lbls k_lbls"] = np.zeros((k_lbls, k_lbls), dtype=int)
            np.add.at(C, (a, b), 1)

            row_ind, col_ind = linear_sum_assignment(-C)
            matches: int = int(C[row_ind, col_ind].sum())

            D[i, j] = n_len - matches  # int is fine; array is float because of NaN

    return D

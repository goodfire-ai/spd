from typing import Literal
import numpy as np
from jaxtyping import Float

MatrixSortMetric = Literal["cosine", "dot"]


def sort_by_similarity(
    arr: Float[np.ndarray, "m n"],
    axis: int = 0,
    metric: MatrixSortMetric = "cosine",
) -> Float[np.ndarray, "m n"]:
    """Sort a 2D array by similarity of rows or columns using a greedy heuristic,
    with a secondary tie-breaker by absolute sum magnitude.
    
    Starts from the first row (or column) and iteratively adds the most similar
    remaining vector until all are ordered. Uses cosine or dot-product similarity.
    When multiple candidates have equal similarity to the last selected vector,
    the one with the largest absolute sum is chosen.

    # Parameters:
     - `arr : Float[np.ndarray, "m n"]`
        Input 2D array to sort.
     - `axis : int`
        0 to sort rows, 1 to sort columns.
        (defaults to 0)
     - `metric : str`
        'cosine' or 'dot' similarity.
        (defaults to 'cosine')
     - `seed : int | None`
        Random seed for reproducibility.
        (defaults to None)

    # Returns:
     - `Float[np.ndarray, "m n"]`
        Array reordered by similarity, tie-broken by abs-sum.

    # Usage:
    ```python
    >>> a = np.random.rand(5, 4)
    >>> sort_by_similarity(a, axis=0)
    >>> sort_by_similarity(a, axis=1)
    ```
    """
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D, got shape {arr.shape}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if metric not in MatrixSortMetric.__args__:
        raise ValueError(f"metric must be in {MatrixSortMetric.__args__ = }")

    data: Float[np.ndarray, "n d"] = arr if axis == 0 else arr.T
    n: int = data.shape[0]

    # normalize for cosine similarity
    if metric == "cosine":
        norms: Float[np.ndarray, "n 1"] = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        data = data / norms

    # similarity matrix (dot product or cosine)
    sim: Float[np.ndarray, "n n"] = data @ data.T
    abs_sums: Float[np.ndarray, "n"] = np.sum(np.abs(data), axis=1)

    # greedy ordering
    remaining: list[int] = list(range(n))
    order: list[int] = [remaining.pop(0)]  # start from first row/col

    while remaining:
        last: int = order[-1]

        # choose next by (similarity, abs_sum)
        next_idx: int = max(
            remaining,
            key=lambda i: (sim[last, i], abs_sums[i]),
        )

        order.append(next_idx)
        remaining.remove(next_idx)

    sorted_arr: Float[np.ndarray, "m n"] = arr[order, :] if axis == 0 else arr[:, order]
    return sorted_arr
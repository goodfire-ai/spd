"""jaccard index between clusterings


we start with a matrix X: Int[np.ndarray, "k n"] where each of the k rows is a label vector of length n
we want to compute a Float["k k"] matrix $J$ of pairwise jaccard indices between the rows of X

jaccard index between two partitions A and B is defined as:
J(A, B) = M11 / (M11 + M10 + M01)

where:
- M11 = number of pairs clustered together in both partitions
- M10 = number of pairs clustered together in A but not in B
- M01 = number of pairs clustered together in B but not in A

"""

# %%
import torch
from torch import Tensor
from jaxtyping import Bool, Int

from muutils.dbg import dbg

# def per_row_label_counts(X: Int[Tensor, "k n"]) -> list[Tensor]:
#     """Return a list of 1D count arrays, one per row."""
#     return [
#         torch.bincount(x)
#         for x in X
#     ]


def relabel_singletons(
    x: Int[Tensor, " n"],
) -> Int[Tensor, " n"]:
    """relabel anything in a singleton cluster to -1, relabel other clusters to minimize labels"""
    assert (x >= 0).all(), "input labels must be non-negative"
    # figure out where the singletons are
    counts: Int[Tensor, " k"] = torch.bincount(x)
    singleton_mask: Bool[Tensor, " k"] = counts == 1

    
    x_relabel: Int[Tensor, " n"] = x.clone()
    dbg(x)
    dbg(singleton_mask)
    dbg(singleton_mask[x])
    dbg(x_relabel)
    dbg(x_relabel[singleton_mask[x]])

    # map singletons to -1
    x_relabel[singleton_mask[x]] = -1
    dbg(x_relabel)

    # map every non `-1` label to a new label
    non_singleton_labels: Int[Tensor, " m"] = x_relabel[~singleton_mask[x]].unique()
    dbg(non_singleton_labels)
    n_unique_nonsingleton_labels: int = non_singleton_labels.shape[0]
    dbg(n_unique_nonsingleton_labels)
    old_to_new: dict[int, int] = {
        old: new
        for new, old in enumerate(sorted(non_singleton_labels.tolist()))
    }
    dbg(old_to_new)

    for old, new in old_to_new.items():
        x_relabel[x == old] = new
    dbg(x_relabel)


X = torch.tensor(
    [0, 3, 3, 2, 4, 0, 5, 6, 7, 7, 7],
)
dbg(X)

relabel_singletons(X)


# dbg(X - z[0])

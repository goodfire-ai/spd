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
from jaxtyping import Bool, Int, Float
from muutils.dbg import dbg
from torch import Tensor

# def per_row_label_counts(X: Int[Tensor, "k n"]) -> list[Tensor]:
#     """Return a list of 1D count arrays, one per row."""
#     return [
#         torch.bincount(x)
#         for x in X
#     ]


def process_singletons(
    x: Int[Tensor, " n"],
) -> tuple[Int[Tensor, " n"], int]:
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
        old: new for new, old in enumerate(sorted(non_singleton_labels.tolist()))
    }
    dbg(old_to_new)

    for old, new in old_to_new.items():
        x_relabel[x == old] = new
    dbg(x_relabel)

    return x_relabel, n_unique_nonsingleton_labels


# X_1 = torch.tensor([0, 3, 3, 2, 4, 0, 5, 6, 7, 7, 7])
# X_2 = torch.tensor([1, 1, 2, 3, 3, 1, 4, 5, 6, 6, 6])
# dbg(X_1)
# process_singletons(X_1)


# def to_matrix(
#     self, device: torch.device | None = None
# ) -> Bool[Tensor, "k_groups n_components"]:
#     if device is None:
#         device = self.group_idxs.device
#     mat: Bool[Tensor, "k_groups n_components"] = torch.zeros(
#         (self.k_groups, self._n_components), dtype=torch.bool, device=device
#     )
#     idxs: Int[Tensor, " n_components"] = torch.arange(
#         self._n_components, device=device, dtype=torch.int
#     )
#     mat[self.group_idxs.to(dtype=torch.int), idxs] = True
#     return mat

def expand_to_onehot(
    x: Int[Tensor, " n"],
    k_groups: int,
) -> Bool[Tensor, " k_groups+1 n_components"]:
    """expand a label (possibly having -1s) vector to a one-hot matrix"""
    n_components: int = x.shape[0]

    # add 1 as -1 will map to last index and be ignored
    mat: Bool[Tensor, " k_groups n_components"] = torch.zeros(
        (k_groups+1, n_components), dtype=torch.bool
    )
    idxs: Int[Tensor, " n_components"] = torch.arange(
        n_components, dtype=torch.int
    )
    mat[x.to(dtype=torch.int), idxs] = True
    return mat

import matplotlib.pyplot as plt

# plt.imshow(expand_to_onehot(*process_singletons(X_1)))
# plt.show()
# plt.imshow(expand_to_onehot(*process_singletons(X_2)))
# plt.show()


def jaccard_index(
    X: Int[Tensor, " s n"],
) -> Float[Tensor, " s s"]:
    """compute pairwise jaccard indices between rows of X"""
    s: int
    n: int
    s, n = X.shape

    X_expanded_list: list[Int[Tensor, " k n"]] = [
        expand_to_onehot(*process_singletons(X[i]))
        for i in range(s)
    ]

    # compute jaccard for each pair of rows
    jaccard: dict[
        tuple[int, int], # key is (i, j) from the rows of X
        Int[Tensor, " k_i k_j"], # value at (p, q) is jaccard index between two clusters
    ] = {}
    for i in range(s):
        for j in range(i, s):
            X_i: Int[Tensor, " k_i n"] = X_expanded_list[i].to(torch.int16)
            X_j: Int[Tensor, " k_j n"] = X_expanded_list[j].to(torch.int16)
            intersects: Int[Tensor, " k_i k_j"] = X_i @ X_j.T
            unions: Int[Tensor, " k_i k_j"] = X_i.sum(dim=1, keepdim=True) + X_j.sum(dim=1, keepdim=True) - intersects
            jaccard_mat: Int[Tensor, " k_i k_j"]  = intersects / unions

            plt.matshow(X_i.cpu())
            plt.title(f"One-hot matrix for row {i} of X\nshape={X_i.shape}")
            plt.show()
            plt.matshow(X_j.cpu())
            plt.title(f"One-hot matrix for row {j} of X\nshape={X_j.shape}")
            plt.show()

            plt.matshow(jaccard_mat.cpu())
            plt.title(f"Gram matrix between row {i} and row {j}\n$[{jaccard_mat.min()}, {jaccard_mat.max()}]$")
            plt.show()


jaccard_index(torch.tensor([
    # [1, 2, 3, 3],
    [0, 1, 1, 2, 3, 3],
    [0, 1, 1, 1, 2, 2],
    # [0, 1, 2, 3],
]))

# dbg(X - z[0])

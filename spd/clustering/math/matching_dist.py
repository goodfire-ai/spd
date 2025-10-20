import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def matching_dist(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    """Compute the pairwise jaccard index between rows of X"""

    s_ensemble, _n_components = X.shape
    matches: Bool[Tensor, "s n n"] = X[:, :, None] == X[:, None, :]

    dists: Float[Tensor, "s s"] = torch.full((s_ensemble, s_ensemble), torch.nan)

    for i in range(s_ensemble):
        for j in range(s_ensemble, i):
            _largest_grp_size_j: int = int(matches[j].sum(dim=1).max().item())
            dist_mat = matches[i].float() - matches[j].float()
            dists[i, j] = torch.tril(dist_mat, diagonal=-1).abs().sum()

    return dists


def matching_dist_np(X: Int[np.ndarray, "s n"]) -> Float[np.ndarray, "s s"]:
    """Compute the pairwise jaccard index between rows of X"""

    return matching_dist(torch.from_numpy(X)).numpy()

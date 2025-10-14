"""jaccard index between clusterings


we start with a matrix X: Int[np.ndarray, "s n"] where each of the s rows is a label vector of length n
we want to compute a Float["s s"] matrix $J$ of pairwise jaccard indices between the rows of X

jaccard index between two partitions A and B is defined as:
J(A, B) = M11 / (M11 + M10 + M01)

where:
- M11 = number of pairs clustered together in both partitions
- M10 = number of pairs clustered together in A but not in B
- M01 = number of pairs clustered together in B but not in A

"""

# %%
import numpy as np

import matplotlib.pyplot as plt
import torch
from jaxtyping import Bool, Float, Int
from muutils.dbg import dbg_auto
from torch import Tensor

from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix


def show_matrix(
        mat: Tensor|np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        vlims: tuple[float, float] | float | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
    """Display a matrix with values annotated on each cell."""
    mat_np: np.ndarray
    mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else mat

    if ax is None:
        _fig, ax = plt.subplots()
        show_colorbar = True
        show_plot = True
    else:
        show_colorbar = False
        show_plot = False

    im = ax.matshow(
        mat_np,
        cmap=cmap,
        **(
            {"vmin": vlims[0], "vmax": vlims[1]} if isinstance(vlims, tuple) else
            {"vmin": -vlims, "vmax": vlims} if isinstance(vlims, float) else {}
        ),
    )

    # Add text annotations
    for i in range(mat_np.shape[0]):
        for j in range(mat_np.shape[1]):
            ax.text(
                j,
                i,
                f"{mat_np[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat_np[i, j] < mat_np.max() / 2 else "black",
            )

    if title:
        ax.set_title(title)
    if show_colorbar:
        plt.colorbar(im, ax=ax)
    if show_plot:
        plt.show()


def expand_to_onehot(
    x: Int[Tensor, " n"],
    k_groups: int,
) -> Bool[Tensor, " k_groups+1 n_components"]:
    """expand a label (possibly having -1s) vector to a one-hot matrix"""
    n_components: int = x.shape[0]

    # add 1 as -1 will map to last index and be ignored
    mat: Bool[Tensor, " k_groups n_components"] = torch.zeros(
        (k_groups + 1, n_components), dtype=torch.bool
    )
    idxs: Int[Tensor, " n_components"] = torch.arange(n_components, dtype=torch.int)
    mat[x.to(dtype=torch.int), idxs] = True
    return mat


def jaccard_index(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    """Compute the pairwise jaccard index between rows of X"""

    s_ensemble, _n_components = X.shape
    dbg_auto(X)
    matches: Bool[Tensor, "s n n"] = X[:, :, None] == X[:, None, :]
    dbg_auto(matches)

    _jaccard: Float[Tensor, "s s"] = torch.full((s_ensemble, s_ensemble), torch.nan)

    # Create single large plot: (s + 2) rows × s cols
    # Row 0: onehot expanded
    # Row 1: matches
    # Rows 2 to s+1: dist_mat for each pair
    fig, axes = plt.subplots(s_ensemble + 2, s_ensemble, figsize=(s_ensemble * 3, (s_ensemble + 2) * 3))

    # Top row: onehot expanded
    for i in range(s_ensemble):
        onehot = expand_to_onehot(X[i], k_groups=int(X[i].max())).cpu().numpy()
        axes[0, i].matshow(onehot, cmap="Blues")
        axes[0, i].set_title(f"onehot {i}")
        axes[0, i].axis("off")

    # Second row: matches
    for i in range(s_ensemble):
        axes[1, i].matshow(matches[i].cpu().numpy())
        axes[1, i].set_title(f"matches {i}")
        axes[1, i].axis("off")

    # Lower s rows: dist_mat for each pair
    for i in range(s_ensemble):
        for j in range(s_ensemble):
            dist_mat = (matches[i].float() - matches[j].float())
            dbg_auto(dist_mat)

            # Plot dist_mat on axes[i + 2, j]
            im = axes[i + 2, j].matshow(dist_mat.cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
            axes[i + 2, j].set_title(f"diff {i}-{j}", fontsize=8)
            axes[i + 2, j].axis("off")

            # Compute distance
            dist: float = torch.tril(dist_mat, diagonal=-1).abs().max(dim=-1).values.sum().item()
            _jaccard[i, j] = dist
            dbg_auto((i, j, dist))

    plt.tight_layout()
    plt.show()

    # for i in range(s_ensemble):
    #     for j in range(i, s_ensemble):
    #         M11: int = int((matches[i] & matches[j]).sum() - n_components) // 2
    #         M10: int = int((matches[i] & ~matches[j]).sum()) // 2
    #         M01: int = int((~matches[i] & matches[j]).sum()) // 2
    #         if M11 + M10 + M01 == 0:
    #             jaccard[i, j] = float("nan")
    #         else:
    #             jaccard[i, j] = M11 / (M11 + M10 + M01)
    #         jaccard[j, i] = jaccard[i, j]
    #         dbg_auto(i, j, M11, M10, M01, jaccard[i, j])

    return _jaccard



# TODO: doesnt work when large groups??????
X_test = torch.tensor(
    [
        # [1, 2, 3, 3],
        # [0, 1, 2, 3],
        # 
        # [0, 1, 1, 2, 3, 3],
        # [3, 0, 0, 1, 2, 2],
        # [0, 3, 1, 1, 2, 2],
        # [0, 3, 0, 0, 1, 1],
        # [0, 0, 1, 0, 0, 1],
        # [0, 0, 0, 0, 0, 1],
        # [2, 3, 0, 0, 1, 1],
        # [2, 3, 0, 0, 1, 2],
        # 
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 3, 2, 1],
        [0, 0, 4, 3, 2, 1],
        [0, 5, 4, 3, 2, 1],

    ]
)



dbg_auto(X_test)

pih_raw = perm_invariant_hamming_matrix(X_test.numpy())
# make pih symmetric, fill diag w/ 0
pih = np.full_like(pih_raw, fill_value=0.0)
pih = pih + np.tril(pih_raw, k=-1) + np.tril(pih_raw, k=-1).T


jcrd = jaccard_index(X_test)


# plt.matshow(pih)
# plt.title("perm-invariant Hamming distance")
# plt.colorbar()
# plt.show()
show_matrix(pih, title="perm-invariant Hamming distance", cmap="viridis")

# plt.matshow(jcrd.numpy())
# plt.title("jaccard index")
# plt.colorbar()
# plt.show()
show_matrix(jcrd, title="jaccard index", cmap="viridis")

diff = jcrd.numpy() - pih
dbg_auto(diff)
vlim = np.max(np.abs(diff))
dbg_auto(vlim)
# plt.matshow(
#     diff,
#     vmin=-vlim,
#     vmax=vlim,
#     cmap="RdBu",
# )
# plt.title("difference")
# plt.colorbar()
# plt.show()

show_matrix(diff, title="difference", cmap="RdBu", vlims=vlim)


ratio = jcrd.numpy() / pih

show_matrix(ratio, title="ratio", cmap="RdBu", vlims=(-3, 5))

# dbg(X - z[0])

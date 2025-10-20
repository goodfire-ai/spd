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
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from muutils.dbg import dbg_auto
from torch import Tensor

from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix


def show_matrix(
    mat: Tensor | np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vlims: tuple[float, float] | float | None = None,
    ax: plt.Axes | None = None,
    num_fmt: str = ".2f",
) -> None:
    """Display a matrix with values annotated on each cell."""
    mat_np: np.ndarray
    mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else mat

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))
        show_colorbar = True
        show_plot = True
    else:
        show_colorbar = False
        show_plot = False

    im = ax.matshow(
        mat_np,
        cmap=cmap,
        **(
            {"vmin": vlims[0], "vmax": vlims[1]}
            if isinstance(vlims, tuple)
            else {"vmin": -vlims, "vmax": vlims}
            if isinstance(vlims, float)
            else {}
        ),
    )

    # Add text annotations
    for i in range(mat_np.shape[0]):
        for j in range(mat_np.shape[1]):
            ax.text(
                j,
                i,
                f"{mat_np[i, j]:{num_fmt}}",
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

    for i in range(s_ensemble):
        for j in range(s_ensemble):
            _largest_grp_size_j: int = int(matches[j].sum(dim=1).max().item())
            dist_mat = matches[i].float() - matches[j].float()
            dist: float = (
                torch.tril(dist_mat, diagonal=-1).abs().sum()
            ).item() 
            _jaccard[i, j] = dist

    return _jaccard


# TODO: doesnt work when large groups??????
# X_test = torch.tensor(
#     [
#         # [1, 2, 3, 3],
#         # [0, 1, 2, 3],
#         #
#         [0, 1, 1, 2, 3, 3],
#         [3, 0, 0, 1, 2, 2],
#         [0, 3, 1, 1, 2, 2],
#         [0, 3, 0, 0, 1, 1],
#         [0, 0, 1, 0, 0, 1],
#         [0, 0, 0, 0, 0, 1],
#         [2, 3, 0, 0, 1, 1],
#         [2, 3, 0, 0, 1, 2],
#         #
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 2, 0, 0],
#         [0, 0, 0, 3, 2, 1],
#         [0, 0, 4, 3, 2, 1],
#         [0, 5, 4, 3, 2, 1],
#     ]
# )


X_test = torch.randint(low=0, high=20, size=(100, 20))

dbg_auto(X_test)

pih_raw = perm_invariant_hamming_matrix(X_test.numpy())
# make pih symmetric, fill diag w/ 0
pih = np.full_like(pih_raw, fill_value=0.0)
pih = pih + np.tril(pih_raw, k=-1) + np.tril(pih_raw, k=-1).T


jcrd = jaccard_index(X_test)


# # plt.matshow(pih)
# # plt.title("perm-invariant Hamming distance")
# # plt.colorbar()
# # plt.show()
# show_matrix(pih, title="perm-invariant Hamming distance", cmap="viridis", num_fmt=".0f")

# # plt.matshow(jcrd.numpy())
# # plt.title("jaccard index")
# # plt.colorbar()
# # plt.show()
# show_matrix(jcrd, title="jaccard index", cmap="viridis", num_fmt=".0f")

# diff = jcrd.numpy() - pih
# dbg_auto(diff)
# vlim = np.max(np.abs(diff))
# dbg_auto(vlim)
# plt.matshow(
#     diff,
#     vmin=-vlim,
#     vmax=vlim,
#     cmap="RdBu",
# )
# plt.title("difference")
# plt.colorbar()
# plt.show()

# show_matrix(diff, title="difference", cmap="RdBu", vlims=vlim, num_fmt=".0f")


# ratio = jcrd.numpy() / pih

# show_matrix(ratio, title="ratio", cmap="RdBu", vlims=(-5, 7))

# Scatter plot showing correlation between the two methods
# Extract upper triangle to avoid duplicates (excluding diagonal)
s = pih.shape[0]
mask = np.triu(np.ones((s, s), dtype=bool), k=1)
pih_pairs = pih[mask]
jcrd_pairs = jcrd.numpy()[mask]

fig, ax = plt.subplots(figsize=(8, 8))
# add some x jitter
pih_pairs_jitter = pih_pairs + np.random.normal(scale=0.15, size=pih_pairs.shape)
ax.plot(pih_pairs_jitter, jcrd_pairs, "o", alpha=0.05, markeredgewidth=0)
ax.set_xlabel("Permutation-Invariant Hamming Distance")
ax.set_ylabel("Jaccard Distance")

# Add diagonal reference line
# max_val = max(pih_pairs.max(), jcrd_pairs.max())
# min_val = min(pih_pairs.min(), jcrd_pairs.min())


# Calculate and display correlation
correlation = np.corrcoef(pih_pairs, jcrd_pairs)[0, 1]
ax.set_title(f"Correlation between Distance Methods\nCorrelation: {correlation:.2f}")


ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# dbg(X - z[0])

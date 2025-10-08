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
import torch
from jaxtyping import Bool, Float, Int
from muutils.dbg import dbg_auto
from torch import Tensor


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
        plt.matshow(matches[i].cpu().numpy())
        plt.title(f"matches for row {i}")
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


jaccard_index(
    torch.tensor(
        [
            # [1, 2, 3, 3],
            [0, 1, 1, 2, 3, 3],
            [3, 0, 0, 1, 2, 2],
            [0, 3, 1, 1, 2, 2],
            [0, 3, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            # [0, 1, 2, 3],
        ]
    )
)

# dbg(X - z[0])

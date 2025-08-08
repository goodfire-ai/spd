# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int, UInt8
from muutils.dbg import dbg_auto
from torch import Tensor


def to_onehot(
    x: Int[Tensor, " n_components"],
) -> Bool[Tensor, "k_groups n_components"]:
    k_groups: int = int(x.max().item() + 1)
    n_components: int = x.shape[0]
    device: torch.device = x.device
    mat: Bool[Tensor, "k_groups n_components"] = torch.zeros(
        (k_groups, n_components), dtype=torch.bool, device=device
    )
    mat[x, torch.arange(n_components, device=device)] = True
    return mat


def to_onehot_pad(
    x: Int[Tensor, " n_components"],
    K: int,
) -> Bool[Tensor, "K n_components"]:
    n_components: int = int(x.shape[0])
    device: torch.device = x.device
    mat: Bool[Tensor, "K n_components"] = torch.zeros(
        (K, n_components), dtype=torch.bool, device=device
    )
    mat[x, torch.arange(n_components, device=device)] = True
    return mat


def pih_dev(
    X: Int[Tensor, " n_ensemble n_components"],
) -> Float[Tensor, " n_ensemble n_ensemble"]:
    n_ensemble: int = X.shape[0]
    n_len: int = X.shape[1]
    max_label: int = int(X.max().item())
    assert max_label < n_len, "Maximum label must be less than the number of elements in each row"

    dbg_auto(X)

    # for each row, compute the counts of each label
    counts: Int[Tensor, "n_ensemble max_label+1"] = torch.stack(
        [torch.bincount(row, minlength=max_label + 1) for row in X], dim=0
    )
    dbg_auto(counts)
    # create a mask for each row, true where the label has a count of 1 (is unique)
    # each pos says if its label is unique in that row
    unique_label_mask_bool: Bool[Tensor, "n_ensemble n_components"] = (
        counts.gather(1, X.to(torch.int64)) == 1
    )
    unique_label_mask_int: UInt8[Tensor, "n_ensemble n_components"] = unique_label_mask_bool.to(
        torch.uint8
    )
    dbg_auto(unique_label_mask_int)

    # compute (for all pairs of rows) the number of times both rows identify elements as being having a unique label
    both_unique: Bool[Tensor, "n_ensemble n_ensemble"] = (
        unique_label_mask_int @ unique_label_mask_int.T
    )
    dbg_auto(both_unique)

    # initialize the output matrix with NaNs
    distances: Float[Tensor, "n_ensemble n_ensemble"] = torch.full(
        (n_ensemble, n_ensemble),
        float("nan"),
        dtype=torch.float32,
    )
    # set lower triangular entries to 0 to start
    distances.tril_()

    distances = both_unique

    # filter those out

    # relable rows to minimize the labels, so that the labels are in the range [0, k_rows - unique_labels - 1]

    # expand each row to a one-hot matrix, compute outer products between all of these

    return distances


data_path = "../../../data/clustering/n8_b4_e04ad4/distances/ensemble_merge_array.npz"
x = torch.tensor(np.load(data_path)["merges"], dtype=torch.int32)
dbg_auto(x)


# c = 10
# for i_e, e in enumerate(x):
#     for i_iter, r in enumerate(e):
#         dbg_auto((i_e, i_iter))
#         dbg_auto(r)
#         r_1h = to_onehot(r)
#         dbg_auto(r_1h.sum(dim=0))
#         plt.matshow(r_1h, cmap="Blues")
#         plt.show()
#         c += 1
#         if c > 20:
#             break

plt.matshow(pih_dev(x[:, 1900]))
plt.colorbar()

# %%


# find the distribution of cluster sizes over time for a single trace
for i in range(x.shape[1]):
    merge = x[1, i]
    counts = torch.bincount(merge).tolist()
    counts = [c for c in counts if c > 0]
    plt.plot([i for _ in range(len(counts))], counts, "bo", alpha=0.1)
plt.yscale("log")

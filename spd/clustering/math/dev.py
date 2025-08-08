# %%
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool


def to_onehot(
	x: Int[Tensor, " n_components"],
) -> Bool[Tensor, "k_groups n_components"]:
	k_groups: int = int(x.max().item() + 1)
	n_components: int = x.shape[0]
	device: torch.device = x.device
	mat: Bool[Tensor, "k_groups n_components"] = torch.zeros((k_groups, n_components), dtype=torch.bool, device=device)
	mat[x, torch.arange(n_components, device=device)] = True
	return mat


def pih_dev(
    X: Int[Tensor, " k n"],
) -> Float[Tensor, " k k"]:
	k_rows: int = X.shape[0]
	n_len: int = X.shape[1]

	dbg_auto(X)

	# for each row, compute the counts of each label
	# we can safely assume that the maximum label is les than n

	# create a mask for each row, true where the label has a count of 1 (is unique)

	# initialize the output matrix with NaNs
	distances: Float[Tensor, "k k"] = torch.full(
		(k_rows, k_rows),
		float('nan'),
		dtype=torch.float32,
	)
	# set lower triangular entries to 0
	distances.tril_()

	# compute (for all pairs of rows) the number of times both rows identify elements as being having a unique label

	# expand each row to a one-hot matrix, 



	return distances


from muutils.dbg import dbg_auto
import matplotlib.pyplot as plt

data_path = "../../../data/clustering/n4_b4_c3a6aa/distances/run_c3a6aa/ensemble_merge_array.npz"
x = torch.tensor(np.load(data_path)["merges"], dtype=torch.int32)
dbg_auto(x)


# c = 10
# for i_e, e in enumerate(x):
# 	for i_iter, r in enumerate(e):
# 		dbg_auto((i_e, i_iter))
# 		dbg_auto(r)
# 		r_1h = to_onehot(r)
# 		dbg_auto(r_1h.sum(dim=0))
# 		plt.matshow(r_1h, cmap="Blues")
# 		plt.show()
# 		c += 1
# 		if c > 20:
# 			break

plt.matshow(pih_dev(x[:, 10]))
plt.colorbar()

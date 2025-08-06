from __future__ import annotations

from functools import partial
import hashlib
import math
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import tqdm
from jaxtyping import Bool, Float, Int
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from muutils.parallel import run_maybe_parallel
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor

from spd.clustering.math.merge_matrix import BatchedGroupMerge, GroupMerge
from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix
from spd.spd_types import Probability


MergesAtIterArray = Int[np.ndarray, "n_ens c_components"]
MergesArray = Int[np.ndarray, "n_ens n_iters c_components"]
DistancesMethod = Literal["perm_invariant_hamming"]
DistancesArray = Float[np.ndarray, "n_iters n_ens n_ens"]


DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
	"perm_invariant_hamming": partial(perm_invariant_hamming_matrix, dtype=np.int16),
}

def compute_distances(
	normalized_merge_array: MergesArray,
	method: DistancesMethod = "perm_invariant_hamming",
) -> DistancesArray:
	n_iters: int = normalized_merge_array.shape[1]
	match method:
		case "perm_invariant_hamming":
			merges_array_list: list[Int[np.ndarray, "n_ens c_components"]] = [
				normalized_merge_array[:, i, :] for i in range(n_iters)
			]

			distances_list: list[Float[np.ndarray, "n_ens n_ens"]] = run_maybe_parallel(
				func=perm_invariant_hamming_matrix,
				iterable=merges_array_list,
				parallel=8,
			)

			return np.stack(distances_list, axis=0)
		case _:
			raise ValueError(f"Unknown distance method: {method}")
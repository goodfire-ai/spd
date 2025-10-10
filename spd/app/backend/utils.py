import torch
from jaxtyping import Float
from torch import Tensor

from spd.app.backend.api import SparseVector


def tensor_to_sparse_vector(tensor: Float[Tensor, " C"]) -> SparseVector:
    assert tensor.ndim == 1, "Tensor must be 1D"
    assert torch.all(tensor >= 0), "Tensor must be non-negative"
    return SparseVector(
        l0=int((tensor > 0).sum().item()),
        indices=tensor.nonzero().flatten().tolist(),
        values=tensor.tolist(),
    )

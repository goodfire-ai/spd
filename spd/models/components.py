from abc import ABC, abstractmethod
from typing import Literal, override

import einops
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from spd.utils.module_utils import init_param_

GateType = Literal["mlp", "vector_mlp"]


class ParallelLinear(nn.Module):
    """C parallel linear layers"""

    def __init__(self, C: int, input_dim: int, output_dim: int, nonlinearity: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(C, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(C, output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
        return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b


class GateMLPs(nn.Module):
    """MLP based gates that map a scalar input to a scalar output."""

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = 1 if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        x = einops.rearrange(x, "... C -> ... C 1")
        x = self.layers(x)
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorGateMLPs(nn.Module):
    """MLP based gates that map a vector valued input to a single output."""

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())

        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        # this 1 will broadcast out to actual C size, but no need to expand out yet
        x = self.layers(einops.rearrange(x, "... d_in -> ... 1 d_in"))
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class Components(ABC, nn.Module):
    def __init__(self, C: int, v_dim: int, u_dim: int):
        """
        Base class for components in a single layer (that would replace nn.Linear or nn.Embedding weight matrices).
        Initializes matrices V (which transforms the input activations) and U (which transforms the output of in_acts @ V)"

        Args:
            C: Number of components
            v_dim: Number of rows in the target weight matrix
            u_dim: Number of columns in the target weight matrix
        """
        super().__init__()
        self.C = C
        self.V = nn.Parameter(torch.empty(v_dim, C))
        self.U = nn.Parameter(torch.empty(C, u_dim))

    @property
    @abstractmethod
    def weight(self) -> Float[Tensor, "rows cols"]:
        raise NotImplementedError()

    def init_from_target_weight(self, target_weight: Tensor) -> None:
        """Initialize the V and U matrices.
        1. Normalize every component to 1.
        2. Take inner product with original model
        3. This gives you roughly how much overlap there is with the target model.
        4. Scale the Us by this value (we can choose either matrix)

        args:
            target_weight: The weight matrix of the original model. In the orientation of V @ U.
            Note that this is the transpose of the orientation of the weight matrix in the original code.
        """
        target_weight = target_weight.to(self.U.device)

        V = self.V
        U = self.U

        # Make V and U have unit norm in the d_in and d_out dimensions
        V.data[:] = torch.randn_like(V.data)
        U.data[:] = torch.randn_like(U.data)
        V.data[:] = V.data / V.data.norm(dim=-2, keepdim=True)
        U.data[:] = U.data / U.data.norm(dim=-1, keepdim=True)

        # Calculate inner products
        inner = einops.einsum(U, target_weight, "C cols, rows cols -> C rows")
        C_norms = einops.einsum(inner, V, "C rows, rows C -> C")

        # Scale U by the inner product.
        U.data[:] = U.data * C_norms.unsqueeze(-1)

    @override
    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Forward pass through the component."""
        raise NotImplementedError()

    @abstractmethod
    def get_inner_acts(self, x: Tensor) -> Tensor:
        """Get the inner acts of the component."""
        raise NotImplementedError()


class LinearComponents(Components):
    """A floating point linear component. The basic building block of SPD."""

    def __init__(
        self,
        C: int,
        d_in: int,
        d_out: int,
        bias: Tensor | None = None,
    ):
        super().__init__(C, v_dim=d_in, u_dim=d_out)  # NOTE: linear weights are (d_out, d_in)
        self.d_in = d_in
        self.d_out = d_out
        self.bias = bias

    @property
    @override
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """(V @ U).T. Transposed to match nn.Linear which uses (d_out, d_in)"""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    @override
    def get_inner_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(x, self.V, "... d_in, d_in C -> ... C")

    @override
    def forward(
        self, x: Float[Tensor, "... d_in"], mask: Tensor | None = None
    ) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        # V is (d_out, C). Multiply this way because we use (out, in) as in nn.Linear
        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponents(Components):
    """Efficient embedding components that avoid one-hot encoding."""

    def __init__(
        self,
        C: int,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__(C, v_dim=vocab_size, u_dim=embedding_dim)
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim

    @property
    @override
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """V @ U"""
        return einops.einsum(
            self.V, self.U, "vocab_size C, C embedding_dim -> vocab_size embedding_dim"
        )

    @override
    def get_inner_acts(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... C"]:
        return self.V[x]

    @override
    def forward(
        self,
        x: Int[Tensor, "..."],
        mask: Float[Tensor, "... C"] | Bool[Tensor, "... C"] | None,
    ) -> Float[Tensor, "... embedding_dim"]:
        """Forward through the embedding component using indexing instead of one-hot matmul.

        Args:
            x: Input tensor of token indices
            mask: Tensor which masks parameter components. May be boolean or float.
        """
        assert x.dtype == torch.long, "x must be an integer tensor"

        component_acts: Float[Tensor, "... C"] = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        out = einops.einsum(component_acts, self.U, "... C, C embedding_dim -> ... embedding_dim")
        return out


class ComponentsOrModule(nn.Module):
    def __init__(
        self,
        original: nn.Linear | nn.Embedding,
        components: Components,
    ):
        super().__init__()
        self.original = original
        self.components = components

        self.forward_mode: Literal["original"] | Literal["components"] | None = None
        self.mask: Tensor | None = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.forward_mode == "original":
            assert self.mask is None, "Mask should not be present in original mode"
            return self.original(x)
        elif self.forward_mode == "components":
            # mask *can* but doesn't *need to* be present here
            return self.components(x, self.mask)
        raise ValueError(f"Invalid forward mode: {self.forward_mode}")

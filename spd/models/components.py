from abc import ABC, abstractmethod
from typing import Literal, override

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers.modeling_utils import Conv1D as RadfordConv1D

from spd.utils.module_utils import _NonlinearityType, init_param_

GateType = Literal["mlp", "vector_mlp", "layerwise_global_mlp"]


class ParallelLinear(nn.Module):
    """C parallel linear layers"""

    def __init__(self, C: int, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(C, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(C, output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
        return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b


class Linear(nn.Module):
    """Linear layer with biases initialized to 0 and weights initialized using fan_val."""

    def __init__(self, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(x, self.W, "... d_in, d_in d_out -> ... d_out") + self.b


class GateMLPs(nn.Module):
    """MLP-based gates that map component 'inner acts' to a scalar output for each component."""

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
    """MLP-based gates that map a module's input vector to a scalar output for each component."""

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


class LayerwiseGlobalGateMLP(nn.Module):
    """MLP-based gate that maps a module's input vector to a scalar output for each component."""

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.layers.append(Linear(final_dim, C, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return self.layers(x)


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
        init_param_(self.V, fan_val=v_dim, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

    @property
    @abstractmethod
    def weight(self) -> Float[Tensor, "rows cols"]:
        raise NotImplementedError()

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

    bias: Float[Tensor, "... d_out"] | None

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

        # We don't train biases in SPD
        self.register_buffer("bias", bias)

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
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta: Float[Tensor, "... d_out d_in"] | None = None,
        weight_delta_mask: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components.
            weight_delta: Tensor which contains the weight differences between the target model and component weights.
            weight_delta_mask: Tensor which masks weight delta components.
        Returns:
            output: The summed output across all components
        """
        component_acts = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        # V is (d_out, C). Multiply this way because we use (out, in) as in nn.Linear
        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if weight_delta is not None:
            assert weight_delta_mask is not None
            unmasked_delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... d_out -> ... d_out"
            )

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
        mask: Float[Tensor, "... C"] | None,
        weight_delta: Float[Tensor, "... embedding_dim"] | None = None,
        weight_delta_mask: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... embedding_dim"]:
        """Forward through the embedding component using indexing instead of one-hot matmul.

        Args:
            x: Input tensor of token indices
            mask: Tensor which masks parameter components. May be boolean or float.
            weight_delta: Tensor which contains the weight differences between the target model and
                component weights.
            weight_delta_mask: Tensor which masks weight delta components.
        """
        assert x.dtype == torch.long, "x must be an integer tensor"

        component_acts: Float[Tensor, "... C"] = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        out = einops.einsum(component_acts, self.U, "... C, C embedding_dim -> ... embedding_dim")

        if weight_delta is not None:
            assert weight_delta_mask is not None
            unmasked_delta_out = weight_delta[x]
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... embedding_dim -> ... embedding_dim"
            )

        return out


class ComponentsOrModule(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        components: Components | None = None,
        identity_components: Components | None = None,
    ):
        super().__init__()
        assert components is not None or identity_components is not None, (
            "At least one of components or identity_components must be provided"
        )

        self.original = original
        self.components = components
        self.identity_components = identity_components

        self.forward_mode: Literal["original"] | Literal["components"] | None = None
        self.mask: Tensor | None = None
        self.identity_mask: Tensor | None = None
        self.identity_weight_delta: Tensor | None = None
        self.weight_delta: Tensor | None = None
        self.weight_delta_mask: Tensor | None = None
        self.identity_weight_delta_mask: Tensor | None = None

    @property
    def components_weight(self) -> Float[Tensor, "rows cols"]:
        assert self.components is not None
        return self.components.weight

    @property
    def identity_weight(self) -> Float[Tensor, "d d"]:
        assert self.identity_components is not None
        return self.identity_components.weight

    @property
    def original_weight(self) -> Float[Tensor, "rows cols"]:
        if isinstance(self.original, RadfordConv1D):
            return self.original.weight.T
        elif isinstance(self.original, nn.Linear | nn.Embedding):
            return self.original.weight
        else:
            raise AttributeError(
                f"Module {type(self.original)} not one of nn.Linear, nn.Embedding, or RadfordConv1D"
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.forward_mode == "original":
            assert self.mask is None and self.identity_mask is None
            x = self.original(x)
        elif self.forward_mode == "components":
            if self.identity_mask is not None:
                assert self.identity_components is not None
                x = self.identity_components(
                    x,
                    mask=self.identity_mask,
                    weight_delta=self.identity_weight_delta,
                    weight_delta_mask=self.identity_weight_delta_mask,
                )

            if self.mask is not None:
                assert self.components is not None
                x = self.components(
                    x,
                    mask=self.mask,
                    weight_delta=self.weight_delta,
                    weight_delta_mask=self.weight_delta_mask,
                )
            else:
                x = self.original(x)
        else:
            raise ValueError(f"Invalid forward mode: {self.forward_mode}")
        return x

    def make_pristine(self) -> None:
        """Set forward_mode, mask, and identity_mask to None."""
        self.forward_mode = None
        self.mask = None
        self.identity_mask = None
        self.weight_delta = None
        self.weight_delta_mask = None
        self.identity_weight_delta = None
        self.identity_weight_delta_mask = None

    def assert_pristine(self) -> None:
        """Assert that forward_mode, mask, and identity_mask are None."""
        assert self.forward_mode is None, f"forward_mode should be None, got {self.forward_mode}"
        assert self.mask is None, f"mask should be None, got {self.mask}"
        assert self.identity_mask is None, f"identity_mask should be None, got {self.identity_mask}"
        assert self.weight_delta is None, f"weight_delta should be None, got {self.weight_delta}"
        assert self.weight_delta_mask is None, (
            f"weight_delta_mask should be None, got {self.weight_delta_mask}"
        )
        assert self.identity_weight_delta is None, (
            f"identity_weight_delta should be None, got {self.identity_weight_delta}"
        )
        assert self.identity_weight_delta_mask is None, (
            f"identity_weight_delta_mask should be None, got {self.identity_weight_delta_mask}"
        )

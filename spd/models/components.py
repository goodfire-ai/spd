from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, override

import einops
import torch
from jaxtyping import Bool, Float, Int
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
    """Contains a separate network for each component and takes a module's input vector as input."""

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


WeightDeltaAndMask = tuple[Float[Tensor, " d_out d_in"], Float[Tensor, "..."]]


class LayerwiseGlobalGateMLP(nn.Module):
    """Maps a module's input vector to a scalar output for each component with a 'pure' MLP."""

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
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
    def forward(
        self,
        x: Tensor,
        mask: Tensor | bool | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Tensor:
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
        mask: Float[Tensor, "... C"] | bool | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components.
            weight_delta_and_mask: Optional tuple of tensors containing:
                0: the weight differences between the target model and summed component weights
                1: mask over the weight delta component for each sample
        Returns:
            output: The summed output across all components
        """
        component_acts = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        # V is (d_out, C). Multiply this way because we use (out, in) as in nn.Linear
        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
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
        mask: Float[Tensor, "... C"] | bool | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Float[Tensor, "... embedding_dim"]:
        """Forward through the embedding component using indexing instead of one-hot matmul.

        Args:
            x: Input tensor of token indices
            mask: Tensor which masks parameter components. May be boolean or float.
            weight_delta_and_mask: Optional tuple of tensors containing:
                0: the weight differences between the target model and summed component weights
                1: mask over the weight delta component for each sample
        """
        assert x.dtype == torch.long, "x must be an integer tensor"

        component_acts: Float[Tensor, "... C"] = self.get_inner_acts(x)

        if mask is not None:
            component_acts *= mask

        out = einops.einsum(component_acts, self.U, "... C, C embedding_dim -> ... embedding_dim")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = weight_delta[x]
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... embedding_dim -> ... embedding_dim"
            )

        return out


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied to a ComponentOrModule object."""

    routing_mask: Float[Tensor, " ..."] | Literal[True]
    """specifies which positions (usually batch, seq) to route to components vs target"""

    component_mask: Float[Tensor, "... C"] | bool
    """when components are active, this specifies which subcomponents to use"""

    weight_delta_and_mask: WeightDeltaAndMask | None


def make_mask_infos(
    component_masks: Mapping[str, Float[Tensor, "... C"] | bool],
    routing_masks: Mapping[str, Bool[Tensor, "..."] | Literal[True]] | None = None,
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Create ComponentsMaskInfo dict from dicts of component masks, routing masks, and weight deltas and weight delta masks.
    Keys of all dicts must be the same.

    Args:
        component_masks: Dict of component masks.
        routing_masks: Dict of routing masks. Defaults to True (enable components) for all outputs if not provided.
        weight_deltas_and_masks: Dict of weight deltas and masks for each module to be decomposed. Defaults to None (disable weight delta component) if not provided.
    turns:
        Dict mapping module names to ComponentsMaskInfo objects.
    """
    if routing_masks is not None:
        assert set(routing_masks) == set(component_masks)

    if weight_deltas_and_masks is not None:
        assert set(weight_deltas_and_masks) == set(component_masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in component_masks:
        result[name] = ComponentsMaskInfo(
            routing_mask=routing_masks[name] if routing_masks is not None else True,
            component_mask=component_masks[name],
            weight_delta_and_mask=None
            if weight_deltas_and_masks is None
            else weight_deltas_and_masks[name],
        )

    return result


# class ComponentsOrModule(nn.Module):
#     def __init__(
#         self,
#         target: nn.Module,
#         components: Components,
#     ):
#         super().__init__()
#         self.target = target
#         self.components = components

#         self.forward_mode: (
#             None | Literal["target"] | tuple[Literal["mixed"], ComponentsMaskInfo]
#         ) = None

#     @property
#     def target_weight(self) -> Float[Tensor, "rows cols"]:
#         if isinstance(self.target, RadfordConv1D):
#             return self.target.weight.T
#         elif isinstance(self.target, nn.Linear | nn.Embedding):
#             return self.target.weight
#         else:
#             raise AttributeError(
#                 f"Module {type(self.target)} not one of nn.Linear, nn.Embedding, or RadfordConv1D"
#             )

#     @override
#     def forward(self, x: Tensor) -> Tensor:
#         assert self.forward_mode is not None

#         match self.forward_mode:
#             case "target":
#                 return self.target(x)
#             case ("mixed", mask_info):
#                 target_out = self.target(x)
#                 mask = mask_info.component_mask
#                 weight_delta_and_mask = mask_info.weight_delta_and_mask
#                 components_out = self.components.forward(x, mask, weight_delta_and_mask)

#                 # this allows passing in bools to mean all or none, instead of requiring a cumbersome torch.ones_like(...) or torch.zeros_like(...)
#                 routing_mask = (
#                     mask_info.routing_mask[..., None]
#                     if isinstance(mask_info.routing_mask, Tensor)
#                     else torch.tensor(mask_info.routing_mask)
#                 )
#                 return torch.where(routing_mask, components_out, target_out)

#     def make_pristine(self) -> None:
#         self.forward_mode = None

#     def assert_pristine(self) -> None:
#         assert self.forward_mode is None, f"forward_mode should be None, got {self.forward_mode}"

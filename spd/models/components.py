import math
from typing import Literal, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.utils.module_utils import init_param_

GateType = Literal["mlp", "vector_mlp", "star_graph"]


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


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a scalar input to a scalar output."""

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


class VectorGateMLP(nn.Module):
    """An MLP based gate that maps a vector valued input to a single output."""

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


class StarGraphGate(nn.Module):
    """
    Representing relationships as a star-graph allows us to compute everything efficiently
    with broadcasting and einsum.
    """

    def __init__(self, C: int, node_dims: list[int], components: nn.ModuleDict):
        super().__init__()
        self.C = C
        assert len(node_dims) == 1, "StarGraphGate only supports a single node dimension"
        self.d_node = node_dims[0]  # Assuming all nodes have the same dimension
        self.n = len(components)

        # Each subcomponent (in each layer) gets its own MLP
        # As per the original implementation
        self.mlp_in = nn.Parameter(torch.empty((C, self.n, self.d_node)))
        self.in_bias = nn.Parameter(torch.zeros((C, self.n, self.d_node)))
        self.mlp_out = nn.Parameter(torch.empty((C, self.n, self.d_node)))
        self.out_bias = nn.Parameter(torch.zeros((C, self.n)))
        # Additionally, each component gets its own summary MLP
        self.mlp_summary = nn.Parameter(torch.empty((self.n, self.d_node)))
        self.bias_summary = nn.Parameter(torch.zeros((self.n, self.d_node)))

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=self.d_node, nonlinearity="linear")
        init_param_(self.mlp_summary, fan_val=self.d_node, nonlinearity="linear")

    @override
    def forward(self, inner_act: dict[str, Tensor]) -> dict[str, Float[Tensor, "... C n"]]:
        x = torch.stack(list(inner_act.values()), dim=-1)

        sub = einops.einsum(x, self.mlp_in, "... C n, C n d_node -> ... C n d_node") + self.in_bias
        sub = nn.functional.gelu(sub)

        summary = sub.sum(dim=-3) / self.C
        summary = (
            einops.einsum(summary, self.mlp_summary, "... n d_node, n d_node -> ... n d_node")
            + self.bias_summary
        )
        summary = nn.functional.gelu(summary)

        global_sum = summary.sum(dim=-2, keepdim=True) / self.n

        summary = summary + global_sum

        sub = sub + summary.unsqueeze(-3)

        scale = 1 / math.sqrt(self.d_node)
        scores = (
            einops.einsum(sub, self.mlp_out, "... C n d_node, C n d_node -> ... C n") * scale
            + self.out_bias
        )

        return {name: scores[..., :, i] for i, name in enumerate(inner_act)}


class LinearComponent(nn.Module):
    """A linear transformation made from V and U matrices for SPD.

    The weight matrix W is decomposed as W = U^T @ V^T, where V and U are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, bias: Tensor | None):
        super().__init__()
        self.C = C
        self.d_in = d_in
        self.d_out = d_out

        self.V = nn.Parameter(torch.empty(d_in, C))
        self.U = nn.Parameter(torch.empty(C, d_out))
        self.bias = bias

        init_param_(self.V, fan_val=d_out, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """U^T @ V^T"""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.V, "... d_in, d_in C -> ... C")

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.C: int = C

        self.V: nn.Parameter = nn.Parameter(torch.empty(vocab_size, C))
        self.U: nn.Parameter = nn.Parameter(torch.empty(C, embedding_dim))

        init_param_(self.V, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """V @ U"""
        return einops.einsum(
            self.V, self.U, "vocab_size C, C embedding_dim -> vocab_size embedding_dim"
        )

    @override
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.V[x]  # (batch pos C)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.U, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out

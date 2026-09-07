"""The LM model-output edge: materialized logits or the factored streamed package.

`LMOutput` is the `Out` every LM target binds (`core.model.ForwardResult[LMOutput]`); the
protocol output operations dispatch on it exhaustively. The comparison kernels live in
`targets.losses`.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
from jax.sharding import Mesh
from jaxtyping import Array, Float

from param_decomp.core.sharding import batch_shard_leading


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("activations", "head"),
    meta_fields=("n_chunks",),
)
@dataclass(frozen=True)
class StreamedLinearOutput:
    """A model output left FACTORED at its final linear map: `activations @ head.T` IS
    the materialized output (an LM's logits), deliberately never formed — at a 248k
    vocab one `[B, S, vocab]` buffer dwarfs everything else in the arena. Consumers
    that compare outputs stream over the head's output axis in `n_chunks` chunks with
    fp32 online-softmax accumulators (`targets.losses`).

    The head rides the package because comparison fns see only outputs; under jit it is
    the same traced array as the model's stored weight — a reference, never a copy. It
    has no batch axis, so batch pinning applies to `activations` alone; the head stays at
    the target's declared operand layout.
    """

    activations: Float[Array, "*leading d_model"]
    head: Float[Array, "d_out d_model"]
    n_chunks: int


LMOutput = Array | StreamedLinearOutput
"""The LM output edge. The clean and masked forwards of one model always share one
member; every pairwise consumer refuses a mix."""


def pin_lm_output_batch(output: LMOutput, mesh: Mesh | None) -> LMOutput:
    match output:
        case StreamedLinearOutput():
            return replace(output, activations=batch_shard_leading(output.activations, mesh))
        case jax.Array():
            return batch_shard_leading(output, mesh)

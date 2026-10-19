"""`KL(clean pattern ‖ masked pattern)` for every row of one causal attention layer, from the
operands of its scores, without forming a `[batch, head, query, key]` array.

The interface is `attention_pattern_kl(clean, masked)` on two `AttentionScores`: the q and k a
layer's scores are dot products of, each token's document, the layer's `AttentionPatternDefinition`
and its attention implementation. `dense_log_pattern` forms one pattern whole, for evals. Patterns are
causal: a query attends to keys at or before it, further restricted by the layer's definition.

What a pattern *is* belongs to the model: an `AttentionPatternDefinition` turns raw dot products `q·kᵀ` into
logits and says which earlier keys each query attends to. The kernels compose it per tile at
trace time (no runtime callback); the backward takes the logits' derivative by `jax.jvp`, so
a definition only states its forward.

Flash layers run Pallas Triton kernels (any CUDA GPU; the Pallas interpreter elsewhere).
Forward, per query tile, over key tiles up to the diagonal: both score tiles, running softmax
statistics for both, and the running `Σ P·(s − s')` with the clean pattern's rescaling. Each
row's KL is `E_P[s − s'] − LSE(s) + LSE(s')`; the two LSEs are saved for the backward.
Backward: `∂KL_t/∂s'_tj = Q_tj − P_tj`, so with the row cotangent `g_t` and
`D = ∂s'/∂(q'·k'ᵀ)`, `dq'_t = Σ_j g_t (Q − P)_tj D_tj k'_j` and
`dk'_j = Σ_t g_t (Q − P)_tj D_tj q'_t`, each recomputing both tiles from the saved LSEs.
Every sum is FP32. XLA layers form both patterns, as their attention does anyway.

Grouped-query attention reads key head `h // (q_heads // kv_heads)`. The clean side is a
constant: it receives no gradient.
"""

import functools
import math
from typing import NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Float32, Int

from param_decomp.attention import AttentionImplementation

_LOG2E = math.log2(math.e)
_MASKED = -0.7 * float(jnp.finfo(jnp.float32).max)
"""Score of a masked key: finite, so fully masked tiles rescale away instead of producing NaN."""
_TILE = 64
"""Query and key tile of the Triton kernels; sequence lengths must be multiples of it."""
_NUM_WARPS = 4


class AttentionPatternDefinition(Protocol):
    """How one causal attention layer turns raw dot products `q·kᵀ` into logits, and which
    earlier keys each query attends to: the definition of the pattern the KL compares. Static
    (a hashable value) and composed into the kernels per tile. Indices are token positions.
    Arguments are positional, so a definition names the ones it ignores as such."""

    def logits(self, dots: Array, q_index: Array, k_index: Array, /) -> Array:
        """FP32 logits of a `[q, k]` block from its FP32 dot products, elementwise in `dots`."""
        ...

    def attends(
        self, q_index: Array, k_index: Array, q_document: Array, k_document: Array, /
    ) -> Array:
        """Whether each query of a `[q, k]` block attends to each key at or before it."""
        ...


def _causal_attends(
    definition: AttentionPatternDefinition,
    q_index: Array,
    k_index: Array,
    q_document: Array,
    k_document: Array,
) -> Array:
    """The causal pattern's support: keys at or before the query that `definition` attends to."""
    causal = q_index[:, None] >= k_index[None, :]
    return causal & definition.attends(q_index, k_index, q_document, k_document)


def _dots(q: Array, k: Array) -> Array:
    return plgpu.dot(q, k.T)


def _masked_base2(
    definition: AttentionPatternDefinition,
    dots: Array,
    q_index: Array,
    k_index: Array,
    attends: Array,
) -> Array:
    return jnp.where(attends, definition.logits(dots, q_index, k_index) * _LOG2E, _MASKED)


def _forward_kernel(
    clean_q_ref: jax.Ref,
    clean_k_ref: jax.Ref,
    masked_q_ref: jax.Ref,
    masked_k_ref: jax.Ref,
    document_ref: jax.Ref,
    kl_ref: jax.Ref,
    clean_lse_ref: jax.Ref,
    masked_lse_ref: jax.Ref,
    *,
    definition: AttentionPatternDefinition,
    block: int,
):
    q_tile = pl.program_id(0)
    q_index = q_tile * block + jnp.arange(block)
    q_document = document_ref[pl.ds(q_tile * block, block)]
    clean_q, masked_q = clean_q_ref[...], masked_q_ref[...]

    def body(
        k_tile: Array, carry: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        clean_max, clean_sum, score_drop, masked_max, masked_sum = carry
        ks = pl.ds(k_tile * block, block)
        k_index = k_tile * block + jnp.arange(block)
        attends = _causal_attends(definition, q_index, k_index, q_document, document_ref[ks])
        clean_scores = _masked_base2(
            definition, _dots(clean_q, clean_k_ref[ks, :]), q_index, k_index, attends
        )
        masked_scores = _masked_base2(
            definition, _dots(masked_q, masked_k_ref[ks, :]), q_index, k_index, attends
        )

        new_clean_max = jnp.maximum(clean_max, jnp.max(clean_scores, axis=-1))
        correction = jnp.exp2(clean_max - new_clean_max)
        clean_pattern = jnp.exp2(clean_scores - new_clean_max[:, None])
        clean_sum = correction * clean_sum + jnp.sum(clean_pattern, axis=-1)
        score_drop = correction * score_drop + jnp.sum(
            clean_pattern * (clean_scores - masked_scores), axis=-1
        )

        new_masked_max = jnp.maximum(masked_max, jnp.max(masked_scores, axis=-1))
        masked_sum = jnp.exp2(masked_max - new_masked_max) * masked_sum + jnp.sum(
            jnp.exp2(masked_scores - new_masked_max[:, None]), axis=-1
        )
        return new_clean_max, clean_sum, score_drop, new_masked_max, masked_sum

    minus_inf = jnp.full((block,), -jnp.inf, jnp.float32)
    zeros = jnp.zeros((block,), jnp.float32)
    clean_max, clean_sum, score_drop, masked_max, masked_sum = lax.fori_loop(
        0, q_tile + 1, body, (minus_inf, zeros, zeros, minus_inf, zeros)
    )
    clean_lse = clean_max + jnp.log2(clean_sum)
    masked_lse = masked_max + jnp.log2(masked_sum)
    kl_ref[...] = (score_drop / clean_sum - clean_lse + masked_lse) / _LOG2E
    clean_lse_ref[...] = clean_lse
    masked_lse_ref[...] = masked_lse


def _pattern_gap(
    definition: AttentionPatternDefinition,
    clean_q: Array,
    clean_k: Array,
    masked_q: Array,
    masked_k: Array,
    q_index: Array,
    k_index: Array,
    attends: Array,
    clean_lse: Array,
    masked_lse: Array,
    cotangent: Array,
) -> Array:
    """`g · (Q − P) · ∂s'/∂(q'·k'ᵀ)` for one tile, in FP32, zero outside the mask: the
    cotangent of the masked tile's dot products. Patterns are exponentiated unmasked (a
    masked entry may overflow) and the one final select discards them."""
    clean_logits = definition.logits(_dots(clean_q, clean_k), q_index, k_index)
    masked_dots = _dots(masked_q, masked_k)
    masked_logits, logits_slope = jax.jvp(
        lambda dots: definition.logits(dots, q_index, k_index),
        (masked_dots,),
        (jnp.ones_like(masked_dots),),
    )
    clean_pattern = jnp.exp2(clean_logits * _LOG2E - clean_lse[:, None])
    masked_pattern = jnp.exp2(masked_logits * _LOG2E - masked_lse[:, None])
    return jnp.where(
        attends, cotangent[:, None] * (masked_pattern - clean_pattern) * logits_slope, 0.0
    )


def _dq_kernel(
    clean_q_ref: jax.Ref,
    clean_k_ref: jax.Ref,
    masked_q_ref: jax.Ref,
    masked_k_ref: jax.Ref,
    document_ref: jax.Ref,
    clean_lse_ref: jax.Ref,
    masked_lse_ref: jax.Ref,
    cotangent_ref: jax.Ref,
    dq_ref: jax.Ref,
    *,
    definition: AttentionPatternDefinition,
    block: int,
):
    q_tile = pl.program_id(0)
    q_index = q_tile * block + jnp.arange(block)
    q_document = document_ref[pl.ds(q_tile * block, block)]
    clean_q, masked_q = clean_q_ref[...], masked_q_ref[...]
    clean_lse, masked_lse, cotangent = clean_lse_ref[...], masked_lse_ref[...], cotangent_ref[...]

    def body(k_tile: Array, dq: Array) -> Array:
        ks = pl.ds(k_tile * block, block)
        k_index = k_tile * block + jnp.arange(block)
        masked_k = masked_k_ref[ks, :]
        gap = _pattern_gap(
            definition,
            clean_q,
            clean_k_ref[ks, :],
            masked_q,
            masked_k,
            q_index,
            k_index,
            _causal_attends(definition, q_index, k_index, q_document, document_ref[ks]),
            clean_lse,
            masked_lse,
            cotangent,
        )
        return dq + plgpu.dot(gap.astype(masked_k.dtype), masked_k)

    dq = lax.fori_loop(0, q_tile + 1, body, jnp.zeros((block, masked_q.shape[-1]), jnp.float32))
    dq_ref[...] = dq.astype(dq_ref.dtype)


def _dk_kernel(
    clean_q_ref: jax.Ref,
    clean_k_ref: jax.Ref,
    masked_q_ref: jax.Ref,
    masked_k_ref: jax.Ref,
    document_ref: jax.Ref,
    clean_lse_ref: jax.Ref,
    masked_lse_ref: jax.Ref,
    cotangent_ref: jax.Ref,
    dk_ref: jax.Ref,
    *,
    definition: AttentionPatternDefinition,
    block: int,
):
    k_tile = pl.program_id(0)
    k_index = k_tile * block + jnp.arange(block)
    k_document = document_ref[pl.ds(k_tile * block, block)]
    clean_k, masked_k = clean_k_ref[...], masked_k_ref[...]

    def body(q_tile: Array, dk: Array) -> Array:
        qs = pl.ds(q_tile * block, block)
        q_index = q_tile * block + jnp.arange(block)
        masked_q = masked_q_ref[qs, :]
        gap = _pattern_gap(
            definition,
            clean_q_ref[qs, :],
            clean_k,
            masked_q,
            masked_k,
            q_index,
            k_index,
            _causal_attends(definition, q_index, k_index, document_ref[qs], k_document),
            clean_lse_ref[qs],
            masked_lse_ref[qs],
            cotangent_ref[qs],
        )
        return dk + plgpu.dot(gap.astype(masked_q.dtype).T, masked_q)

    n_tiles = clean_q_ref.shape[0] // block
    dk = lax.fori_loop(k_tile, n_tiles, body, jnp.zeros((block, masked_k.shape[-1]), jnp.float32))
    dk_ref[...] = dk.astype(dk_ref.dtype)


class _BlockSpecs(NamedTuple):
    """Operand blocks for a grid `(seq_len // block, batch, q_heads)`."""

    q_tile: pl.BlockSpec
    """This tile of a q-head array."""
    q_whole: pl.BlockSpec
    """The whole sequence of a q-head array."""
    kv_tile: pl.BlockSpec
    """This tile of the kv-head array the q head reads."""
    kv_whole: pl.BlockSpec
    """The whole sequence of the kv-head array the q head reads."""
    rows_tile: pl.BlockSpec
    """This tile of a per-row vector."""
    rows_whole: pl.BlockSpec
    """The whole sequence of a per-row vector."""
    documents: pl.BlockSpec
    """The batch row's document ids."""


def _block_specs(seq_len: int, n_rep: int, head_dim: int, block: int) -> _BlockSpecs:
    def heads(extent: int, kv: bool, tiled: bool) -> pl.BlockSpec:
        return pl.BlockSpec(
            (None, extent, None, head_dim),
            lambda i, b, h: (b, i if tiled else 0, h // n_rep if kv else h, 0),
        )

    return _BlockSpecs(
        q_tile=heads(block, kv=False, tiled=True),
        q_whole=heads(seq_len, kv=False, tiled=False),
        kv_tile=heads(block, kv=True, tiled=True),
        kv_whole=heads(seq_len, kv=True, tiled=False),
        rows_tile=pl.BlockSpec((None, None, block), lambda i, b, h: (b, h, i)),
        rows_whole=pl.BlockSpec((None, None, seq_len), lambda i, b, h: (b, h, 0)),
        documents=pl.BlockSpec((None, seq_len), lambda i, b, h: (b, 0)),
    )


def _forward(
    clean_q: Array,
    clean_k: Array,
    masked_q: Array,
    masked_k: Array,
    document_ids: Array,
    definition: AttentionPatternDefinition,
) -> tuple[Float32[Array, "b qh t"], Float32[Array, "b qh t"], Float32[Array, "b qh t"]]:
    """Each row's KL with both patterns' base-2 log-normalizers."""
    batch, seq_len, q_heads, head_dim = clean_q.shape
    specs = _block_specs(seq_len, q_heads // clean_k.shape[2], head_dim, _TILE)
    rows = jax.ShapeDtypeStruct((batch, q_heads, seq_len), jnp.float32)
    return pl.pallas_call(
        functools.partial(_forward_kernel, definition=definition, block=_TILE),
        grid=(seq_len // _TILE, batch, q_heads),
        in_specs=[specs.q_tile, specs.kv_whole, specs.q_tile, specs.kv_whole, specs.documents],
        out_specs=[specs.rows_tile] * 3,
        out_shape=[rows] * 3,
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=2),
        interpret=_interpreted(),
        name="attention_kl_forward",
    )(clean_q, clean_k, masked_q, masked_k, document_ids)


_Residuals = tuple[Array, Array, Array, Array, Array, Array, Array]
"""The forward's operands and both patterns' log-normalizers, saved for the backward:
`(clean_q, clean_k, masked_q, masked_k, document_ids, clean_lse, masked_lse)`."""


def _backward(
    residuals: _Residuals, cotangent: Array, definition: AttentionPatternDefinition
) -> tuple[Array, Array]:
    """Gradients of `Σ cotangent · KL` with respect to `masked_q` and `masked_k`."""
    clean_q, clean_k, masked_q, masked_k, _, _, _ = residuals
    batch, seq_len, q_heads, head_dim = clean_q.shape
    kv_heads = clean_k.shape[2]
    n_rep = q_heads // kv_heads
    specs = _block_specs(seq_len, n_rep, head_dim, _TILE)
    call = functools.partial(
        pl.pallas_call,
        grid=(seq_len // _TILE, batch, q_heads),
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=2),
        interpret=_interpreted(),
    )
    operands = (*residuals, cotangent)
    dq = call(
        functools.partial(_dq_kernel, definition=definition, block=_TILE),
        in_specs=[specs.q_tile, specs.kv_whole, specs.q_tile, specs.kv_whole, specs.documents]
        + [specs.rows_tile] * 3,
        out_specs=specs.q_tile,
        out_shape=jax.ShapeDtypeStruct(masked_q.shape, masked_q.dtype),
        name="attention_kl_dq",
    )(*operands)
    # One key-gradient block per query head; heads sharing a key head are summed after.
    dk_per_q_head = call(
        functools.partial(_dk_kernel, definition=definition, block=_TILE),
        in_specs=[specs.q_whole, specs.kv_tile, specs.q_whole, specs.kv_tile, specs.documents]
        + [specs.rows_whole] * 3,
        out_specs=specs.q_tile,
        out_shape=jax.ShapeDtypeStruct((batch, seq_len, q_heads, head_dim), jnp.float32),
        name="attention_kl_dk",
    )(*operands)
    dk = dk_per_q_head.reshape(batch, seq_len, kv_heads, n_rep, head_dim).sum(axis=3)
    return dq, dk.astype(masked_k.dtype)


def _interpreted() -> bool:
    """Whether the Triton kernels run through the Pallas interpreter: off a GPU."""
    return jax.default_backend() != "gpu"


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _attention_kl(
    clean_q: Array,
    clean_k: Array,
    masked_q: Array,
    masked_k: Array,
    document_ids: Array,
    definition: AttentionPatternDefinition,
) -> Array:
    return _forward(clean_q, clean_k, masked_q, masked_k, document_ids, definition)[0]


def _attention_kl_fwd(
    clean_q: Array,
    clean_k: Array,
    masked_q: Array,
    masked_k: Array,
    document_ids: Array,
    definition: AttentionPatternDefinition,
) -> tuple[Array, _Residuals]:
    kl, clean_lse, masked_lse = _forward(
        clean_q, clean_k, masked_q, masked_k, document_ids, definition
    )
    return kl, (clean_q, clean_k, masked_q, masked_k, document_ids, clean_lse, masked_lse)


def _attention_kl_bwd(
    definition: AttentionPatternDefinition, residuals: _Residuals, cotangent: Array
) -> tuple[None, None, Array, Array, None]:
    dq, dk = _backward(residuals, cotangent, definition)
    return None, None, dq, dk, None


_attention_kl.defvjp(_attention_kl_fwd, _attention_kl_bwd)


class AttentionScores(eqx.Module):
    """One forward's attention scores as their operands rather than an array: the q and k
    whose dot products they are (post-norm, post-RoPE, token-before-head), each token's
    document, the definition that turns the dot products into the layer's pattern, and the
    implementation the layer's attention ran with — a flash layer never forms its pattern,
    so neither may a comparison of its scores."""

    q: Float[Array, "b t qh hd"]
    k: Float[Array, "b t kvh hd"]
    document_ids: Int[Array, "b t"]
    pattern_definition: AttentionPatternDefinition = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)


def attention_pattern_kl(
    clean: AttentionScores, masked: AttentionScores
) -> Float32[Array, "b qh t"]:
    """`KL(clean pattern ‖ masked pattern)` per attention row between one layer's scores on
    the same tokens, whose documents `masked` carries; only `masked`'s q and k receive
    gradients. Flash and Mosaic layers never form their pattern, so their scores run the
    kernels on each device's rows of `masked.q`'s batch split; XLA layers' form both
    patterns."""
    assert clean.pattern_definition == masked.pattern_definition, (clean, masked)
    assert clean.implementation == masked.implementation, (clean, masked)
    assert clean.q.shape == masked.q.shape and clean.k.shape == masked.k.shape, (clean, masked)
    match masked.implementation:
        case "flash" | "mosaic":
            return _fused_attention_kl(clean, masked)
        case "xla":
            documents, definition = masked.document_ids, masked.pattern_definition
            log_p = jax.lax.stop_gradient(
                dense_log_pattern(clean.q, clean.k, documents, definition=definition)
            )
            log_q = dense_log_pattern(masked.q, masked.k, documents, definition=definition)
            attended = log_p > -jnp.inf
            return jnp.sum(jnp.where(attended, jnp.exp(log_p) * (log_p - log_q), 0.0), -1)


def dense_log_pattern(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b t kvh hd"],
    document_ids: Int[Array, "b t"],
    *,
    definition: AttentionPatternDefinition,
) -> Float32[Array, "b qh t t"]:
    """The log-softmax causal pattern `definition` defines, whole: keys a query does not attend
    to are `-inf`, and attended keys keep a finite log-probability where the probability
    underflows."""
    positions = jnp.arange(q.shape[1])
    k = jnp.repeat(k, q.shape[2] // k.shape[2], axis=2)
    dots = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
    logits = definition.logits(dots, positions, positions)
    attends = jax.vmap(
        lambda documents: _causal_attends(definition, positions, positions, documents, documents)
    )(document_ids)
    return jax.nn.log_softmax(jnp.where(attends[:, None], logits, -jnp.inf), axis=-1)


def _fused_attention_kl(clean: AttentionScores, masked: AttentionScores) -> Array:
    """The Triton kernels, on each device's rows of `masked.q`'s batch split (heads stay
    whole, as in flash attention)."""
    batch, seq_len, q_heads, head_dim = masked.q.shape
    assert q_heads % masked.k.shape[2] == 0, (masked.q.shape, masked.k.shape)
    assert masked.document_ids.shape == (batch, seq_len), masked.document_ids.shape
    assert seq_len % _TILE == 0, f"sequence length {seq_len} must be a multiple of {_TILE}"
    assert head_dim & (head_dim - 1) == 0 and head_dim >= 16, f"head_dim {head_dim}"

    def local(
        clean_q: Array, clean_k: Array, masked_q: Array, masked_k: Array, document_ids: Array
    ) -> Array:
        return _attention_kl(
            jax.lax.stop_gradient(clean_q), jax.lax.stop_gradient(clean_k), masked_q, masked_k,
            document_ids, masked.pattern_definition,
        )  # fmt: skip

    operands = (clean.q, clean.k, masked.q, masked.k, masked.document_ids)
    sharding = jax.typeof(masked.q).sharding
    if sharding.mesh.empty:
        return local(*operands)
    rows = sharding.spec[0] if sharding.spec else None
    tokens = P(rows, None, None, None)
    # Pallas outputs declare no varying mesh axes (vma), which the checker refuses inside
    # a manual map; the specs here are the contract.
    return jax.shard_map(
        local,
        mesh=sharding.mesh,
        in_specs=(tokens, tokens, tokens, tokens, P(rows, None)),
        out_specs=P(rows, None, None),
        check_vma=False,
    )(*operands)

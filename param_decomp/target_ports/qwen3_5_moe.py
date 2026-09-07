"""JAX ports of the `qwen3_5_moe` numeric kernels (the HF architecture behind
Qwen3.6-*-A3B): zero-centered RMSNorm, the gated-DeltaNet pieces (causal depthwise conv,
l2norm, the gated delta rule), the gated output norm, and partial RoPE. NOT written from
memory: every computational convention mirrors
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (transformers 4.57 lineage,
generated from `modular_qwen3_5_moe.py`); the one deliberate departure is the chunkwise
arm's triangular solve (blocked substitution instead of HF's row loop — same system,
stability note at `_solve_unit_lower_triangular`).

Two RMSNorm conventions coexist in this architecture and must never be conflated:
every `*_layernorm` / `q_norm` / `k_norm` / final `norm` weight is stored ZERO-CENTERED
and applies as `(1 + w)` with the multiply in fp32 (`rms_norm_zero_centered`); the
DeltaNet output norm alone stores a plain ones-convention weight (`gated_rms_norm`).

The gated delta rule ships as a closed two-arm enumeration (`GatedDeltaKernel`), both
fp32-state per HF's `mamba_ssm_dtype: float32`: `sequential` is the mathematically-simple
per-token recurrence (one `lax.scan` over time, mirroring HF's
`torch_recurrent_gated_delta_rule`) and serves as the parity oracle; `chunkwise` is the
algebraically-identical WY form (HF's `torch_chunk_gated_delta_rule`), whose sequential
dependency is one scan step per `GATED_DELTA_CHUNK`-token chunk instead of one per token.
"""

import math
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from param_decomp.target_ports.llama import rotate_half


def rms_norm_zero_centered(
    x: Float[Array, "... d"], weight: Float[Array, " d"], eps: float
) -> Float[Array, "... d"]:
    """`Qwen3_5MoeRMSNorm`: the stored weight is zero-centered, applied as `(1 + w)`,
    with the weight multiply in fp32 BEFORE the downcast (`(x * w).to(dtype)`, not
    Llama's `x.to(dtype) * w`)."""
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def gated_rms_norm(
    x: Float[Array, "... d"],
    weight: Float[Array, " d"],
    gate: Float[Array, "... d"],
    eps: float,
) -> Float[Array, "... d"]:
    """`Qwen3_5MoeRMSNormGated` (the DeltaNet output norm): PLAIN ones-convention weight
    applied in the input dtype, then the silu gate applied in fp32 — norm before gate."""
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    out = weight * normed.astype(in_dtype)
    return (out * jax.nn.silu(gate.astype(jnp.float32))).astype(in_dtype)


def l2norm(x: Float[Array, "... d"], eps: float = 1e-6) -> Float[Array, "... d"]:
    """FLA-convention l2 normalization: eps added to the SUM of squares inside rsqrt
    (not to the norm), computed in the input dtype."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_depthwise_conv1d_silu(
    x: Float[Array, "b t c"], weight: Float[Array, "c k"]
) -> Float[Array, "b t c"]:
    """HF `causal_conv1d_fn` (bias-free): per-channel causal cross-correlation over time
    (left-padded k−1) followed by silu."""
    kernel = weight.shape[1]
    t = x.shape[1]
    padded = jnp.pad(x, ((0, 0), (kernel - 1, 0), (0, 0)))
    out = sum(padded[:, j : j + t, :] * weight[:, j] for j in range(kernel))
    return jax.nn.silu(out)


GatedDeltaKernel = Literal["sequential", "chunkwise"]
"""The closed gated-delta-rule arms: `sequential` is the per-token scan (the everywhere-
correct oracle); `chunkwise` computes the identical algebra chunk-closed-form, sequential
only across chunk boundaries."""

GATED_DELTA_CHUNK = 64
"""Chunk length of the chunkwise arm (HF's constant): the recurrent dependency chain
shrinks from T scan steps to T/chunk, bought with chunk² intra-chunk einsums. Safe at 64
only because the triangular system is solved by block substitution — see
`_solve_unit_lower_triangular` for why the log-depth inverse constructions are not."""


def gated_delta_rule(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h"],
    beta: Float[Array, "b t h"],
    kernel: GatedDeltaKernel,
) -> Float[Array, "b t h dv"]:
    """The gated delta rule, fp32 (`use_qk_l2norm_in_kernel` convention):

        S_t = exp(g_t)·S_{t−1};  S_t += k_t ⊗ β_t·(v_t − S_tᵀk_t);  o_t = S_tᵀ(q_t/√dk)

    q/k arrive already repeated to the value-head count; l2norm runs in the input dtype
    (mirroring HF), everything after in fp32, output cast back."""
    in_dtype = q.dtype
    q = l2norm(q)
    k = l2norm(k)
    q = q.astype(jnp.float32) * q.shape[-1] ** -0.5
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)
    initial_state = _initial_state(v, q.shape[-1])
    match kernel:
        case "sequential":
            out, _ = _gated_delta_rule_sequential(q, k, v, g, beta, initial_state)
        case "chunkwise":
            out, _ = _gated_delta_rule_chunkwise(q, k, v, g, beta, initial_state, GATED_DELTA_CHUNK)
    return out.astype(in_dtype)


def _initial_state(v: Float[Array, "b t h dv"], dk: int) -> Float[Array, "b h dk dv"]:
    """Zero fp32 state. On an explicit mesh it follows v's (batch, head) sharding — the
    scan carry must enter with the type its body maintains, and a fresh zeros would
    otherwise type replicated. Layout only; the recurrence is unchanged."""
    b, _t, h, dv = v.shape
    v_sharding = jax.typeof(v).sharding
    if v_sharding.mesh.empty:
        return jnp.zeros((b, h, dk, dv), jnp.float32)
    spec = v_sharding.spec
    return jnp.zeros(
        (b, h, dk, dv),
        jnp.float32,
        out_sharding=jax.NamedSharding(v_sharding.mesh, jax.P(spec[0], spec[2], None, None)),
    )


def _gated_delta_rule_sequential(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h"],
    beta: Float[Array, "b t h"],
    state: Float[Array, "b h dk dv"],
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b h dk dv"]]:
    """One scan step per token, mirroring HF's `torch_recurrent_gated_delta_rule`."""

    def step(
        state: Float[Array, "b h dk dv"], inputs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        q_t, k_t, v_t, g_t, beta_t = inputs
        state = state * jnp.exp(g_t)[..., None, None]
        kv_mem = jnp.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv_mem) * beta_t[..., None]
        state = state + jnp.einsum("bhk,bhv->bhkv", k_t, delta)
        return state, jnp.einsum("bhkv,bhk->bhv", state, q_t)

    time_leading = tuple(x.transpose(1, 0, *range(2, x.ndim)) for x in (q, k, v, g, beta))
    state, out = jax.lax.scan(step, state, time_leading)
    return out.transpose(1, 0, 2, 3), state


def _unit_lower_triangular_inverse(a_strict: Float[Array, "... c c"]) -> Float[Array, "... c c"]:
    """(I + A)⁻¹ for strictly-lower-triangular A, by Newton iteration X ← X(2I − MX):
    the residual I − MX starts at −A and SQUARES each step, and A is nilpotent (Aᶜ = 0),
    so ⌈log₂ c⌉ − 1 iterations from X₀ = I − A are exact — pure batched matmuls, no
    triangular-solve primitive (differentiable and explicit-sharding-transparent;
    `jax.scipy.linalg.solve_triangular` has no explicit-mesh sharding rule).

    TINY c only (the solver's diagonal blocks): the intermediate X_k are the partial
    Neumann sums Σ_{j<2^k}(−A)ʲ, whose entries grow binomially in c when A's entries
    approach ±1 — at c = 8 they stay ≤ ~10², at c = 64 they reach ~1e17 and fp32 loses
    the cancellation even where the exact inverse is O(1)."""
    c = a_strict.shape[-1]
    eye = jnp.eye(c, dtype=a_strict.dtype)
    m = eye + a_strict
    x = eye - a_strict
    for _ in range(max(math.ceil(math.log2(c)) - 1, 0)):
        x = x @ (2.0 * eye - m @ x)
    return x


_SOLVE_BLOCK = 8
"""Diagonal-block size of the substitution solve: large enough that the block count
stays a handful of einsums, small enough that the block inverse cannot amplify
(worst-case ‖(I + A_bb)⁻¹‖ < 2^{block−1} = 128 in fp32's 2²⁴ of headroom)."""


def _solve_unit_lower_triangular(
    a_strict: Float[Array, "... c c"], rhs: Float[Array, "... c r"]
) -> Float[Array, "... c r"]:
    """x with (I + A)x = rhs, A strictly lower triangular, by BLOCK forward
    substitution. Substitution is the one stable construction here: its partial results
    are prefixes of x itself — in this kernel, physically bounded recurrence values —
    whereas any log-depth inverse (Newton, the Π(I + A^(2ⁱ)) doubling product)
    materializes partial Neumann sums that grow binomially as A's entries approach ±1,
    the regime the masked passes' adversarially perturbed activations reach: fp32 can
    lose the cancellation and produce non-finite values at practical chunk sizes."""
    c = a_strict.shape[-1]
    block = min(_SOLVE_BLOCK, c)
    assert c % block == 0, (c, block)
    n_blocks = c // block
    diagonal_blocks = jnp.stack(
        [
            a_strict[..., i * block : (i + 1) * block, i * block : (i + 1) * block]
            for i in range(n_blocks)
        ],
        axis=-3,
    )
    diagonal_inverse = _unit_lower_triangular_inverse(diagonal_blocks)
    xs: list[Array] = []
    for i in range(n_blocks):
        acc = rhs[..., i * block : (i + 1) * block, :]
        for j, x_j in enumerate(xs):
            acc = (
                acc - a_strict[..., i * block : (i + 1) * block, j * block : (j + 1) * block] @ x_j
            )
        xs.append(diagonal_inverse[..., i, :, :] @ acc)
    return jnp.concatenate(xs, axis=-2)


def _gated_delta_rule_chunkwise(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h"],
    beta: Float[Array, "b t h"],
    state: Float[Array, "b h dk dv"],
    chunk: int,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b h dk dv"]]:
    """The WY/Householder-product form of the same recurrence, mirroring HF's
    `torch_chunk_gated_delta_rule` (with its C-step substitution loop blocked, and the
    inverse never materialized). Unrolling the per-token rule within a chunk, with
    G_i = Σ_{j≤i} g_j the inclusive in-chunk cumulative log-decay and S₀ the chunk-entry
    state, the per-token updates δ_i = β_i(v_i − exp(g_i)·…ᵀk_i) satisfy the
    unit-lower-triangular system

        (I + A) δ = βv − (β·exp(G)·k) S₀,   A[i,j] = β_i·exp(G_i − G_j)·(k_i·k_j), j < i

    solved once per chunk against the state-independent right-hand sides w := T·βv and
    u := T·(β·exp(G)·k) (both physically bounded: δ at zero entry state, and the
    entry-state read coefficients), so the scan body is einsums only:

        δ = w − u·S₀
        o_i = exp(G_i)·S₀ᵀq_i + Σ_{j≤i} exp(G_i − G_j)·(q_i·k_j)·δ_j
        S_C = exp(G_C)·S₀ + Σ_j exp(G_C − G_j)·k_j ⊗ δ_j

    T pads up to a chunk multiple with zeros: padded tokens carry g = 0 (no decay) and
    β = 0 (δ = 0), so they touch neither the real outputs nor the final state."""
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    n = -(-t // chunk)
    pad = n * chunk - t
    if pad:
        q, k, v = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (q, k, v))
        g, beta = (jnp.pad(x, ((0, 0), (0, pad), (0, 0))) for x in (g, beta))

    def chunked(x: Array) -> Array:
        parts = x.reshape(b, n, chunk, h, *x.shape[3:])
        return parts.transpose(1, 0, 3, 2, *range(4, parts.ndim))

    # (n b h chunk d) — the scan consumes one chunk per step.
    q, k, v, g, beta = map(chunked, (q, k, v, g, beta))
    g_cum = jnp.cumsum(g, axis=-1)
    idx = jnp.arange(chunk)
    lower = idx[:, None] >= idx[None, :]
    strictly_lower = idx[:, None] > idx[None, :]
    # decay[i,j] = exp(G_i − G_j) on j ≤ i, 0 above — masked before AND after the exp so
    # the dead upper triangle never overflows (G_i − G_j > 0 there).
    diff = g_cum[..., :, None] - g_cum[..., None, :]
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, diff, 0.0)), 0.0)
    k_beta = k * beta[..., None]
    a = jnp.where(strictly_lower, jnp.einsum("...ik,...jk->...ij", k_beta, k) * decay, 0.0)
    rhs = jnp.concatenate([v * beta[..., None], k_beta * jnp.exp(g_cum)[..., None]], axis=-1)
    solved = _solve_unit_lower_triangular(a, rhs)
    w, u = solved[..., :dv], solved[..., dv:]
    q_decay = q * jnp.exp(g_cum)[..., None]
    local = jnp.einsum("...ik,...jk->...ij", q, k) * decay
    g_last = g_cum[..., -1]
    k_out_decay = k * jnp.exp(g_last[..., None] - g_cum)[..., None]

    def chunk_step(
        state: Float[Array, "b h dk dv"],
        inputs: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array]:
        q_dec, w_c, u_c, local_c, k_dec, g_l = inputs
        delta = w_c - jnp.einsum("bhck,bhkv->bhcv", u_c, state)
        out = jnp.einsum("bhck,bhkv->bhcv", q_dec, state) + jnp.einsum(
            "bhij,bhjv->bhiv", local_c, delta
        )
        state = state * jnp.exp(g_l)[..., None, None] + jnp.einsum("bhck,bhcv->bhkv", k_dec, delta)
        return state, out

    state, out = jax.lax.scan(chunk_step, state, (q_decay, w, u, local, k_out_decay, g_last))
    return out.transpose(1, 0, 3, 2, 4).reshape(b, n * chunk, h, dv)[:, :t], state


def apply_partial_rope(
    q: Float[Array, "b t h hd"],
    k: Float[Array, "b t kvh hd"],
    cos: Float[Array, "t r"],
    sin: Float[Array, "t r"],
) -> tuple[Array, Array]:
    """RoPE on the FIRST `r` dims of each head (partial_rotary_factor), rotate_half
    convention, pass-through on the rest — heads on axis −2 (no transposes)."""
    rotary_dim = cos.shape[-1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def rotate(x: Array) -> Array:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        return jnp.concatenate([x_rot * cos + rotate_half(x_rot) * sin, x_pass], axis=-1)

    return rotate(q), rotate(k)

"""Scaled dot-product attention with explicit backend and batch/head sharding."""

import math
from functools import partial
from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jax.extend.backend import get_default_device
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float

from param_decomp.sequence import SequenceLayout

AttentionImplementation = Literal["flash", "mosaic", "xla"]
"""`flash` is cuDNN's fused kernel; `mosaic` is tokamax's Pallas Mosaic GPU flash kernel
(Hopper and Blackwell), which also takes head dims cuDNN rejects (not a multiple of 8);
`xla` materializes the scores, and is the one that takes a logit softcap. No choice falls
back to another."""


def jax_attention_implementation(
    requested: AttentionImplementation, backend: str, dtype: jnp.dtype
) -> Literal["cudnn", "mosaic", "xla"]:
    """Translate the requested algorithm to the kernel that runs it, without fallback."""
    assert requested in get_args(AttentionImplementation), (
        f"attention implementation must explicitly select one of "
        f"{get_args(AttentionImplementation)}, got {requested!r}"
    )
    match requested:
        case "flash":
            _assert_flash_operands("cuDNN", backend, dtype)
            return "cudnn"
        case "mosaic":
            _assert_flash_operands("tokamax Mosaic GPU", backend, dtype)
            return "mosaic"
        case "xla":
            return "xla"


def _assert_flash_operands(kernel: str, backend: str, dtype: jnp.dtype) -> None:
    assert backend == "gpu" and dtype in (jnp.float16, jnp.bfloat16), (
        f"{kernel} flash attention requires a GPU and float16/bfloat16 operands; "
        f"got backend={backend!r}, dtype={dtype}. "
        "Select 'xla' explicitly to allow non-flash attention."
    )


def placed_dot_product_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    implementation: AttentionImplementation,
    qkv_sharding: NamedSharding | None,
    logit_softcap: float | None,
) -> Float[Array, "b t qh hd"]:
    """`logit_softcap`, when set, softmaxes `cap · tanh(s / cap)` of each logit
    `s = q·k / √hd`; `None` softmaxes the logits as they are. Only the materialized XLA path
    applies a cap (cuDNN's fused kernel has no logit hook), so it needs `implementation="xla"`."""
    backend = jax_attention_implementation(implementation, get_default_device().platform, q.dtype)
    if logit_softcap is None:
        attention = partial(_uncapped_attention, is_causal=is_causal, backend=backend)
    else:
        assert backend == "xla", (
            f"a logit softcap needs implementation='xla', got {implementation!r}"
        )
        attention = partial(_softcapped_attention, is_causal=is_causal, cap=logit_softcap)

    if qkv_sharding is None:
        return attention(q, k, v, mask)
    spec = qkv_sharding.spec
    assert len(spec) == 4 and spec[1] is None and spec[3] is None, (
        f"Attention may shard batch and heads only, got {spec}"
    )
    match implementation:
        case "flash" | "mosaic":
            # The saved output and softmax statistics must share Q's local batch.
            # Enclose both directions of the kernel's custom VJP in the same shard map.
            # tokamax's Pallas outputs declare no varying mesh axes (vma), which the
            # checker refuses inside a manual map; the specs here are the contract.
            return jax.shard_map(
                attention,
                mesh=qkv_sharding.mesh,
                in_specs=(spec, spec, spec, P(spec[0], None, None, None)),
                out_specs=spec,
                check_vma=implementation == "flash",
            )(q, k, v, mask)
        case "xla":
            return jax.sharding.auto_axes(out_sharding=qkv_sharding)(attention)(q, k, v, mask)


def _uncapped_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    backend: Literal["cudnn", "mosaic", "xla"],
) -> Float[Array, "b t qh hd"]:
    match backend:
        case "cudnn" | "xla":
            return jax.nn.dot_product_attention(
                q, k, v, mask=mask, is_causal=is_causal, implementation=backend
            )
        case "mosaic":
            # Imported at trace time: tokamax must import after the launch env and
            # before pyarrow (see `experiments/lm/run.py`).
            import tokamax

            return tokamax.dot_product_attention(
                q, k, v, mask=mask, is_causal=is_causal, implementation="mosaic"
            )


def _softcapped_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    cap: float,
) -> Float[Array, "b t qh hd"]:
    """Softmax attention over `cap · tanh(q·k / √hd / cap)`, its logits and softmax in fp32."""
    assert cap > 0, cap
    group = q.shape[2] // k.shape[2]
    k, v = jnp.repeat(k, group, axis=2), jnp.repeat(v, group, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
    logits = cap * jnp.tanh(logits / (math.sqrt(q.shape[-1]) * cap))
    if is_causal:
        mask = mask & jnp.tril(jnp.ones((q.shape[1], k.shape[1]), bool))[None, None]
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def causal_attention_head_first(
    q: Float[Array, "b qh t hd"],
    k: Float[Array, "b kvh t hd"],
    v: Float[Array, "b kvh t hd"],
    sequence: SequenceLayout,
    qkv_sharding: NamedSharding | None,
    implementation: AttentionImplementation,
) -> Float[Array, "b qh t hd"]:
    """Document-isolated causal attention with heads before tokens.

    ``qkv_sharding`` describes the token-before-head layout used by the kernel.
    """
    qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    out = placed_dot_product_attention(
        qt,
        kt,
        vt,
        sequence.attention_mask()[:, None, :, :],
        is_causal=True,
        implementation=implementation,
        qkv_sharding=qkv_sharding,
        logit_softcap=None,
    )
    return out.transpose(0, 2, 1, 3)

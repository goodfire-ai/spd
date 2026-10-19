"""`attention_pattern_kl` against the KL between explicit softmax patterns.

Flash layers' scores run the Triton kernels, through the Pallas interpreter on CPU, and XLA
layers' form both patterns. Both are checked with grouped-query heads, document boundaries
inside rows and several query/key tiles per row, under three pattern definitions: plain attention,
a logit softcap and a sliding window. On a GPU the kernels run compiled in bf16.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.attention import AttentionImplementation
from param_decomp.attention_kl import (
    AttentionPatternDefinition,
    AttentionScores,
    attention_pattern_kl,
)

B, T, Q_HEADS, KV_HEADS, HEAD_DIM = 2, 192, 4, 2, 16
DOCUMENT_IDS = jnp.array([[0] * 80 + [1] * 112, [5] * T], jnp.int32)
"""Row 0 has a document boundary inside its second 64-token tile."""


@dataclass(frozen=True)
class _Documents:
    scale: float

    def logits(self, dots: Array, _q_index: Array, _k_index: Array, /) -> Array:
        return dots * self.scale

    def attends(
        self, _q_index: Array, _k_index: Array, q_document: Array, k_document: Array, /
    ) -> Array:
        return q_document[:, None] == k_document[None, :]


@dataclass(frozen=True)
class _Softcapped(_Documents):
    cap: float

    @override
    def logits(self, dots: Array, _q_index: Array, _k_index: Array, /) -> Array:
        return self.cap * jnp.tanh(dots * self.scale / self.cap)


@dataclass(frozen=True)
class _SlidingWindow(_Documents):
    window: int

    @override
    def attends(
        self, q_index: Array, k_index: Array, q_document: Array, k_document: Array, /
    ) -> Array:
        recent = q_index[:, None] - k_index[None, :] < self.window
        return super().attends(q_index, k_index, q_document, k_document) & recent


PLAIN = _Documents(HEAD_DIM**-0.5)
SOFTCAPPED = _Softcapped(HEAD_DIM**-0.5, cap=1.5)
WINDOWED = _SlidingWindow(HEAD_DIM**-0.5, window=20)


def _explicit_kl(
    q: Array,
    k: Array,
    masked_q: Array,
    masked_k: Array,
    document_ids: Array,
    *,
    softcap: float | None = None,
    window: int | None = None,
) -> Array:
    """Per-row `KL(softmax(s) ‖ softmax(s'))` over causal, same-document keys (within
    `window` of the query when given), with logits `q·kᵀ/√d` (soft-capped when given), in
    FP32 over whole `[t, t]` score arrays."""
    seq_len, head_dim = q.shape[1], q.shape[3]
    attends = (document_ids[:, :, None] == document_ids[:, None, :]) & jnp.tril(
        jnp.ones((seq_len, seq_len), bool)
    )
    if window is not None:
        attends &= jnp.triu(jnp.ones((seq_len, seq_len), bool), -(window - 1))

    def log_pattern(q: Array, k: Array) -> Array:
        k = jnp.repeat(k.astype(jnp.float32), q.shape[2] // k.shape[2], axis=2)
        s = jnp.einsum("bthd,bshd->bhts", q.astype(jnp.float32), k, precision="highest")
        s = s / head_dim**0.5
        if softcap is not None:
            s = softcap * jnp.tanh(s / softcap)
        return jax.nn.log_softmax(jnp.where(attends[:, None], s, -jnp.inf), axis=-1)

    log_p = jax.lax.stop_gradient(log_pattern(q, k))
    log_q = log_pattern(masked_q, masked_k)
    return jnp.sum(jnp.where(attends[:, None], jnp.exp(log_p) * (log_p - log_q), 0.0), -1)


def _inputs(perturbation: float) -> tuple[Array, Array, Array, Array]:
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (B, T, Q_HEADS, HEAD_DIM))
    k = jax.random.normal(keys[1], (B, T, KV_HEADS, HEAD_DIM))
    masked_q = q + perturbation * jax.random.normal(keys[2], q.shape)
    masked_k = k + perturbation * jax.random.normal(keys[3], k.shape)
    return q, k, masked_q, masked_k


def _kl(
    implementation: AttentionImplementation,
    definition: AttentionPatternDefinition,
    document_ids: Array = DOCUMENT_IDS,
) -> Callable[[Array, Array, Array, Array], Array]:
    """`attention_pattern_kl` between two layers' scores, as a function of their q and k."""

    def kl(clean_q: Array, clean_k: Array, masked_q: Array, masked_k: Array) -> Array:
        return attention_pattern_kl(
            AttentionScores(clean_q, clean_k, document_ids, definition, implementation),
            AttentionScores(masked_q, masked_k, document_ids, definition, implementation),
        )

    return kl


@pytest.mark.parametrize("implementation", ["flash", "xla"])
@pytest.mark.parametrize(
    ("definition", "softcap", "window"),
    [(PLAIN, None, None), (SOFTCAPPED, 1.5, None), (WINDOWED, None, 20)],
    ids=["plain", "softcap", "sliding_window"],
)
@pytest.mark.parametrize("perturbation", [0.05, 1.0])
def test_attention_pattern_kl_matches_explicit_patterns(
    perturbation: float,
    definition: AttentionPatternDefinition,
    softcap: float | None,
    window: int | None,
    implementation: AttentionImplementation,
):
    q, k, masked_q, masked_k = _inputs(perturbation)
    kl = _kl(implementation, definition)

    def explicit(mq: Array, mk: Array) -> Array:
        return _explicit_kl(q, k, mq, mk, DOCUMENT_IDS, softcap=softcap, window=window)

    with jax.default_matmul_precision("highest"):
        rows = kl(q, k, masked_q, masked_k)
        assert rows.shape == (B, Q_HEADS, T) and rows.dtype == jnp.float32
        # A row's KL is a difference of log-normalizers near 10, so FP32 bounds its absolute
        # error near 1e-6 however it is computed; compiled kernels reach that bound.
        np.testing.assert_allclose(rows, explicit(masked_q, masked_k), rtol=1e-4, atol=5e-6)

        weights = jax.random.uniform(jax.random.key(1), rows.shape)

        def loss(mq: Array, mk: Array) -> Array:
            return jnp.sum(weights * kl(q, k, mq, mk))

        def explicit_loss(mq: Array, mk: Array) -> Array:
            return jnp.sum(weights * explicit(mq, mk))

        got_grads = jax.grad(loss, argnums=(0, 1))(masked_q, masked_k)
        want_grads = jax.grad(explicit_loss, argnums=(0, 1))(masked_q, masked_k)
    for got, want_grad in zip(got_grads, want_grads, strict=True):
        np.testing.assert_allclose(got, want_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("implementation", ["flash", "xla"])
def test_attention_pattern_kl_is_zero_for_identical_patterns_and_ignores_the_clean_side(
    implementation: AttentionImplementation,
):
    q, k, _, _ = _inputs(0.0)
    kl = _kl(implementation, PLAIN)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(kl(q, k, q, k), 0.0, atol=1e-5)

        def loss(clean_q: Array, clean_k: Array) -> Array:
            return jnp.sum(kl(clean_q, clean_k, q, k))

        clean_grads = jax.grad(loss, argnums=(0, 1))(q, k)
    for grad in clean_grads:
        np.testing.assert_array_equal(grad, 0.0)


def test_attention_pattern_kl_refuses_scores_of_different_layers():
    q, k, _, _ = _inputs(0.0)
    with pytest.raises(AssertionError):
        attention_pattern_kl(
            AttentionScores(q, k, DOCUMENT_IDS, PLAIN, "xla"),
            AttentionScores(q, k, DOCUMENT_IDS, WINDOWED, "xla"),
        )
    with pytest.raises(AssertionError):
        attention_pattern_kl(
            AttentionScores(q, k, DOCUMENT_IDS, PLAIN, "xla"),
            AttentionScores(q, k, DOCUMENT_IDS, PLAIN, "flash"),
        )


def test_flash_scores_run_the_kernels_on_each_devices_rows():
    """Batch-split scores give the unsplit value and gradients."""
    q, k, masked_q, masked_k = _inputs(0.5)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    tokens = NamedSharding(mesh, P("data", None, None, None))
    placed_q, placed_k = jax.device_put((q, k), tokens)
    documents = jax.device_put(DOCUMENT_IDS, NamedSharding(mesh, P("data", None)))
    placed = _kl("flash", PLAIN, documents)
    unplaced = _kl("flash", PLAIN)

    def loss(mq: Array, mk: Array) -> Array:
        return jnp.sum(placed(placed_q, placed_k, mq, mk))

    def unplaced_loss(mq: Array, mk: Array) -> Array:
        return jnp.sum(unplaced(q, k, mq, mk))

    with jax.default_matmul_precision("highest"):
        want_rows = unplaced(q, k, masked_q, masked_k)
        want = jax.grad(unplaced_loss, argnums=(0, 1))(masked_q, masked_k)
        with jax.set_mesh(mesh):
            placed_masked = jax.device_put((masked_q, masked_k), tokens)
            rows = jax.jit(lambda mq, mk: placed(placed_q, placed_k, mq, mk))(*placed_masked)
            got = jax.jit(jax.grad(loss, argnums=(0, 1)))(*placed_masked)
        np.testing.assert_allclose(rows, want_rows, rtol=1e-6)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-7)


def _ragged_documents(seq_len: int) -> Array:
    """Row 0: boundaries off the tile grid, a one-token document and right padding; row 1:
    one document."""
    row = np.zeros(seq_len, np.int32)
    for document, start in enumerate((37, 38, 200, 333)):
        row[start:] = document + 1
    row[seq_len - 50 :] = -1
    return jnp.asarray(np.stack([row, np.zeros(seq_len, np.int32)]))


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a GPU for compiled kernels")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_compiled_kernels_match_explicit_patterns_in_bf16(head_dim: int):
    q_heads, kv_heads, seq_len = 16, 8, 512
    keys = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(keys[0], (2, seq_len, q_heads, head_dim), jnp.bfloat16)
    k = jax.random.normal(keys[1], (2, seq_len, kv_heads, head_dim), jnp.bfloat16)
    masked_q = (q + 0.5 * jax.random.normal(keys[2], q.shape)).astype(jnp.bfloat16)
    masked_k = (k + 0.5 * jax.random.normal(keys[3], k.shape)).astype(jnp.bfloat16)
    document_ids = _ragged_documents(seq_len)
    weights = jax.random.uniform(keys[4], (2, q_heads, seq_len))
    kl = _kl("flash", _Documents(head_dim**-0.5), document_ids)

    def fused(mq: Array, mk: Array) -> Array:
        return kl(q, k, mq, mk)

    def explicit(mq: Array, mk: Array) -> Array:
        return _explicit_kl(q, k, mq, mk, document_ids)

    def weighted(rows: Callable[[Array, Array], Array]) -> Callable[[Array, Array], Array]:
        return lambda mq, mk: jnp.sum(weights * rows(mq, mk))

    rows = jax.jit(fused)(masked_q, masked_k)
    want_rows = jax.jit(explicit)(masked_q, masked_k)
    assert rows.shape == (2, q_heads, seq_len) and rows.dtype == jnp.float32
    np.testing.assert_allclose(rows, want_rows, rtol=1e-3, atol=1e-4)
    got = jax.jit(jax.grad(weighted(fused), argnums=(0, 1)))(masked_q, masked_k)
    want = jax.jit(jax.grad(weighted(explicit), argnums=(0, 1)))(masked_q, masked_k)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == jnp.bfloat16
        g, w = g.astype(jnp.float32).ravel(), w.astype(jnp.float32).ravel()
        assert float(jnp.dot(g, w) / (jnp.linalg.norm(g) * jnp.linalg.norm(w))) > 0.99999
        assert float(jnp.linalg.norm(g - w) / jnp.linalg.norm(w)) < 0.01

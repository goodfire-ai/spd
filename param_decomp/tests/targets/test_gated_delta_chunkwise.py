"""Chunkwise-vs-sequential parity for the gated delta rule.

The chunkwise WY form is the wired production arm; the per-token scan is the oracle
(parity-pinned against HF). Both arms run fp32 internally, so they agree to
reassociation level — tolerances here carry roughly two orders of magnitude of headroom
over the observed maxima. Gradients matter even though the mixer is frozen: masked
passes differentiate THROUGH the recurrence to upstream activations."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.target_ports.qwen3_5_moe import (
    GATED_DELTA_CHUNK,
    GatedDeltaKernel,
    _gated_delta_rule_chunkwise,
    _gated_delta_rule_sequential,
    gated_delta_rule,
    l2norm,
)

H, DK, DV = 4, 16, 24


Inputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _arm_inputs(key: jax.Array, b: int, t: int) -> Inputs:
    """fp32 inputs in the convention the arms receive: l2-normed q/k (the delta rule is
    contractive only there — β·‖k‖² ≤ 2), q pre-scaled, g a negative log-decay, β a
    sigmoid write strength."""
    ks = jax.random.split(key, 5)
    q = l2norm(jax.random.normal(ks[0], (b, t, H, DK))) * DK**-0.5
    k = l2norm(jax.random.normal(ks[1], (b, t, H, DK)))
    v = jax.random.normal(ks[2], (b, t, H, DV))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, H)))
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, H)))
    return q, k, v, g, beta


@pytest.mark.parametrize(
    ("b", "t", "chunk"),
    [
        (2, 128, GATED_DELTA_CHUNK),  # exact multiple
        (2, 129, GATED_DELTA_CHUNK),  # one token past a boundary
        (1, 17, GATED_DELTA_CHUNK),  # t < chunk: a single padded chunk
        (1, 1, GATED_DELTA_CHUNK),  # degenerate sequence
        (2, 100, 16),  # partial final chunk, non-default chunk
        (3, 7, 1),  # chunk=1 degenerates to the per-token rule
    ],
)
def test_chunkwise_matches_sequential_outputs_and_state(b: int, t: int, chunk: int):
    q, k, v, g, beta = _arm_inputs(jax.random.PRNGKey(0), b, t)
    state0 = jnp.zeros((b, H, DK, DV), jnp.float32)
    out_seq, state_seq = _gated_delta_rule_sequential(q, k, v, g, beta, state0)
    out_chunk, state_chunk = _gated_delta_rule_chunkwise(q, k, v, g, beta, state0, chunk)
    np.testing.assert_allclose(out_chunk, out_seq, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_chunk, state_seq, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_public_kernels_agree(dtype: jax.typing.DTypeLike):
    ks = jax.random.split(jax.random.PRNGKey(1), 5)
    b, t = 2, 100
    q = jax.random.normal(ks[0], (b, t, H, DK), dtype)
    k = jax.random.normal(ks[1], (b, t, H, DK), dtype)
    v = jax.random.normal(ks[2], (b, t, H, DV), dtype)
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, H), dtype))
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, H), dtype))
    out_seq = gated_delta_rule(q, k, v, g, beta, "sequential").astype(jnp.float32)
    out_chunk = gated_delta_rule(q, k, v, g, beta, "chunkwise").astype(jnp.float32)
    # Both arms compute fp32 internally, so bf16 outputs differ only where reassociation
    # noise crosses a rounding boundary — one output ulp.
    rtol = 1e-5 if dtype == jnp.float32 else 2**-7
    np.testing.assert_allclose(out_chunk, out_seq, rtol=rtol, atol=1e-5)


def _identical_keys(b: int, t: int) -> jax.Array:
    return l2norm(jnp.broadcast_to(jnp.ones((DK,), jnp.float32) / DK**0.5, (b, t, H, DK)))


def _clustered_signed_keys(b: int, t: int) -> jax.Array:
    """A few shared directions with random signs: drives A's entries to ±1 in mixed
    sign patterns — the structure (unlike aligned keys) where the exact triangular
    inverse itself grows and log-depth inverse constructions go to garbage first."""
    directions = jax.random.normal(jax.random.PRNGKey(7), (4, DK))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    picks = jax.random.randint(jax.random.PRNGKey(8), (t,), 0, 4)
    signs = jnp.where(jax.random.bernoulli(jax.random.PRNGKey(9), 0.5, (t,)), 1.0, -1.0)
    return l2norm(
        jnp.broadcast_to((directions[picks] * signs[:, None])[None, :, None, :], (b, t, H, DK))
    )


@pytest.mark.parametrize("keys", [_identical_keys, _clustered_signed_keys])
def test_adversarial_worst_case_stays_finite_and_close(keys: Callable[[int, int], jax.Array]):
    """β = 1, no decay, keys driving A's entries to ±1 — the degenerate regime the
    masked passes' adversarially perturbed activations approach. Log-depth inverse
    constructions can become non-finite here; blocked substitution stays at oracle-level
    error because its partials are bounded physical values."""
    b, t, dv = 1, 128, 16
    k = keys(b, t)
    q = jnp.broadcast_to(jnp.linspace(-1.0, 1.0, DK, dtype=jnp.float32), (b, t, H, DK))
    v = jax.random.normal(jax.random.PRNGKey(5), (b, t, H, dv), jnp.float32)
    g = jnp.zeros((b, t, H), jnp.float32)
    beta = jnp.ones((b, t, H), jnp.float32)
    state0 = jnp.zeros((b, H, DK, dv), jnp.float32)
    out_seq, state_seq = _gated_delta_rule_sequential(q, k, v, g, beta, state0)
    out_chunk, state_chunk = _gated_delta_rule_chunkwise(
        q, k, v, g, beta, state0, GATED_DELTA_CHUNK
    )
    assert bool(jnp.isfinite(out_chunk).all())
    np.testing.assert_allclose(out_chunk, out_seq, rtol=0, atol=2e-5)
    np.testing.assert_allclose(state_chunk, state_seq, rtol=0, atol=2e-5)


def test_gradients_finite_at_production_decay_magnitudes():
    """Strong log-decays (|g| ~ 40/step) push the decay matrix's upper triangle far
    beyond fp32's exp range. The forward masks it either way; the trap is the BACKWARD
    — an unmasked exp argument leaves 0 × inf = NaN in g's gradient, which no
    forward parity test can see."""
    b, t = 2, 100
    q, k, v, _, beta = _arm_inputs(jax.random.PRNGKey(4), b, t)
    g = -40.0 * (1.0 + jax.random.uniform(jax.random.PRNGKey(6), (b, t, H)))
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (b, t, H, DV))

    def loss(inputs: Inputs, kernel: GatedDeltaKernel) -> jax.Array:
        return jnp.sum(gated_delta_rule(*inputs, kernel) * cotangent)

    for kernel in ("sequential", "chunkwise"):
        grads = jax.grad(loss)((q, k, v, g, beta), kernel)
        for name, grad in zip(("q", "k", "v", "g", "beta"), grads, strict=True):
            assert bool(jnp.isfinite(grad).all()), (kernel, name)


def test_gradients_flow_identically_through_both_kernels():
    b, t = 2, 100
    inputs = _arm_inputs(jax.random.PRNGKey(2), b, t)
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (b, t, H, DV))

    def loss(inputs: Inputs, kernel: GatedDeltaKernel) -> jax.Array:
        return jnp.sum(gated_delta_rule(*inputs, kernel) * cotangent)

    grads_seq = jax.grad(loss)(inputs, "sequential")
    grads_chunk = jax.grad(loss)(inputs, "chunkwise")
    for name, g_s, g_c in zip(("q", "k", "v", "g", "beta"), grads_seq, grads_chunk, strict=True):
        np.testing.assert_allclose(g_c, g_s, rtol=1e-4, atol=1e-4, err_msg=name)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")
@pytest.mark.multidevice
def test_placed_remat_scan_grads_match_unplaced_and_stay_finite():
    """The kernel in its training context — under `jax.checkpoint` inside a `lax.scan`
    (the stage scan's structure), on an explicit (data, tp) mesh with batch/head-sharded
    operands — at adversarial magnitudes (clustered ±keys, β → 1, strong decays,
    T spanning a padded multi-chunk layout). This exercises the complete placed
    training context: gradients must be finite and match the unplaced execution."""
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    b, t = 4, 96
    x0 = _clustered_signed_keys(b, t) * 2.0
    g = -40.0 * (1.0 + jax.random.uniform(jax.random.PRNGKey(6), (b, t, H)))
    beta = jax.nn.sigmoid(4.0 + jax.random.normal(jax.random.PRNGKey(10), (b, t, H)))
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (b, t, H, DK))

    def loss(x0: jax.Array, g: jax.Array, beta: jax.Array, cotangent: jax.Array) -> jax.Array:
        def body(x: jax.Array, _: None) -> tuple[jax.Array, None]:
            def sublayer(x: jax.Array) -> jax.Array:
                return x + gated_delta_rule(x, x, x, g, beta, "chunkwise")

            return jax.checkpoint(sublayer)(x), None

        x_final, _ = jax.lax.scan(body, x0, None, length=3)
        return jnp.sum(x_final * cotangent)

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    expected = grad_fn(x0, g, beta, cotangent)

    mesh = Mesh(
        np.asarray(jax.devices()[:8]).reshape(4, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    wide = NamedSharding(mesh, P("data", None, "tp", None))
    slim = NamedSharding(mesh, P("data", None, "tp"))
    # the production contract: placed forwards execute under the ambient mesh (without
    # it, an in-scan `out_sharding` silently types replicated and the kernel's carry
    # einsums fail to type-check).
    with jax.set_mesh(mesh):
        got = grad_fn(
            jax.device_put(x0, wide),
            jax.device_put(g, slim),
            jax.device_put(beta, slim),
            jax.device_put(cotangent, wide),
        )
    for name, got_leaf, expected_leaf in zip(("x0", "g", "beta"), got, expected, strict=True):
        assert bool(jnp.isfinite(got_leaf).all()), name
        np.testing.assert_allclose(got_leaf, expected_leaf, rtol=1e-4, atol=1e-4, err_msg=name)

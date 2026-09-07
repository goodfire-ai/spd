"""The streamed model-output edge: `StreamedLinearOutput` + the `targets.losses`
streamed kernels against their materialized twins.

Parity is checked at both scales the edge serves: a tiny vocab (every chunking, both
frozen-weight dtypes, gradients) and the production 248,320-entry vocab (bf16 inputs,
fp32 accumulators — the reassociation regime the large-capacity reference uses). The kernels
fp32-accumulate each chunk's logits from the native-dtype operands, so the kernel-level
references form the full logits the same way (`_fp32_logits`) and the two spellings
differ only by fp32 reassociation in the softmax reductions; gradients differ by bf16
ulps through the shared matmul transpose. The model-level block pins the qwen36_moe
edge flip, where the materialized edge's bf16-rounded logits are the one seam between
the edges; the CE/KL eval-tier flip lives with the eval suite
(`experiments/lm/test_eval.py`)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.typing import DTypeLike
from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.model import (
    PlacedModel,
    StochasticMasking,
    prepare_compute_weights,
)
from param_decomp.targets.lm_output import StreamedLinearOutput
from param_decomp.targets.losses import (
    kl_per_position,
    streamed_kl_per_position,
    streamed_next_token_cross_entropy,
    streamed_position_kl,
    streamed_position_next_token_ce,
)
from param_decomp.targets.qwen36_moe import (
    MaterializedOutputEdge,
    Qwen36MoeDecomposedModel,
    StreamedOutputEdge,
    full_site_cs,
    qwen36_35b_a3b_config,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)


def _fp32_logits(activations: Array, head: Array) -> Array:
    """The full logits formed as the streamed kernels form each chunk: native-dtype
    operands, fp32 accumulation."""
    return jnp.dot(activations, head.T, preferred_element_type=jnp.float32)


def _materialized_next_token_ce(logits: Array, token_ids: Array) -> Array:
    """The eval tier's `next_token_cross_entropy`, restated to keep this suite
    experiments-free (targets tests import no composition module)."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    labels = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[..., 0]
    return -labels.mean()


def _random_pair(
    key: Array, shape: tuple[int, int, int], vocab: int, dtype: DTypeLike
) -> tuple[Array, Array, Array, Array]:
    k_clean, k_masked, k_head, k_tokens = random.split(key, 4)
    batch, seq, d_model = shape
    return (
        random.normal(k_clean, (batch, seq, d_model), dtype),
        random.normal(k_masked, (batch, seq, d_model), dtype),
        random.normal(k_head, (vocab, d_model), dtype),
        random.randint(k_tokens, (batch, seq), 0, vocab),
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n_chunks", [1, 4, 64])
def test_streamed_kl_and_ce_match_materialized_tiny_vocab(dtype: DTypeLike, n_chunks: int):
    vocab = 64
    h_clean, h_masked, head, tokens = _random_pair(random.PRNGKey(0), (3, 5, 16), vocab, dtype)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=n_chunks)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=n_chunks)

    kl_expected = kl_per_position(_fp32_logits(h_masked, head), _fp32_logits(h_clean, head))
    np.testing.assert_allclose(
        np.asarray(streamed_kl_per_position(masked, clean)),
        np.asarray(kl_expected),
        rtol=1e-6,
        atol=1e-6,
    )
    ce_expected = _materialized_next_token_ce(_fp32_logits(h_masked, head), tokens)
    np.testing.assert_allclose(
        np.asarray(streamed_next_token_cross_entropy(masked, tokens)),
        np.asarray(ce_expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_streamed_gradients_match_materialized():
    vocab, n_chunks = 64, 4
    h_clean, h_masked, head, tokens = _random_pair(
        random.PRNGKey(1), (3, 5, 16), vocab, jnp.bfloat16
    )
    logits_clean = _fp32_logits(h_clean, head)

    def package(activations: Array) -> StreamedLinearOutput:
        return StreamedLinearOutput(
            activations=activations.astype(jnp.bfloat16), head=head, n_chunks=n_chunks
        )

    def assert_close_frobenius(got: Array, expected: Array) -> None:
        # both paths pull the softmax cotangent back through one bf16 matmul transpose;
        # the streamed chunking reassociates its products, so pointwise ulp bounds don't
        # hold on the smallest elements — the norm-level one does (the placed suites'
        # criterion; observed ~2e-3)
        got_np, expected_np = np.asarray(got), np.asarray(expected)
        error = np.linalg.norm(got_np - expected_np) / np.linalg.norm(expected_np)
        assert error < 1e-2, error

    clean = package(h_clean)
    grad_kl_materialized = jax.grad(
        lambda h: kl_per_position(_fp32_logits(h.astype(jnp.bfloat16), head), logits_clean)
    )(h_masked.astype(jnp.float32))
    grad_kl_streamed = jax.grad(lambda h: streamed_kl_per_position(package(h), clean))(
        h_masked.astype(jnp.float32)
    )
    assert_close_frobenius(grad_kl_streamed, grad_kl_materialized)

    grad_ce_materialized = jax.grad(
        lambda h: _materialized_next_token_ce(_fp32_logits(h.astype(jnp.bfloat16), head), tokens)
    )(h_masked.astype(jnp.float32))
    grad_ce_streamed = jax.grad(lambda h: streamed_next_token_cross_entropy(package(h), tokens))(
        h_masked.astype(jnp.float32)
    )
    assert_close_frobenius(grad_ce_streamed, grad_ce_materialized)


def test_streamed_kernels_match_at_the_production_vocab():
    """The large-vocabulary regime: the full 248,320-entry axis, bf16 logits chunks, fp32
    online accumulators — the reassociation the tiny-vocab cases cannot exercise."""
    vocab = qwen36_35b_a3b_config().vocab_size
    n_chunks = 32
    assert vocab % n_chunks == 0, (vocab, n_chunks)
    h_clean, h_masked, head, tokens = _random_pair(
        random.PRNGKey(2), (2, 4, 64), vocab, jnp.bfloat16
    )
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=n_chunks)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=n_chunks)

    kl_expected = kl_per_position(_fp32_logits(h_masked, head), _fp32_logits(h_clean, head))
    np.testing.assert_allclose(
        np.asarray(streamed_kl_per_position(masked, clean)),
        np.asarray(kl_expected),
        rtol=1e-5,
        atol=1e-5,
    )
    ce_expected = _materialized_next_token_ce(_fp32_logits(h_masked, head), tokens)
    np.testing.assert_allclose(
        np.asarray(streamed_next_token_cross_entropy(masked, tokens)),
        np.asarray(ce_expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_per_position_kernels_reduce_like_their_means():
    """The row-masked eval consumers compose the per-position kernels with their own
    means; the scalar kernels must be exactly those means."""
    h_clean, h_masked, head, tokens = _random_pair(random.PRNGKey(3), (2, 6, 8), 32, jnp.float32)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=4)
    position_kl = streamed_position_kl(masked, clean)
    assert position_kl.shape == (2, 6)
    np.testing.assert_allclose(
        np.asarray(position_kl.mean()), np.asarray(streamed_kl_per_position(masked, clean))
    )
    position_ce = streamed_position_next_token_ce(masked, tokens)
    assert position_ce.shape == (2, 5)
    np.testing.assert_allclose(
        np.asarray(position_ce.mean()),
        np.asarray(streamed_next_token_cross_entropy(masked, tokens)),
    )


def test_streamed_ce_yields_nan_for_an_out_of_range_label_like_its_twin():
    """A label in no vocab chunk must not come out as a finite CE (it would be logZ): it is
    NaN — for an overflowing label where the materialized `take_along_axis` fill puts one,
    and for a negative label too (the twin wraps those) — and in-range positions of the
    same batch are untouched."""
    h_clean, _, head, tokens = _random_pair(random.PRNGKey(5), (2, 4, 8), 32, jnp.float32)
    out_of_range = tokens.at[0, 1].set(32).at[1, 3].set(-1)
    package = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    streamed = np.asarray(streamed_position_next_token_ce(package, out_of_range))
    materialized = -np.asarray(
        jnp.take_along_axis(
            jax.nn.log_softmax(h_clean @ head.T, axis=-1)[:, :-1],
            out_of_range[:, 1:, None],
            axis=-1,
        )[..., 0]
    )
    bad = np.zeros_like(streamed, dtype=bool)
    bad[0, 0] = bad[1, 2] = True
    np.testing.assert_array_equal(np.isnan(streamed), bad)
    assert np.isnan(materialized[0, 0]) and np.isfinite(materialized[1, 2])
    np.testing.assert_allclose(streamed[~bad], materialized[~bad], rtol=1e-6, atol=1e-6)


def test_recon_loss_fn_dispatches_on_the_output_edge_and_refuses_a_mix():
    h_clean, h_masked, head, _ = _random_pair(random.PRNGKey(4), (2, 3, 8), 32, jnp.float32)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=4)
    streamed = Qwen36MoeDecomposedModel.recon_loss_fn(masked, clean)
    materialized = Qwen36MoeDecomposedModel.recon_loss_fn(h_masked @ head.T, h_clean @ head.T)
    np.testing.assert_allclose(np.asarray(streamed), np.asarray(materialized), rtol=1e-6, atol=1e-6)
    with pytest.raises(AssertionError, match="mixed model-output edges"):
        Qwen36MoeDecomposedModel.recon_loss_fn(masked, h_clean @ head.T)


# ── the qwen36_moe edge flip, model level ─────────────────────────────────────


def _model_and_vu(key: Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _streamed_twin(model: Qwen36MoeDecomposedModel, n_chunks: int) -> Qwen36MoeDecomposedModel:
    return dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=n_chunks))


def test_qwen36_edges_share_one_forward():
    """The edge flip changes only the output's spelling: the streamed package's
    materialized product is bit-identical to the materialized edge's logits."""
    model, _vu = _model_and_vu(jax.random.PRNGKey(0))
    assert model.output_edge == MaterializedOutputEdge()
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, model.cfg.vocab_size)

    logits = PlacedModel(model=model, placement=None).clean_forward(tokens).output
    package = (
        PlacedModel(model=_streamed_twin(model, 4), placement=None).clean_forward(tokens).output
    )
    assert isinstance(logits, jax.Array)
    assert isinstance(package, StreamedLinearOutput)
    assert package.n_chunks == 4
    np.testing.assert_array_equal(
        np.asarray(package.activations @ package.head.T), np.asarray(logits)
    )


def test_qwen36_masked_recon_grads_agree_across_the_edge_flip():
    """d(recon KL)/d(V/U) through the stochastic masked forward, streamed vs
    materialized — the training path's gradient, edge-flip invariant."""
    model, components = _model_and_vu(jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, model.cfg.vocab_size)

    def loss(target: Qwen36MoeDecomposedModel, value: ComponentStacks) -> Array:
        placed = PlacedModel(model=target, placement=None)
        clean = jax.tree.map(jax.lax.stop_gradient, placed.clean_forward(tokens).output)
        ci = {spec.name: jnp.full((*tokens.shape, spec.C), 0.4) for spec in target.sites}
        masked = placed.masked_forward(
            prepare_compute_weights(placed, value),
            tokens,
            masking=StochasticMasking(
                ci_stacked=placed.stack_ci(ci), draw_key=jax.random.PRNGKey(3), routes=None
            ),
            remat=True,
        ).output
        return placed.recon_loss_fn(masked, clean)

    grads_materialized = jax.jit(jax.grad(loss, argnums=1))(model, components)
    grads_streamed = jax.jit(jax.grad(loss, argnums=1))(_streamed_twin(model, 4), components)
    for got, expected in zip(
        jax.tree.leaves(grads_streamed), jax.tree.leaves(grads_materialized), strict=True
    ):
        got_np, expected_np = np.asarray(got), np.asarray(expected)
        denom = np.linalg.norm(expected_np)
        error = np.linalg.norm(got_np - expected_np) / (denom if denom > 0 else 1.0)
        assert error < 2e-2, error

"""CPU oracle tests for `Qwen36MoeDecomposedModel.component_activation_forward`.

The harvest adapter reads each requested site's ``x @ V`` off one clean forward. Shared
sites emit full `[B, T, C]` arrays; expert sites emit `NarrowCI` bundles on the captured
routing, computed in job space — here each is compared against a per-token dense gather
oracle (`V[ids]` einsums over the same captured taps), so the jobs plumbing (schedule,
grouped matmuls, unsort) is pinned to the arithmetic it claims."""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import ComponentStacks, NarrowCI, init_component_stacks
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    full_site_cs,
    parse_site_name,
    qwen36_moe_site_specs,
    router_idx_tap_key,
    router_weights_tap_key,
    site_name,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import (
    mlp_input_tap_key,
    resid_tap_key,
    site_output_tap_key,
)

B, T = 2, 12


def _model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _oracle_taps(model: Qwen36MoeDecomposedModel, tokens: jax.Array) -> dict[str, jax.Array]:
    cfg = model.cfg
    keys = frozenset(
        key
        for layer in range(cfg.n_layer)
        for key in (
            mlp_input_tap_key(layer),
            router_idx_tap_key(layer),
            router_weights_tap_key(layer),
            site_output_tap_key(site_name(layer, "shared_gate")),
            site_output_tap_key(site_name(layer, "shared_up")),
        )
    )
    return dict(model.clean_forward(tokens, keys, placement=None).captures)


def _expected_site_activation(
    model: Qwen36MoeDecomposedModel,
    prepared: dict[str, dict[str, jax.Array]],
    taps: dict[str, jax.Array],
    site: str,
) -> jax.Array:
    """The per-token dense gather oracle for one site's x @ V (expert sites at the
    narrow `[B, T, k·c]` layout)."""
    cfg = model.cfg
    layer, kind = parse_site_name(site)
    V = prepared[kind]["V"][layer]
    h2 = taps[mlp_input_tap_key(layer)]
    match kind:
        case "shared_gate" | "shared_up":
            return h2 @ V
        case "shared_down":
            gate = taps[site_output_tap_key(site_name(layer, "shared_gate"))]
            up = taps[site_output_tap_key(site_name(layer, "shared_up"))]
            return (jax.nn.silu(gate) * up) @ V
        case "experts_gate" | "experts_up":
            ids = taps[router_idx_tap_key(layer)]
            slot_values = jnp.einsum("btd,btkdc->btkc", h2, V[ids])
            return slot_values.reshape(B, T, -1)
        case "experts_down":
            ids = taps[router_idx_tap_key(layer)]
            weights = taps[router_weights_tap_key(layer)]
            di = cfg.moe_intermediate
            gate_blocks = model.moe.experts_gate[layer].reshape(cfg.n_experts, di, cfg.n_embd)
            up_blocks = model.moe.experts_up[layer].reshape(cfg.n_experts, di, cfg.n_embd)
            gate = jnp.einsum("btd,btkid->btki", h2, gate_blocks[ids])
            up = jnp.einsum("btd,btkid->btki", h2, up_blocks[ids])
            hidden = jax.nn.silu(gate) * up * weights[..., None]
            return jnp.einsum("btki,btkic->btkc", hidden, V[ids]).reshape(B, T, -1)
        case _:
            raise AssertionError(kind)


def test_component_activations_match_the_dense_gather_oracle():
    model, vu = _model_and_vu(jax.random.PRNGKey(0))
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    prepared = model.prepare_compute_weights(vu, None)
    taps = _oracle_taps(model, tokens)

    _, activations = model.component_activation_forward(
        prepared, tokens, sites=model.site_names, capture_keys=frozenset(), placement=None
    )

    assert set(activations) == set(model.site_names)
    for spec in model.sites:
        layer, kind = parse_site_name(spec.name)
        expected = _expected_site_activation(model, prepared, taps, spec.name)
        value = activations[spec.name]
        if kind.startswith("experts_"):
            assert isinstance(value, NarrowCI), spec.name
            assert value.n_experts == cfg.n_experts
            np.testing.assert_array_equal(value.router_indices, taps[router_idx_tap_key(layer)])
            np.testing.assert_allclose(value.values, expected, rtol=2e-4, atol=2e-5)
        else:
            assert isinstance(value, jax.Array), spec.name
            assert value.shape == (B, T, spec.C)
            np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-5)


def test_component_activation_forward_returns_only_requested_sites_and_captures():
    model, vu = _model_and_vu(jax.random.PRNGKey(2))
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(3), (B, T), 0, cfg.vocab_size)
    prepared = model.prepare_compute_weights(vu, None)
    requested_capture = resid_tap_key(0)
    sites = (site_name(0, "experts_gate"), site_name(1, "shared_down"))

    forward, activations = model.component_activation_forward(
        prepared,
        tokens,
        sites=sites,
        capture_keys=frozenset({requested_capture}),
        placement=None,
    )

    assert tuple(activations) == sites
    assert set(forward.captures) == {requested_capture}
    clean = model.clean_forward(tokens, placement=None)
    np.testing.assert_array_equal(forward.output, clean.output)

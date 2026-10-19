"""Legacy attention checkpoints retain parameters, gradients and optimizer trajectory."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CIFnBlock,
    GQACIFnAttention,
    MHACIFnAttention,
    QueryKeyNormalization,
    RMSNormalizedQueryKey,
    init_transformer_parameters,
)
from param_decomp.migrations.ci_attention import migrate_item, migrate_tree
from param_decomp.tests.core.test_global_transformer_ci import _arch


class LegacyBlock(eqx.Module):
    """The persisted fields of a CI block before attention owned its parameters."""

    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    gate: tuple[jax.Array, jax.Array] | None
    norm_scales: tuple[jax.Array, jax.Array] | None


class LegacyQKNormBlock(LegacyBlock):
    qk_norm_scales: tuple[jax.Array, jax.Array] | None


def _legacy(block: CIFnBlock, optional_field: bool) -> LegacyBlock:
    attn = block.attention
    legacy = LegacyBlock(
        wq=attn.wq,
        wk=attn.wk,
        wv=attn.wv,
        wo=attn.wo,
        w1=block.w1,
        b1=block.b1,
        w2=block.w2,
        b2=block.b2,
        gate=block.gate,
        norm_scales=block.norm_scales,
    )
    normalization = attn.query_key_norm
    if isinstance(normalization, RMSNormalizedQueryKey) or optional_field:
        gains = (
            (normalization.q_gain, normalization.k_gain)
            if isinstance(normalization, RMSNormalizedQueryKey)
            else None
        )
        return LegacyQKNormBlock(
            wq=legacy.wq,
            wk=legacy.wk,
            wv=legacy.wv,
            wo=legacy.wo,
            w1=legacy.w1,
            b1=legacy.b1,
            w2=legacy.w2,
            b2=legacy.b2,
            gate=legacy.gate,
            norm_scales=legacy.norm_scales,
            qk_norm_scales=gains,
        )
    return legacy


def _assert_equal(left: object, right: object) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("optimizer_kind", ["adam", "muon"])
@pytest.mark.parametrize("qk_norm", [QueryKeyNormalization.NONE, QueryKeyNormalization.RMS])
@pytest.mark.parametrize("optional_field", [False, True])
@pytest.mark.parametrize("gqa", [False, True])
def test_migrated_checkpoint_continues_optimizer_exactly(
    tmp_path: Path,
    qk_norm: QueryKeyNormalization,
    optional_field: bool,
    gqa: bool,
    optimizer_kind: Literal["adam", "muon"],
) -> None:
    definition = (
        GQACIFnAttention(
            n_heads=2, n_kv_heads=1, implementation="xla", mask="causal", qk_norm=qk_norm
        )
        if gqa
        else MHACIFnAttention(n_heads=2, implementation="xla", mask="causal", qk_norm=qk_norm)
    )
    arch = replace(_arch("causal", "swiglu"), attention=definition)
    block = init_transformer_parameters(arch, 8, (4,), jax.random.key(1)).blocks[0]
    gradient = jax.tree.map(lambda x: x * 0.13 + 0.017, block)
    params = eqx.filter(block, eqx.is_array)
    grads = eqx.filter(gradient, eqx.is_array)
    old_params = eqx.filter(_legacy(block, optional_field), eqx.is_array)
    old_grads = eqx.filter(_legacy(gradient, optional_field), eqx.is_array)
    match optimizer_kind:
        case "adam":
            optimizer = optax.adam(1e-3)
        case "muon":
            optimizer = optax.contrib.muon(1e-3)
    # Advance once so restore must preserve nonzero first/second moments and count.
    _, old_state = optimizer.update(old_grads, optimizer.init(old_params), old_params)
    _, expected_state = optimizer.update(grads, optimizer.init(params), params)
    source, destination = tmp_path / "legacy", tmp_path / "migrated"
    with ocp.StandardCheckpointer() as cp:
        cp.save(source, {"params": _legacy(block, optional_field), "optimizer": old_state})
    migrate_item(source, destination)
    target = {"params": block, "optimizer": expected_state}
    with ocp.StandardCheckpointer() as cp:
        restored = cp.restore(destination, target=target)
    _assert_equal(restored, target)
    expected_updates, expected_state = optimizer.update(grads, expected_state, params)
    actual_updates, actual_state = optimizer.update(
        grads, restored["optimizer"], restored["params"]
    )
    _assert_equal(actual_updates, expected_updates)
    _assert_equal(actual_state, expected_state)
    # Repeat migration cannot overwrite the destination, and the original remains readable.
    with pytest.raises(AssertionError, match="destination already exists"):
        migrate_item(source, destination)
    with ocp.StandardCheckpointer() as cp:
        original = cp.restore(
            source, target={"params": _legacy(block, optional_field), "optimizer": old_state}
        )
    _assert_equal(original["params"], _legacy(block, optional_field))


def test_migration_preserves_masked_optimizer_leaves_and_rejects_mixed_paths() -> None:
    block = dict(wq=None, wk=None, wv=None, wo=None, w1=None, w2=None, qk_norm_scales=(None, None))
    assert migrate_tree(block) == {
        "w1": None,
        "w2": None,
        "attention": {
            "wq": None,
            "wk": None,
            "wv": None,
            "wo": None,
            "query_key_norm": {"q_gain": None, "k_gain": None},
        },
    }
    with pytest.raises(AssertionError, match="mixed old/new"):
        migrate_tree({**block, "attention": {}})
    current = {"attention": {"wq": None}, "w1": None, "w2": None}
    assert migrate_tree(current) == current


@pytest.mark.multidevice
def test_migration_reads_sharded_bf16_arrays_without_their_saved_layout(tmp_path: Path) -> None:
    """Migration materializes host arrays rather than reusing a training-device layout."""
    if jax.device_count() < 2:
        pytest.skip("requires two local devices")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("heads",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("heads"))
    projection = jax.device_put(np.arange(16).reshape(4, 4).astype(jax.numpy.bfloat16), sharding)
    source, destination = tmp_path / "sharded", tmp_path / "host"
    legacy = {
        "wq": projection,
        "wk": projection,
        "wv": projection,
        "wo": projection,
        "w1": None,
        "w2": None,
    }
    with ocp.StandardCheckpointer() as cp:
        cp.save(source, legacy)
    migrate_item(source, destination)
    with ocp.StandardCheckpointer() as cp:
        restored = cp.restore(destination)
    for name in ("wq", "wk", "wv", "wo"):
        value = restored["attention"][name]
        assert isinstance(value, np.ndarray)
        assert value.dtype == projection.dtype
        np.testing.assert_array_equal(value, projection)

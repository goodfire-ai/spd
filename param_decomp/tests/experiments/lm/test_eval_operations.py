"""Tests for target-generic evaluation operations bound to LM runs."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.ci_fn import resolve_ci_placement
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import CaptureKeys
from param_decomp.core.placement import from_config
from param_decomp.core.run import PassOperation, _run_due_evaluation
from param_decomp.core.sharding import place_target
from param_decomp.core.train import Decomposition, TrainState
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm import eval_operations
from param_decomp.experiments.lm.eval_config import WellTemperednessConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.infra.dataset_store import DatasetMeta, write_dataset_meta
from param_decomp.targets.qwen36_moe import full_site_cs, qwen36_moe_site_specs
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_arch,
    tiny_qwen36_moe_ci_fn,
)


def _eval_config() -> EvalConfig:
    return EvalConfig(
        batch_size=8,
        n_steps=3,
        every=10,
        slow_every=20,
        metrics=[
            WellTemperednessConfig(
                groups=None,
                n_locations=2,
                n_components_per_region=4,
                ablations_per_forward=4,
            )
        ],
    )


def test_global_token_batch_refuses_ids_outside_the_vocab() -> None:
    """The one host boundary every LM token stream crosses: past it the ids are labels
    under jit, where an out-of-range id is at best a NaN CE."""
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.sharding import HSDP_MESH_AXES

    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1),
        HSDP_MESH_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )
    tokens = np.array([[0, 5, 31], [7, 1, 2]], dtype=np.int32)
    assert eval_operations.global_token_batch(tokens, mesh, 2, 32).shape == (2, 3)
    with pytest.raises(AssertionError, match=r"outside \[0, 31\).*max 31"):
        eval_operations.global_token_batch(tokens, mesh, 2, 31)
    with pytest.raises(AssertionError, match=r"min -1"):
        eval_operations.global_token_batch(tokens - 1, mesh, 2, 32)


def test_well_temperedness_uses_named_rng_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def make_operation(
        metric: WellTemperednessConfig,
        schedule: Any,
        model: Any,
        ci_capture_keys: CaptureKeys,
        mesh: Any,
        compiler_options: dict[str, bool | int | str],
        *,
        inputs_for_context: Any,
        figure_rendering: Any,
    ) -> PassOperation[Any]:
        captured.update(
            metric=metric,
            model=model,
            ci_capture_keys=ci_capture_keys,
            mesh=mesh,
            compiler_options=compiler_options,
            inputs_for_context=inputs_for_context,
            figure_rendering=figure_rendering,
        )
        return PassOperation(schedule, lambda _context: {})

    renderer = object()
    monkeypatch.setattr(eval_operations, "make_well_temperedness_operation", make_operation)
    monkeypatch.setattr(eval_operations, "BackgroundRenderer", lambda _sink: renderer)
    monkeypatch.setattr(eval_operations, "scan_shards", lambda _path: ())
    monkeypatch.setattr(eval_operations, "BatchSchedule", lambda *_args: object())
    monkeypatch.setattr(
        eval_operations,
        "read_dataset_meta",
        lambda _path: SimpleNamespace(seq_len=4),
    )
    monkeypatch.setattr(
        eval_operations,
        "ShardServer",
        lambda *_args: SimpleNamespace(per_process=jax.local_device_count()),
    )
    monkeypatch.setattr(eval_operations, "target_vocab_size", lambda _model: 32)
    run_key = jax.random.PRNGKey(11)
    train_steps = 100
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=train_steps, seed=3),
        data=SimpleNamespace(eval_dir=Path("unused")),
        ci_fn=SimpleNamespace(capture_keys=frozenset()),
        target=cast(Any, None),
    )

    evaluation = eval_operations.make_lm_evaluation(
        cast(Any, built),
        _eval_config(),
        cast(Any, SimpleNamespace(site_names=("site",), sites=())),
        run_key,
        cast(Any, None),
        n_proc=1,
        sink=cast(Any, SimpleNamespace(accepts_deferred_media=True)),
        compiler_options={},
    )
    batch = jnp.arange(4)
    _, key = captured["inputs_for_context"](
        LMEvalPass(
            state=cast(Any, None),
            now_step=30,
            placed_ci_fn=cast(Any, None),
            pass_index=3,
            batches=(batch,),
        )
    )

    assert len(evaluation.operations) == 1
    assert captured["figure_rendering"] is renderer
    assert captured["ci_capture_keys"] == frozenset()
    np.testing.assert_array_equal(
        key,
        jax.random.fold_in(
            run_key,
            EvalKeyStream.WELL_TEMPEREDNESS * train_steps + 3,
        ),
    )


def _write_eval_shards(shards_dir: Path, vocab_size: int, seq_len: int, n_rows: int) -> None:
    shards_dir.mkdir(parents=True)
    write_dataset_meta(shards_dir, DatasetMeta(seq_len=seq_len, tokenizer_name="unused"))
    rows = np.random.default_rng(1).integers(0, vocab_size, size=(n_rows, seq_len), dtype=np.int32)
    pq.write_table(
        pa.table({"input_ids": [row.tolist() for row in rows]}), shards_dir / "shard_00000.parquet"
    )


def test_standing_nonlinearity_operation_runs_at_step_zero_on_expert_blocked_sites(
    tmp_path: Path,
) -> None:
    """qwen36's gate/up kinds all carry a `Neurons` partition, so the standing nonlinearity
    operation binds on every qwen36 run with an `eval:` block, and `slow_on_first_step`
    fires it at step 0. The expert kinds' U is block-dim `[E, c, d]`; the operation must
    hand its per-component statistics over in the flat `[C]` order the CI means arrive in."""
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model = tiny_qwen36_decomposed_model(cfg, sites, random.PRNGKey(0))
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed = place_target(model, rules)
    components = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
    ci_fn = tiny_qwen36_moe_ci_fn(model, random.PRNGKey(2))
    seq_len = 8
    eval_dir = tmp_path / "eval"
    _write_eval_shards(eval_dir, cfg.vocab_size, seq_len, n_rows=16)
    eval_config = EvalConfig(
        batch_size=2, n_steps=1, every=10, slow_every=10, slow_on_first_step=True, metrics=[]
    )
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=20, seed=0),
        data=SimpleNamespace(eval_dir=eval_dir),
        ci_fn=ci_fn,
        target=None,
    )

    ci_placement = resolve_ci_placement(tiny_qwen36_moe_ci_arch(model), rules)
    assert ci_placement is not None
    with jax.set_mesh(mesh):
        placed_ci_fn = jax.device_put(ci_fn, ci_fn.shardings(mesh, ci_placement))
        evaluation = eval_operations.make_lm_evaluation(
            cast(Any, built),
            eval_config,
            placed,
            random.PRNGKey(3),
            mesh,
            n_proc=1,
            sink=cast(Any, SimpleNamespace(accepts_deferred_media=False)),
            compiler_options={},
        )
        assert len(evaluation.operations) == 1, "the standing operation binds with no metrics"
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=placed_ci_fn),
            training=cast(Any, None),
        )
        record = _run_due_evaluation(evaluation, state, 0, ci_placement)

    assert record is not None
    partitioned = [site for site in sites if site.nonlinearity_partition is not None]
    assert partitioned
    for site in partitioned:
        prefix = f"eval/nonlinearity/sites/{site.name}"
        n_alive = record[f"{prefix}/mean_ci_gt_0/n_components"]
        assert isinstance(n_alive, float) and 0 <= n_alive <= site.C
        soft = record[f"{prefix}/all/soft_use_count_relative_threshold_4"]
        assert isinstance(soft, float) and np.isfinite(soft)
    assert "eval/nonlinearity/aggregates/neuron/all/effective_use_count_per_subcomponent" in record

"""Tests for the toy binding of authored evaluation operations."""

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import pytest

from param_decomp.experiments import toy_eval
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig, WellTemperednessConfig


@pytest.mark.parametrize(
    ("metric", "reason"),
    [
        (CEandKLLossesConfig(rounding_threshold=0.5), "neither tokens nor logits"),
        (
            WellTemperednessConfig(
                groups=None, n_locations=2, n_components_per_region=4, ablations_per_forward=4
            ),
            "positionless toy target has no positions",
        ),
    ],
)
def test_lm_only_metrics_refuse_at_toy_binding(metric: Any, reason: str) -> None:
    eval_config = EvalConfig(batch_size=8, n_steps=3, every=10, slow_every=20, metrics=[metric])
    with pytest.raises(AssertionError, match=reason):
        toy_eval.make_toy_evaluation_operations(
            eval_config,
            7,
            compiler_options={},
            model=cast(Any, SimpleNamespace(site_names=("site",))),
            ci_capture_keys=frozenset(),
            mesh=cast(Any, None),
            sample_eval_batch=lambda index: jnp.array([index]),
            probe_ci=cast(Any, None),
            wandb_configured=False,
        )

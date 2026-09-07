"""The LM eval pass's shared batch context on either model-output edge.

The context step pins the clean output over the batch axes. On the streamed edge that
output is the factored `StreamedLinearOutput` package, not an array — a raw array pin
died at trace, so the first eval tick of a streamed-edge run never ran. The context must
build, and the CE/KL scorer must consume it, on both edges of the tiny qwen36_moe target
placed on a two-device `(data, tp)` mesh."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.ci_fn import PlacedCIFn, resolve_ci_placement
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import EMPTY_CAPTURE_KEYS
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target, shard_batch
from param_decomp.experiments.lm.eval import make_ce_kl_scorer
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    make_lm_batch_context_step,
    prepared_batch_from_context,
)
from param_decomp.targets.lm_output import StreamedLinearOutput
from param_decomp.targets.qwen36_moe import (
    MaterializedOutputEdge,
    OutputEdge,
    StreamedOutputEdge,
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_arch,
    tiny_qwen36_moe_ci_fn,
)

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 2,
        reason="requires a two-device CPU topology from make test-multidevice",
    ),
]

# Every C tiles the (data=2, tp=1) mesh: expert Cs are n_experts=4 times an even
# c_per_expert, dense Cs are even.
SITE_CS: dict[str, int] = {
    "experts_gate": 8,
    "experts_up": 8,
    "experts_down": 8,
    "shared_gate": 4,
    "shared_up": 4,
    "shared_down": 4,
}
BATCH, SEQ = 4, 8


@pytest.mark.parametrize(
    "output_edge",
    [MaterializedOutputEdge(), StreamedOutputEdge(n_vocab_chunks=4)],
    ids=["materialized", "streamed"],
)
def test_batch_context_builds_and_scores_on_either_output_edge(output_edge: OutputEdge):
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, SITE_CS))
    model = dataclasses.replace(
        tiny_qwen36_decomposed_model(cfg, sites, random.PRNGKey(0)), output_edge=output_edge
    )
    ci_fn = tiny_qwen36_moe_ci_fn(model, random.PRNGKey(2))
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed = place_target(model, rules)
    components = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
    tokens = shard_batch(
        random.randint(random.PRNGKey(3), (BATCH, SEQ), 0, cfg.vocab_size), mesh, batch_axis=0
    )

    ci_placement = resolve_ci_placement(tiny_qwen36_moe_ci_arch(model), rules)
    assert ci_placement is not None
    with jax.set_mesh(mesh):
        placed_ci_fn = PlacedCIFn(
            fn=jax.device_put(ci_fn, ci_fn.shardings(mesh, ci_placement)), placement=ci_placement
        )
        context_step = make_lm_batch_context_step(
            placed, ci_fn.capture_keys, EMPTY_CAPTURE_KEYS, mesh
        )
        ctx_tokens, clean_output, captures, ci, prepared_weights = context_step(
            placed, components, placed_ci_fn, tokens
        )
        match output_edge:
            case MaterializedOutputEdge():
                assert isinstance(clean_output, jax.Array)
                assert clean_output.shape == (BATCH, SEQ, cfg.vocab_size)
            case StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks):
                assert isinstance(clean_output, StreamedLinearOutput)
                assert clean_output.activations.shape == (BATCH, SEQ, cfg.n_embd)
                assert clean_output.n_chunks == n_vocab_chunks

        context = LMBatchContext(
            pass_index=0,
            batch_index=0,
            tokens=ctx_tokens,
            clean_output=clean_output,
            captures=captures,
            ci=ci,
            prepared_weights=prepared_weights,
        )
        metrics = make_ce_kl_scorer(placed, 0.5, mesh)(
            placed, prepared_batch_from_context(context, EMPTY_CAPTURE_KEYS), random.PRNGKey(4)
        )
    assert metrics.keys() >= {"ce_kl/kl_ci_masked", "ce_kl/ce_difference_unmasked"}
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())

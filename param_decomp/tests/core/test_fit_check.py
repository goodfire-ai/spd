"""The deviceless fit check declares the runtime's TrainState: every piece of state the
objective demands must be typed into the standin, or the receipt dies at trace instead
of reaching a verdict."""

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import ExpertBlockedSource
from param_decomp.core.ci_fn import Chunk, ChunkwiseTransformerCIArch, MHACIAttention
from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    PDConfig,
)
from param_decomp.core.model import Positioned
from param_decomp.core.placement import batch_axes, from_config
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.core.tools.fit_check import (
    fit_report_of_compiled,
    lowered_train_step,
    standin_faithfulness_loss,
)
from param_decomp.targets.llama_simple_mlp import (
    KIND_ORDER,
    canonical_site_cs,
    site_name,
    site_specs,
)
from param_decomp.targets.testing import tiny_simple_mlp_cfg, tiny_simple_mlp_decomposed_model

_B, _T, _C = 4, 8, 8


def test_ema_frequency_and_pooled_adversary_reach_a_receipt_verdict():
    """The stand-in state must carry every configured persistent runtime leaf."""
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), _C)
                for layer in range(cfg.n_layer)
                for kind in KIND_ORDER
            )
        ),
    )
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    placed = place_target(model, from_config("zero1", mesh, model.sites))
    ci_fn = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("resid.0",), output_sites=placed.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    pd = PDConfig(
        steps=100,
        batch_size=_B,
        components_optimizer=AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)),
        ci_fn_optimizer=AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)),
        loss_metrics=[
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig.constant(1.0),
                frequency=FrequencyMinimalityConfig(
                    coeff=1e-6, reference_datapoint_count=_B * _T, ema_halflife_steps=8.0
                ),
            ),
            MergedStochasticSubsetPooledPPGDReconLossConfig(
                coeff=1.0,
                pool_size=8,
                adv_fraction=ScheduleConfig.constant(0.5),
                optimizer=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.02)),
                n_warmup_steps=1,
            ),
        ],
    )
    batch = jax.ShapeDtypeStruct(
        (_B, _T), np.int32, sharding=NamedSharding(mesh, P(batch_axes(mesh), None))
    )
    lowered, declared = lowered_train_step(
        pd,
        ci_fn,
        placed,
        Positioned(n_positions=_T),
        batch,
        standin_faithfulness_loss(placed),
        remat_recon_forwards=True,
        remat_ci_fn=False,
        compiler_options=None,
    )
    freq_ema = declared.state.training.freq_ema
    assert freq_ema is not None
    assert {name: leaf.shape for name, leaf in freq_ema.items()} == {
        spec.name: (spec.C,) for spec in placed.sites
    }
    (adversary,) = declared.state.training.adversaries.values()
    for stack in adversary.sources.stacks.values():
        assert stack.delta.shape[1:] == (8,)
        assert not isinstance(stack.components, ExpertBlockedSource)
        assert stack.components.shape[1:] == (8, _C)
    report = fit_report_of_compiled(lowered.compile(), pool_gib=8.0)
    assert "VERDICT" in report.render()

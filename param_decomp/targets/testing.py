"""Tiny random targets — the standard CPU-test fixtures.

One tiny random target per LM family (the Llama-3.1-flavored GLU transformer and the
`LlamaSimpleMLP`), plus a one-chunk chunkwise CI fn over each. Engine tests use these as
the concrete target behind the `DecomposedModel` protocol; the per-target suites
(`param_decomp/tests/targets/`) use them as the system under test. Toy dims throughout —
no real weights, no GPU.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    FullSlot,
    MHACIAttention,
    MoEChunk,
    MoEChunkwiseTransformerCIArch,
    MoEChunkwiseTransformerCIFn,
    NarrowSlot,
    RoutingTap,
    build_ci_fn,
)
from param_decomp.core.components import SiteC, SiteCI, SiteSpec
from param_decomp.core.model import DecomposedModel, MaterializedMasking
from param_decomp.target_ports.llama import LlamaConfig, llama3_inv_freq
from param_decomp.targets import llama_simple_mlp, qwen36_moe
from param_decomp.targets.glu_transformer import (
    FrozenAttn,
    GatedMLP,
    GLUDecomposedModel,
    GLULayer,
    PlainMLP,
    build_decomposed_lm,
    parse_site_name,
)
from param_decomp.targets.llama_simple_mlp import (
    LlamaSimpleMLPConfig,
    build_decomposed_simple_mlp,
)
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import (
    AttnSublayer,
    DeltaNetSublayer,
    FrozenGatedAttention,
    FrozenGatedDeltaNet,
    FrozenMoE,
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    build_qwen36_moe_model,
    layer_is_full_attention,
    router_idx_tap_key,
    router_weights_tap_key,
)
from param_decomp.targets.transformer_taps import resid_tap_key


def _tiny_chunkwise_ci_arch(
    model: GLUDecomposedModel, first_block: int, input_dim: int, n_blocks: int
) -> ChunkwiseTransformerCIArch:
    """One chunk reading the residual entering the first decomposed block, emitting CI
    for every site. `input_dim` is the target residual width (`n_embd`)."""
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=model.site_names),),
        input_dim=input_dim,
        d_model=16,
        n_blocks=n_blocks,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _tiny_chunkwise_ci_fn(
    model: GLUDecomposedModel, key: jax.Array, first_block: int, input_dim: int, n_blocks: int
) -> ChunkwiseTransformerCIFn:
    arch = _tiny_chunkwise_ci_arch(model, first_block, input_dim, n_blocks)
    ci_fn = build_ci_fn(arch, model.sites, key)
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
    return ci_fn


def tiny_glu_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )


def tiny_glu_decomposed_lm(
    cfg: LlamaConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> GLUDecomposedModel:
    """A tiny random `GLUDecomposedModel` (random embedding + full frozen layer stack
    plus the decomposition `sites`) — the CPU-test analog of `load_decomposed_lm_from_hf`."""
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep, "auto",
        )  # fmt: skip

    def layer():
        # attn drawn before the MLP: the key-consumption order the committed
        # fixture-derived tests (slow-eval histograms) were pinned against.
        attn = fattn()
        mlp = GatedMLP(Wg=n((di, d)), Wu=n((di, d)), Wd=n((d, di)))
        return GLULayer(jnp.ones((d,)), jnp.ones((d,)), attn, mlp)

    return build_decomposed_lm(
        embed=n((cfg.vocab_size, d), 0.02),
        layers=[layer() for _ in range(cfg.n_layer)],
        norm=jnp.ones((d,)),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
    )


def tiny_glu_chunkwise_ci_arch(
    model: GLUDecomposedModel, n_blocks: int
) -> ChunkwiseTransformerCIArch:
    first_block = min(parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_arch(model, first_block, tiny_glu_cfg().n_embd, n_blocks)


def tiny_glu_chunkwise_ci_fn(
    model: GLUDecomposedModel, key: jax.Array, n_blocks: int
) -> ChunkwiseTransformerCIFn:
    first_block = min(parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_glu_cfg().n_embd, n_blocks)


def tiny_simple_mlp_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        vocab_size=64,
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rotary_base=10000.0,
        rms_norm_eps=1e-6,
        n_ctx=64,
    )


def _tiny_simple_mlp_layers(cfg: LlamaSimpleMLPConfig, n: int, key: jax.Array) -> list[GLULayer]:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def rand(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(ks), shape) * d**-0.5

    def layer() -> GLULayer:
        # attn drawn before the MLP: the key-consumption order the committed
        # fixture-derived tests (slow-eval histograms) were pinned against.
        attn = FrozenAttn(
            rand((qd, d)),
            rand((kvd, d)),
            rand((kvd, d)),
            rand((d, qd)),
            cfg.n_head,
            cfg.n_kv_head,
            cfg.head_dim,
            cfg.n_rep,
            "auto",
        )
        mlp = PlainMLP(Wfc=rand((di, d)), Wdown=rand((d, di)))
        return GLULayer(ln1=jnp.ones((d,)), ln2=jnp.ones((d,)), attn=attn, mlp=mlp)

    return [layer() for _ in range(n)]


def tiny_simple_mlp_decomposed_model(
    cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> GLUDecomposedModel:
    """A tiny random engine-hosted SimpleMLP carrying a random (tied) embedding + full
    frozen layer stack plus the decomposition `sites`."""
    layers_key, embed_key = jax.random.split(key)
    layers = _tiny_simple_mlp_layers(cfg, cfg.n_layer, layers_key)
    embed = jax.random.normal(embed_key, (cfg.vocab_size, cfg.n_embd)) * 0.02
    return build_decomposed_simple_mlp(
        embed=embed, layers=layers, norm=jnp.ones((cfg.n_embd,)),
        cfg=cfg, sites=sites,
    )  # fmt: skip


SIMPLE_MLP_MIXED_SITE_CS = (
    SiteC("h.2.attn.q_proj", 8),
    SiteC("h.2.attn.v_proj", 12),
    SiteC("h.2.mlp.c_fc", 8),
    SiteC("h.2.mlp.down_proj", 16),
    SiteC("h.3.attn.q_proj", 8),
    SiteC("h.3.attn.v_proj", 12),
    SiteC("h.3.mlp.c_fc", 8),
    SiteC("h.3.mlp.down_proj", 16),
)
"""Mixed attention + MLP kinds with heterogeneous per-kind C, RECTANGULAR over the
contiguous layer range 2..3 — the engine's segmented masked forward requires every
decomposed kind on every decomposed layer."""


def tiny_simple_mlp_chunkwise_ci_fn(
    model: GLUDecomposedModel, key: jax.Array
) -> ChunkwiseTransformerCIFn:
    first_block = min(llama_simple_mlp.parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_simple_mlp_cfg().n_embd, 2)


# These projections deliberately take the RAW `DecomposedModel`, not the `PlacedModel`
# bundle: the per-target suites exercise the protocol surface itself — per-call
# `placement`, here the unplaced (None) CPU-test execution — below the engine's
# one-bundle assembly.


def run_clean[Out](model: DecomposedModel[Out], inputs: Any) -> Out:
    """Test-only output projection of the capture-aware clean forward."""
    return model.clean_forward(inputs, placement=None).output


def materialized_logits(output: LMOutput) -> jax.Array:
    """The materialized edge's logits — the tests that do array arithmetic on an LM output
    narrow through this; a streamed package refuses."""
    assert isinstance(output, jax.Array), type(output).__name__
    return output


def capture_clean[Out](
    model: DecomposedModel[Out], inputs: Any, keys: Iterable[str]
) -> dict[str, jax.Array]:
    return model.clean_forward(inputs, frozenset(keys), placement=None).captures


def run_masked[Out, PreparedT](
    model: DecomposedModel[Out, PreparedT],
    prepared_weights: PreparedT,
    inputs: Any,
    masks: Mapping[str, SiteCI],
    delta_masks: dict[str, jax.Array],
    routes: dict[str, jax.Array] | None,
    uses_weight_deltas: bool,
    *,
    remat: bool,
) -> Out:
    """Test-only output projection of a masked forward. The mask dicts must cover exactly
    the model's sites (the target asserts this)."""
    return model.masked_forward(
        prepared_weights,
        inputs,
        masking=MaterializedMasking(
            component_masks=masks,
            weight_delta_masks=delta_masks if uses_weight_deltas else None,
            routes=routes,
        ),
        placement=None,
        remat=remat,
    ).output


def capture_site_outputs[Out, PreparedT](
    model: DecomposedModel[Out, PreparedT],
    prepared_weights: PreparedT,
    inputs: Any,
    masking: MaterializedMasking,
) -> dict[str, jax.Array]:
    sites = tuple(masking.component_masks)
    site_output_keys = model.site_output_keys(sites)
    masked_forward_result = model.masked_forward(
        prepared_weights,
        inputs,
        masking=masking,
        placement=None,
        capture_keys=frozenset(site_output_keys),
        remat=False,
    )
    return {
        site: masked_forward_result.captures[key]
        for site, key in zip(sites, site_output_keys, strict=True)
    }


def tiny_qwen36_cfg() -> Qwen36MoeConfig:
    """Two stages of one DeltaNet + one attention layer — the fewest that exercise both
    mixers and a stage boundary. Every dimension is distinct so no two axes coincide;
    site `d_in`s (`n_embd`, the two hidden widths) tile the (data=4, tp=2) test mesh."""
    return Qwen36MoeConfig(
        vocab_size=40,
        n_layer=4,
        full_attention_interval=2,
        n_embd=12,
        n_head=2,
        n_kv_head=1,
        head_dim=16,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        linear_num_key_heads=2,
        linear_key_head_dim=6,
        linear_num_value_heads=4,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=3,
        n_experts=4,
        n_experts_per_token=2,
        moe_intermediate=8,
        shared_expert_intermediate=20,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
    )


# Expert-site Cs must be multiples of n_experts=4: components are expert-local, so a
# site's flat C is n_experts * c_per_expert by construction.
TINY_QWEN36_CS: dict[str, int] = {
    "experts_gate": 8,
    "experts_up": 8,
    "experts_down": 12,
    "shared_gate": 4,
    "shared_up": 4,
    "shared_down": 5,
}


def tiny_qwen36_decomposed_model(
    cfg: Qwen36MoeConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> Qwen36MoeDecomposedModel:
    """A tiny random qwen36_moe model — the CPU-test analog of
    `load_decomposed_qwen36_moe_from_hf`. Norm weights are drawn non-trivially in each
    convention (zero-centered near 0, the gated norm near 1) so a conflated convention
    shows."""
    ks = iter(jax.random.split(key, 4096))
    d = cfg.n_embd
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    fused = cfg.n_experts * cfg.moe_intermediate

    def n(shape: tuple[int, ...], scale: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (scale or d**-0.5)

    def centered_norm(shape: tuple[int, ...]) -> jax.Array:
        return 0.1 * jax.random.normal(next(ks), shape)

    def deltanet_sublayer() -> DeltaNetSublayer:
        return DeltaNetSublayer(
            ln1=centered_norm((d,)),
            mixer=FrozenGatedDeltaNet(
                w_q=n((cfg.linear_key_dim, d)),
                w_k=n((cfg.linear_key_dim, d)),
                w_v=n((cfg.linear_value_dim, d)),
                w_z=n((cfg.linear_value_dim, d)),
                w_b=n((cfg.linear_num_value_heads, d)),
                w_a=n((cfg.linear_num_value_heads, d)),
                conv_q=n((cfg.linear_key_dim, cfg.linear_conv_kernel_dim), 0.5),
                conv_k=n((cfg.linear_key_dim, cfg.linear_conv_kernel_dim), 0.5),
                conv_v=n((cfg.linear_value_dim, cfg.linear_conv_kernel_dim), 0.5),
                a_log=0.3 * jax.random.normal(next(ks), (cfg.linear_num_value_heads,)),
                dt_bias=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.linear_num_value_heads,)),
                norm_w=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.linear_value_head_dim,)),
                w_out=n((d, cfg.linear_value_dim)),
                n_k_heads=cfg.linear_num_key_heads,
                n_v_heads=cfg.linear_num_value_heads,
                k_head_dim=cfg.linear_key_head_dim,
                v_head_dim=cfg.linear_value_head_dim,
                eps=cfg.rms_norm_eps,
            ),
        )

    def attn_sublayer() -> AttnSublayer:
        return AttnSublayer(
            ln1=centered_norm((d,)),
            attn=FrozenGatedAttention(
                wq=n((2 * qd, d)),
                wk=n((kvd, d)),
                wv=n((kvd, d)),
                wo=n((d, qd)),
                q_norm=centered_norm((cfg.head_dim,)),
                k_norm=centered_norm((cfg.head_dim,)),
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                eps=cfg.rms_norm_eps,
                implementation="auto",
            ),
        )

    def moe_layer() -> FrozenMoE:
        return FrozenMoE(
            ln=centered_norm((d,)),
            router=n((cfg.n_experts, d)),
            experts_gate=n((fused, d)),
            experts_up=n((fused, d)),
            experts_down=n((d, fused)),
            shared_gate=n((cfg.shared_expert_intermediate, d)),
            shared_up=n((cfg.shared_expert_intermediate, d)),
            shared_down=n((d, cfg.shared_expert_intermediate)),
            shared_expert_gate=n((1, d)),
        )

    return build_qwen36_moe_model(
        cfg,
        sites,
        embed=n((cfg.vocab_size, d), 0.05),
        deltanet_sublayers=[
            deltanet_sublayer()
            for layer in range(cfg.n_layer)
            if not layer_is_full_attention(cfg, layer)
        ],
        attn_sublayers=[
            attn_sublayer() for layer in range(cfg.n_layer) if layer_is_full_attention(cfg, layer)
        ],
        moe_layers=[moe_layer() for _ in range(cfg.n_layer)],
        norm=centered_norm((d,)),
        lm_head=n((cfg.vocab_size, d), 0.05),
        stack=jnp.stack,
        # toy dims sit below the split arm's 64-multiple kernel tiling: the oracle arm
        # serves every engine-semantics fixture (kernel parity lives in
        # tests/routed/test_experts at 64-multiple shapes)
        grouped_matmul_backend="ragged_dot",
    )


def tiny_qwen36_moe_ci_arch(model: Qwen36MoeDecomposedModel) -> MoEChunkwiseTransformerCIArch:
    """One chunk per target stage over the model's OWN sites: expert kinds narrow
    (router = in-stage position), shared kinds full — the resolver's shape at the tiny
    config."""
    cfg = model.cfg
    interval = cfg.full_attention_interval
    chunk_slots: list[list[FullSlot | NarrowSlot]] = [[] for _ in range(cfg.n_stages)]
    for spec in model.sites:
        layer, kind = qwen36_moe.parse_site_name(spec.name)
        chunk_slots[layer // interval].append(
            NarrowSlot(site=spec.name, router=layer % interval)
            if qwen36_moe.is_expert_kind(kind)
            else FullSlot(site=spec.name)
        )
    chunks = tuple(
        MoEChunk(
            input_taps=(resid_tap_key(start),),
            routing=tuple(
                RoutingTap(
                    ids_key=router_idx_tap_key(layer),
                    weights_key=router_weights_tap_key(layer),
                )
                for layer in range(start, start + interval)
            ),
            slots=tuple(chunk_slots[start // interval]),
        )
        for start in range(0, cfg.n_layer, interval)
    )
    # Dims sized to tile the placed suites' (data=4, tp=2) mesh (÷8 on the sharded axes).
    return MoEChunkwiseTransformerCIArch(
        chunks=chunks,
        input_dim=cfg.n_embd + interval * cfg.n_experts,
        d_model=16,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        n_experts=cfg.n_experts,
        expert_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        # toy dims sit below the split arm's 64-multiple kernel tiling: the oracle arm
        grouped_matmul_backend="ragged_dot",
    )


def tiny_qwen36_moe_ci_fn(
    model: Qwen36MoeDecomposedModel, key: jax.Array
) -> MoEChunkwiseTransformerCIFn:
    ci_fn = build_ci_fn(tiny_qwen36_moe_ci_arch(model), model.sites, key)
    assert isinstance(ci_fn, MoEChunkwiseTransformerCIFn)
    return ci_fn

# ruff: noqa: E402, I001
# %%
from dataclasses import dataclass
from fnmatch import fnmatch
import numpy as np
import einops
from jaxtyping import Float, Int

import torch.nn.functional as F
import torch
from typing import Any, Literal, cast
from collections.abc import Callable, Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.configs import PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import ComponentsMaskInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data, runtime_cast
from torch import Tensor
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

from spd.utils.module_utils import get_target_module_paths

# %%

device = get_device()

# %%

RUN = "wandb:goodfire/spd/runs/9gf5ud48"

config = SPDRunInfo.from_path(RUN).config

task_config = config.task_config
assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

model = ComponentModel.from_run_info(SPDRunInfo.from_path(RUN))
model.to(device)
model.eval()
model.target_model.config._attn_implementation = "eager"  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

# %%

tmodel = runtime_cast(LlamaForCausalLM, model.target_model)
# %%

train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=task_config.max_seq_len,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=task_config.shuffle_each_epoch,
    seed=None,
)


tokenizer: Any
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_loader_iter = iter(data_loader)


pgd_config = PGDReconLossConfig(
    init="random",
    step_size=0.1,
    n_steps=20,
    mask_scope="shared_across_batch",
)

# %%


def detach(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def sort_layer(lname: str):
    if lname == "lm_head":
        return 10_000
    if lname == "model.norm":
        return 9_999

    assert lname.startswith("model.layers."), f"Unknown layer: {lname}"
    lname = lname.removeprefix("model.layers.")

    if "." in lname:
        layer_str, rest = lname.split(".", maxsplit=1)
        layer = int(layer_str)
        assert layer in [0, 1, 2, 3], f"Unknown layer: {lname}"
        if rest == "input_layernorm":
            return (layer * 1000) + 1
        if rest == "self_attn.q_proj":
            return (layer * 1000) + 2
        if rest == "self_attn.k_proj":
            return (layer * 1000) + 3
        if rest == "self_attn.v_proj":
            return (layer * 1000) + 4
        if rest == "self_attn.o_proj":
            return (layer * 1000) + 5
        if rest == "post_attention_layernorm":
            return (layer * 1000) + 6
        if rest == "mlp.up_proj":
            return (layer * 1000) + 7
        if rest == "mlp.gate_proj":
            return (layer * 1000) + 7
        if rest == "mlp.down_proj":
            return (layer * 1000) + 7

    return (int(lname) * 1000) + 8


def mse(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).pow(2).mean(dim=-1)


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def l2_norm_ratio(a: Tensor, b: Tensor) -> Tensor:
    return torch.norm(a, dim=-1) / torch.norm(b, dim=-1)


def pw_cos(x: Tensor) -> Tensor:
    return F.cosine_similarity(x[None], x[:, None], dim=-1)


def get_layer_outputs(
    batch: Int[Tensor, "1 seq"],
    paths: list[str],
    mask_infos: dict[str, ComponentsMaskInfo] | None = None,
) -> dict[str, Tensor]:
    with model.cache_modules(paths) as (_, output_cache):
        model(batch, mask_infos=mask_infos)
    layers = {}
    for path in paths:
        layers[path] = output_cache[path]
    return layers


# def get_logits_lens_toks_and_projections(
#     vectors: dict[str, Float[Tensor, "s d"]], topk: int = 10
# ) -> dict[str, list[list[tuple[int, float]]]]:
#     assert next(iter(vectors.values())).ndim == 2, "Expected 2D tensor"
#     logits: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
#     for layer, seq_vec in vectors.items():
#         for tok_vec in seq_vec:
#             # raise ValueError("check this is correct")
#             top_logits = runtime_cast(Tensor, tmodel.lm_head(tmodel.model.norm(tok_vec))).topk(  # pyright: ignore[reportCallIssue, reportArgumentType]
#                 topk, dim=-1
#             )  # pyright: ignore[reportCallIssue, reportArgumentType]  # noqa: F821
#             logits[layer].append(
#                 list(zip(top_logits.indices.tolist(), top_logits.values.tolist(), strict=True))
#             )
#     return logits


def plot_conjoined_pairwise_heatmap(
    title: str,
    target_outputs: Tensor,
    pgd_outputs: Tensor,
    labels: list[str],
    loss_per_position: list[float],
    # imshow_kwargs: dict[str, Any],
) -> go.Figure:
    labels = [f"{idx}: [{tok}]" for idx, tok in enumerate(labels + labels)]  #

    # eos_labels = [
    #     axis_label
    #     for axis_label, token in zip(labels, labels, strict=True)
    #     if "[EOS]" in token.upper()
    # ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        specs=[[{"type": "heatmap"}, {"type": "xy"}]],
        subplot_titles=["conjoined pairwise", "PGD loss"],
        horizontal_spacing=0.02,
    )

    assert target_outputs.shape == pgd_outputs.shape

    joined_matrix = torch.cat([target_outputs, pgd_outputs], dim=0)
    pw_matrix = pw_cos(joined_matrix)
    px_np = detach(pw_matrix)

    heatmap = go.Heatmap(
        z=px_np,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        hovertemplate=(
            "<b>Row Position:</b> %{y}<br>"
            "<b>Column Position:</b> %{x}<br>"
            "<b>Value:</b> %{z:.4f}"
            "<extra></extra>"
        ),
    )
    fig.add_trace(heatmap, row=1, col=1)
    # fig.update_xaxes(title_text="Column Position", row=1, col=1)
    # fig.update_yaxes(
    #     title_text=f"{name} Row Position" if col_idx == 1 else None,
    #     showticklabels=(col_idx == 1),
    #     matches="y1" if col_idx > 1 else None,
    #     row=1,
    #     col=col_idx,
    # )

    # for col_idx in (1, 2):
    #     for label in eos_x_axis_labels:
    #         fig.add_vline(
    #             x=label,
    #             line_color="orange",
    #             line_dash="dash",
    #             line_width=1.5,
    #             row=1,  # pyright: ignore[reportArgumentType]
    #             col=col_idx,  # pyright: ignore[reportArgumentType]
    #         )

    # columns_for_y = (1, 2, 3)
    # for col_idx in columns_for_y:
    #     for label in eos_y_axis_labels:
    #         fig.add_hline(
    #             y=label,
    #             line_color="green",
    #             line_dash="dash",
    #             line_width=1.5,
    #             row=1,  # pyright: ignore[reportArgumentType]
    #             col=col_idx,  # pyright: ignore[reportArgumentType]
    #         )

    # Add rotated line plot (horizontal) aligned with PGD heatmap rows
    line_trace = go.Scatter(
        x=loss_per_position,
        y=labels,
        mode="lines+markers",
        name="PGD loss",
        line=dict(color="black"),
        marker=dict(size=6),
        hovertemplate="<b>Token:</b> %{y}<br><b>PGD loss:</b> %{x:.4f}<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(line_trace, row=1, col=2)
    fig.update_yaxes(matches="y1", row=1, col=2, showticklabels=False)
    fig.update_xaxes(title_text="pgd loss", row=1, col=2)

    fig.update_layout(
        title=title,
        width=1600,
        height=750,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_pairwise_heatmap(
    title: str,
    target_matrix: Tensor,
    pgd_matrix: Tensor,
    loss_per_position: list[float],
    labels: list[str],
    imshow_kwargs: dict[str, Any],
) -> go.Figure:
    x_axis_labels = [f"{idx}: [{tok}]" for idx, tok in enumerate(labels)]
    y_axis_labels = [f"{idx}: [{tok}]" for idx, tok in enumerate(labels)]

    def _is_eos_token(token: str) -> bool:
        token_upper = token.upper()
        return "[EOS]" in token_upper

    eos_x_axis_labels = [
        axis_label
        for axis_label, token in zip(x_axis_labels, labels, strict=True)
        if _is_eos_token(token)
    ]
    eos_y_axis_labels = [
        axis_label
        for axis_label, token in zip(y_axis_labels, labels, strict=True)
        if _is_eos_token(token)
    ]

    colorscale = imshow_kwargs.get("cmap", "RdBu")
    zmin = imshow_kwargs.get("vmin")
    zmax = imshow_kwargs.get("vmax")

    matrices_to_plot = [
        ("Target pairwise", target_matrix),
        ("PGD pairwise", pgd_matrix),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.43, 0.43, 0.14],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "xy"}]],
        subplot_titles=[name for name, _ in matrices_to_plot] + ["PGD loss"],
        horizontal_spacing=0.02,
    )

    for col_idx, (name, matrix) in enumerate(matrices_to_plot, start=1):
        matrix_np = detach(matrix)
        heatmap = go.Heatmap(
            z=matrix_np,
            # x=x_axis_labels,
            # y=y_axis_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=(
                "<b>Row Position:</b> %{y}<br>"
                "<b>Column Position:</b> %{x}<br>"
                "<b>Value:</b> %{z:.4f}"
                "<extra></extra>"
            ),
        )
        fig.add_trace(heatmap, row=1, col=col_idx)
        fig.update_xaxes(title_text=f"{name} Column Position", row=1, col=col_idx)
        fig.update_yaxes(
            title_text=f"{name} Row Position" if col_idx == 1 else None,
            showticklabels=(col_idx == 1),
            matches="y1" if col_idx > 1 else None,
            row=1,
            col=col_idx,
        )

    for col_idx in (1, 2):
        for label in eos_x_axis_labels:
            fig.add_vline(
                x=label,
                line_color="orange",
                line_dash="dash",
                line_width=1.5,
                row=1,  # pyright: ignore[reportArgumentType]
                col=col_idx,  # pyright: ignore[reportArgumentType]
            )

    columns_for_y = (1, 2, 3)
    for col_idx in columns_for_y:
        for label in eos_y_axis_labels:
            fig.add_hline(
                y=label,
                line_color="green",
                line_dash="dash",
                line_width=1.5,
                row=1,  # pyright: ignore[reportArgumentType]
                col=col_idx,  # pyright: ignore[reportArgumentType]
            )

    # Add rotated line plot (horizontal) aligned with PGD heatmap rows
    line_trace = go.Scatter(
        x=loss_per_position,
        y=y_axis_labels,
        mode="lines+markers",
        name="PGD loss",
        line=dict(color="black"),
        marker=dict(size=6),
        hovertemplate="<b>Token:</b> %{y}<br><b>PGD loss:</b> %{x:.4f}<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(line_trace, row=1, col=3)
    fig.update_yaxes(matches="y1", row=1, col=3, showticklabels=False)
    fig.update_xaxes(title_text="pgd loss", row=1, col=3)

    fig.update_layout(
        title=title,
        width=1600,
        height=750,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def get_attn_patterns(model: LlamaModel, q_out: Tensor, k_out: Tensor) -> list[np.ndarray]:
    num_heads = model.config.num_attention_heads
    num_kv_heads = cast(int, model.config.num_key_value_heads)
    head_dim = cast(int, model.config.head_dim)
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    repeat_factor = num_heads // num_kv_heads
    seq_len = q_out.shape[1]
    assert q_out.shape == (1, seq_len, num_heads * head_dim)
    assert k_out.shape == (1, seq_len, num_kv_heads * head_dim)

    q_out_heads = einops.rearrange(q_out, "1 s (h d) -> s h d", d=head_dim)
    k_out_heads = einops.rearrange(k_out, "1 s (h d) -> s h d", d=head_dim)
    k_repeated_heads = einops.repeat(k_out_heads, "s h d -> s (h n) d", n=repeat_factor)

    assert q_out_heads.shape == (seq_len, num_heads, head_dim)
    assert k_repeated_heads.shape == (seq_len, num_heads, head_dim)

    causal_mask = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    layer_heads: list[np.ndarray] = []
    for head in range(num_heads):
        head_q_out = q_out_heads[:, head]
        head_k_out = k_repeated_heads[:, head]

        attention_scores = einops.einsum(head_q_out, head_k_out, "sq d, sk d -> sq sk")
        attention_scores = attention_scores / (head_dim**0.5)
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        attention_weights: Tensor = attention_scores.softmax(dim=-1)

        layer_heads.append(detach(attention_weights))

    return layer_heads


def get_layer_attn_patterns(
    batch: Int[Tensor, "1 seq"],
    mask_infos: dict[str, ComponentsMaskInfo] | None,
):
    assert batch.ndim == 2, "batch must be of shape (1, seq_len)"
    assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"

    with model.cache_attn_weights() as attn_weights:
        model(batch, mask_infos=mask_infos)

    return attn_weights


def _attention_heatmap_trace(
    pattern: np.ndarray,
    tokens: list[str],
    *,
    show_colorbar: bool,
) -> go.Heatmap:
    axis_labels = [f"{i}: [{token}]" for i, token in enumerate(tokens)]
    max_abs = float(np.abs(pattern).max())
    return go.Heatmap(
        z=pattern[::-1],
        x=axis_labels,
        y=axis_labels[::-1],
        colorscale="RdBu",
        zmax=max_abs,
        zmin=-max_abs,
        showscale=show_colorbar,
        colorbar=dict(title="Attention weight") if show_colorbar else None,
    )


def plot_attention_pattern(
    pattern: np.ndarray,
    tokens: list[str],
    *,
    as_trace: bool = False,
) -> go.Figure | go.Heatmap:
    heatmap = _attention_heatmap_trace(
        pattern=pattern,
        tokens=tokens,
        show_colorbar=not as_trace,
    )

    if as_trace:
        return heatmap

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Attention pattern",
        width=1000,
        height=1000,
        xaxis=dict(title="Query token"),
        yaxis=dict(title="Key token"),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def attention_pattern_grid(
    *,
    tokens: list[str],
    patterns_by_layer: Sequence[Sequence[np.ndarray]],
    # layers: Sequence[int],
    title: str,
    max_heads: int | None = 4,
):
    assert patterns_by_layer
    # assert len(patterns_by_layer) == len(layers), "layers and patterns must align"

    total_heads = len(patterns_by_layer[0])
    heads_to_plot = total_heads if max_heads is None else min(max_heads, total_heads)
    layer_indices = list(range(len(patterns_by_layer)))
    head_indices = list(range(heads_to_plot))

    # num_rows = len(layers)
    num_rows = len(patterns_by_layer)
    num_cols = len(head_indices)

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[
            f"L{layer_idx} · H{head}" for layer_idx in layer_indices for head in head_indices
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    for row_idx, (layer_idx, layer_patterns) in enumerate(
        zip(layer_indices, patterns_by_layer, strict=True)
    ):
        for col_idx, head_idx in enumerate(head_indices):
            assert head_idx < len(layer_patterns), (
                f"Head {head_idx} not found for layer {layer_idx}"
            )

            trace = plot_attention_pattern(
                layer_patterns[head_idx],
                tokens,
                as_trace=True,
            )
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

            fig.update_xaxes(
                showticklabels=False,
                title_text=f"Head {head_idx}" if row_idx == num_rows - 1 else None,
                row=row_idx + 1,
                col=col_idx + 1,
            )
            fig.update_yaxes(
                showticklabels=False,
                title_text=f"Layer {layer_idx}" if col_idx == 0 else None,
                row=row_idx + 1,
                col=col_idx + 1,
            )

    fig.update_layout(
        title=title,
        height=max(300, 250 * num_rows),
        width=max(400, 250 * num_cols),
        showlegend=False,
    )
    return fig


def pgd_geometry(
    batch: Int[Tensor, "1 seq"],
    pgd_mask_infos: dict[str, ComponentsMaskInfo],
    adv_sources: Float[Tensor, "n_layers *batch_dims C2"],
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
) -> None:
    nlayers = adv_sources.shape[0]
    fig = make_subplots(
        rows=nlayers,
        cols=6,
        column_widths=[0.45, 0.45, 0.1, 0.1, 0.1, 0.3],
        subplot_titles=["U", "V", "Source", "Output Mag", "Avg Act"],
    )

    # populate cache
    model(batch, mask_infos=pgd_mask_infos)

    for layer_idx, module_name in enumerate(ci):
        component = model.components[module_name]
        assert (C := component.C) == model.C
        u = component.U
        v = component.V
        inner_act = runtime_cast(Tensor, component._inner_acts_cache)
        assert inner_act.shape == (1, 512, C), (
            f"inner_act must be of shape (1, 1, C), got {inner_act.shape}"
        )

        source = adv_sources[layer_idx, ..., :-1]
        assert source.shape == (1, 1, C), f"source must be of shape (1, 1, C), got {source.shape}"
        source = source[0, 0]

        sorted_indices = source.argsort(dim=0)
        assert sorted_indices.shape == (C,), "should be sorted across components"

        sorted_u = u[sorted_indices]
        sorted_v = v.permute(1, 0)[sorted_indices]

        u_pw_cos_sim = pw_cos(sorted_u)
        v_pw_cos_sim = pw_cos(sorted_v)

        # avg_act = detach(inner_act[0].mean(dim=0)[sorted_indices])

        effective_act = inner_act * u.norm(p=2, dim=-1)
        output_mags = detach(effective_act[0].mean(dim=0)[sorted_indices])

        plotly_idx = layer_idx + 1
        u_heatmap = go.Heatmap(z=detach(u_pw_cos_sim.abs()), colorscale="Blues", zmin=0, zmax=1)
        fig.add_trace(u_heatmap, row=plotly_idx, col=1)
        fig.update_yaxes(title_text=f"{module_name} U", row=plotly_idx, col=1)
        v_heatmap = go.Heatmap(z=detach(v_pw_cos_sim.abs()), colorscale="RdBu", zmin=-1, zmax=1)
        fig.add_trace(v_heatmap, row=plotly_idx, col=2)
        fig.update_yaxes(title_text=f"{module_name} V", row=plotly_idx, col=2)

        # sources = go.Scatter(
        #     x=detach(source[sorted_indices]), y=np.arange(C), mode="lines", name="source"
        # )
        # fig.add_trace(sources, row=plotly_idx, col=3)
        # fig.update_yaxes(title_text=f"{module_name} Source", row=plotly_idx, col=3)

        # output_mags = go.Scatter(x=output_mags, y=np.arange(C), mode="lines", name="output_mags")
        # fig.add_trace(output_mags, row=plotly_idx, col=4)
        # fig.update_yaxes(title_text=f"{module_name} Output Mag", row=plotly_idx, col=4)

        # avg_act = go.Scatter(x=avg_act, y=np.arange(C), mode="lines", name="avg_act")
        # fig.add_trace(avg_act, row=plotly_idx, col=5)
        # fig.update_yaxes(title_text=f"{module_name} Avg Act", row=plotly_idx, col=4)

        # scatter plot effective act vs source:
        scatter = go.Scatter(
            x=output_mags, y=detach(source), mode="markers", name="effective_act vs source"
        )
        fig.add_trace(scatter, row=plotly_idx, col=6)
        fig.update_yaxes(title_text="asdf", row=plotly_idx, col=6)

    fig.update_layout(
        title="Pairwise Cosine Similarities (sorted by pgd mask value)",
        width=4000,
        height=1300 * nlayers,
        showlegend=False,
    )
    fig.show(renderer="browser")


def pick_keys[V](d: dict[str, V], filter_fn: Callable[[str], bool]) -> dict[str, V]:
    return {k: v for k, v in d.items() if filter_fn(k)}


@dataclass
class pw_cfg:
    filter_fn: Callable[[str], bool] | None = None


@dataclass
class HeatmapMetrics:
    name: str
    z: np.ndarray
    y: list[str]
    kwargs: dict[str, Any]
    highlight_y_labels: Callable[[str], bool] | None = None


def visualize_metrics_and_heatmaps(
    line_metrics: dict[str, np.ndarray],
    heatmaps: list[HeatmapMetrics],
    x_labels: list[str],
    title: str,
) -> go.Figure:
    n_heatmaps = len(heatmaps)
    n_line_plots = 1  # all lines are overlaid

    fig = make_subplots(
        rows=n_line_plots + n_heatmaps,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        # row_heights=[0.2, 0.4, 0.4],
    )

    # Add all line metrics to the first subplot
    for name, val in line_metrics.items():
        fig.update_yaxes(title_text=name, row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_labels, y=val, mode="lines+markers", name=name),
            row=1,
            col=1,
        )

    for i, heatmap in enumerate(heatmaps, start=2):
        trace = go.Heatmap(
            z=heatmap.z,
            y=heatmap.y,
            x=x_labels,
            hoverongaps=False,
            showscale=False,
            **heatmap.kwargs,
        )
        fig.add_trace(trace, row=i, col=1)
        if heatmap.highlight_y_labels:
            for y_label in heatmap.y:
                if heatmap.highlight_y_labels(y_label):
                    fig.add_hline(
                        y=y_label,
                        line_color="orange",
                        line_dash="dash",
                        line_width=1.5,
                        row=i,  # pyright: ignore[reportArgumentType]
                        col=1,  # pyright: ignore[reportArgumentType]
                    )
        fig.update_yaxes(title_text=heatmap.name, row=i, col=1)

    fig.update_xaxes(title_text="Token Position", row=n_line_plots + n_heatmaps, col=1)

    fig.update_layout(
        title=title,
        width=8000,
        height=2600,
        showlegend=True,
    )

    return fig


@dataclass
class DoSeqHeatmap:
    title: str
    apply_layer_pgd_mask: Callable[[str], bool] | None = None


# Custom colorscale: High fidelity 0.0 -> 0.1 (White -> Blue), then Differentiate 0.1 -> 1.0 (Blue -> Red)
WBR = [
    [0.0, "#ffffff"],  # White
    [0.02, "#d0d1e6"],  # Faint Blue-Purple
    [0.05, "#a6bddb"],  # Light Blue
    [0.1, "#0570b0"],  # Strong Blue at 0.1
    [0.3, "#d7301f"],  # Transition to Red
    [0.6, "#b30000"],  # Dark Red
    [1.0, "#7f0000"],  # Deepest Red
]

INV_COS = [
    [0.0, "#f04040"],
    [0.5, "#4040f0"],
    [1.0, "#ffffff"],
]


def run_batch(
    batch: Int[Tensor, "1 seq"],
    target_output: Float[Tensor, "1 seq vocab"],
    pgd_mask_infos: dict[str, ComponentsMaskInfo],
    do_seq_heatmap: DoSeqHeatmap | Literal[False] = False,
    do_pw_rep_sim: pw_cfg | Literal[False] = False,
    do_patterns: bool = False,
) -> None:
    actual_seq_toks = [tokenizer.decode(tok) for tok in batch[0]]

    pgd_out = model(batch, mask_infos=pgd_mask_infos)
    pgd_loss = calc_kl_divergence_lm(pgd_out, target_output, reduce=False)

    assert pgd_loss.shape == (1, len(actual_seq_toks)), "pgd_loss must be of shape (1, seq_len)"
    pgd_loss_np = detach(pgd_loss[0])

    resid_paths = get_target_module_paths(
        model.target_model,
        [
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.down_proj",
            "model.layers.*.mlp.gate_proj",
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.norm",
        ],
    )

    hidden_paths = get_target_module_paths(
        model.target_model,
        [
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.down_proj",
            "model.layers.*.mlp.gate_proj",
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.norm",
        ],
    )
    all_paths = list(set(resid_paths).union(set(hidden_paths)))

    all_target_outputs = {k: v[0] for k, v in get_layer_outputs(batch, all_paths).items()}
    all_pgd_outputs = {
        k: v[0] for k, v in get_layer_outputs(batch, all_paths, pgd_mask_infos).items()
    }

    if do_seq_heatmap:
        sorted_resid_paths = sorted(resid_paths, key=sort_layer)

        cossim_matrix = np.stack(
            [
                detach(
                    F.cosine_similarity(all_target_outputs[layer], all_pgd_outputs[layer], dim=-1)
                )
                for layer in sorted_resid_paths
            ]
        )

        cossim_heatmap = HeatmapMetrics(
            name="Cosine Similarity (all pgd masks)",
            z=cossim_matrix,
            y=sorted_resid_paths,
            kwargs={"colorscale": "RdBu", "zmin": -1, "zmax": 1},
        )

        # altered cossim heatmap
        filtered_pgd_mask_infos = pick_keys(
            pgd_mask_infos, do_seq_heatmap.apply_layer_pgd_mask or (lambda _: True)
        )
        filtered_pgd_outputs = {
            k: v[0] for k, v in get_layer_outputs(batch, all_paths, filtered_pgd_mask_infos).items()
        }

        filtered_cossim_matrix = np.stack(
            [
                detach(
                    F.cosine_similarity(
                        all_target_outputs[layer], filtered_pgd_outputs[layer], dim=-1
                    )
                )
                for layer in sorted_resid_paths
            ]
        )

        filtered_cossim_heatmap = HeatmapMetrics(
            name="Cosine Similarity (filtered pgd masks)",
            z=filtered_cossim_matrix,
            y=sorted_resid_paths,
            highlight_y_labels=do_seq_heatmap.apply_layer_pgd_mask,
            kwargs={"colorscale": "RdBu", "zmin": -1, "zmax": 1},
        )

        # Add attention means
        attn_weights = get_layer_attn_patterns(batch, mask_infos=None)

        pgd_attn_weights = get_layer_attn_patterns(batch, mask_infos=filtered_pgd_mask_infos)

        attn_means_rows = []
        attn_means_labels = []
        for layer_path in attn_weights:
            diff = pgd_attn_weights[layer_path] - attn_weights[layer_path]
            layer_idx = layer_path.split(".")[2]
            assert diff.shape[1] == 4, "Expected 4 heads per layer"
            for h in range(4):
                attn_means_rows.append(detach(diff[0, h].mean(dim=0)))
                attn_means_labels.append(f"L{layer_idx}H{h}")

        attn_heatmap = HeatmapMetrics(
            name="Mean Attention Mass",
            z=np.stack(attn_means_rows),
            y=attn_means_labels,
            kwargs={"colorscale": WBR, "zmin": 0, "zmax": 1},
        )

        filtered_pgd_out = model(batch, mask_infos=filtered_pgd_mask_infos)
        filtered_pgd_loss = calc_kl_divergence_lm(filtered_pgd_out, target_output, reduce=False)[0]
        filtered_pgd_loss_np = detach(filtered_pgd_loss)

        visualize_metrics_and_heatmaps(
            title=do_seq_heatmap.title,
            line_metrics={"PGD Loss": pgd_loss_np, "Filtered PGD Loss": filtered_pgd_loss_np},
            heatmaps=[filtered_cossim_heatmap, attn_heatmap, cossim_heatmap],
            x_labels=[f"{i}: [{token}]" for i, token in enumerate(actual_seq_toks)],
        ).show(renderer="browser")

    pairwise_cos_sim_val = {
        layer: (pw_cos(all_target_outputs[layer]), pw_cos(all_pgd_outputs[layer]))
        for layer in hidden_paths
    }

    if do_pw_rep_sim:
        target_matrices = {k: v for k, (v, _) in pairwise_cos_sim_val.items()}
        pgd_matrices = {k: v for k, (_, v) in pairwise_cos_sim_val.items()}
        assert set(target_matrices) == set(pgd_matrices)
        sorted_keys = sorted(target_matrices.keys(), key=sort_layer)
        loss_per_position = pgd_loss_np.tolist()
        for layer in sorted_keys:
            if do_pw_rep_sim.filter_fn is not None and not do_pw_rep_sim.filter_fn(layer):
                continue
            # plot_pairwise_heatmap(
            #     title=f"Pairwise cos_similarity across sequence - {layer}",
            #     target_matrix=target_matrices[layer],
            #     pgd_matrix=pgd_matrices[layer],
            #     loss_per_position=loss_per_position,
            #     labels=actual_seq_toks,
            #     imshow_kwargs={"cmap": "RdBu", "vmin": -1, "vmax": 1},
            # ).show(renderer="browser")
            plot_conjoined_pairwise_heatmap(
                title=f"Pairwise cos_similarity across sequence - {layer}",
                target_outputs=all_target_outputs[layer],
                pgd_outputs=all_pgd_outputs[layer],
                labels=actual_seq_toks,
                loss_per_position=loss_per_position,
            ).show(renderer="browser")

    if do_patterns:
        target_attn_patternss_d = get_layer_attn_patterns(batch, mask_infos=None)
        pgd_attn_patternss_d = get_layer_attn_patterns(batch, mask_infos=pgd_mask_infos)
        tokens = [tokenizer.decode(tok) for tok in batch[0]]

        start, end = 0, 512
        tokens = tokens[start:end]
        target_attn_patternss = [
            [detach(pattern[start:end, start:end]) for pattern in patterns[0].unbind()]
            for _layer, patterns in target_attn_patternss_d.items()
        ]
        pgd_attn_patternss = [
            [detach(pattern[start:end, start:end]) for pattern in patterns[0].unbind()]
            for _layer, patterns in pgd_attn_patternss_d.items()
        ]
        diff_attn_patternss = [
            [(p - t) for p, t in zip(ps, ts, strict=True)]
            for ps, ts in zip(pgd_attn_patternss, target_attn_patternss, strict=True)
        ]

        attention_pattern_grid(
            tokens=tokens[start:end],
            patterns_by_layer=target_attn_patternss,
            title="Target attention patterns",
        ).show()

        attention_pattern_grid(
            tokens=tokens[start:end],
            patterns_by_layer=pgd_attn_patternss,
            title="PGD attention patterns",
        ).show()

        attention_pattern_grid(
            tokens=tokens[start:end],
            patterns_by_layer=diff_attn_patternss,
            title="PGD attention patterns",
        ).show(renderer="browser")


def is_qk_layer(x: str) -> bool:
    return any(fnmatch(x, matcher) for matcher in ["*self_attn.q_proj", "*self_attn.k_proj"])


def is_mlp_down_proj_layer(x: str) -> bool:
    return fnmatch(x, "*mlp.down_proj")


def is_attn_layer(x: str) -> bool:
    return fnmatch(x, "*self_attn*")


def inv(f: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda x: not f(x)


# %%

_batch = extract_batch_data(next(data_loader_iter)).to(device)[0:1]

_target_output = model(_batch, cache_type="input")

_ci = model.calc_causal_importances(
    pre_weight_acts=_target_output.cache,
    detach_inputs=False,
    sampling=config.sampling,
)

_, _, _pgd_mask_infos, _, _adv_sources = pgd_masked_recon_loss_update(
    model=model,
    batch=_batch,
    ci=_ci.lower_leaky,
    weight_deltas=model.calc_weight_deltas(),
    target_out=_target_output.output,
    output_loss_type=config.output_loss_type,
    routing="all",
    pgd_config=pgd_config,
)


# %%

configs = [
    DoSeqHeatmap(title="all"),
    DoSeqHeatmap(title="no qk", apply_layer_pgd_mask=inv(is_qk_layer)),
    # DoSeqHeatmap(title="qk only", apply_layer_pgd_mask=is_qk_layer),
    # DoSeqHeatmap(title="no mlp down proj", apply_layer_pgd_mask=inv(is_mlp_down_proj_layer)),
    # DoSeqHeatmap(title="mlp down proj only", apply_layer_pgd_mask=is_mlp_down_proj_layer),
    # DoSeqHeatmap(title="Attention only", apply_layer_pgd_mask=is_attn_layer),
    # DoSeqHeatmap(title="No Attention", apply_layer_pgd_mask=inv(is_attn_layer)),
]
for s_config in configs:
    run_batch(
        batch=_batch,
        target_output=_target_output.output,
        pgd_mask_infos=_pgd_mask_infos,
        do_seq_heatmap=s_config,
    )
# %%


pgd_geometry(_batch, _pgd_mask_infos, _adv_sources, _ci.lower_leaky)
# %%
import plotly.express as px

u_norms = model.components["model.layers.0.self_attn.q_proj"].U.norm(p=2, dim=1)
v_norms = model.components["model.layers.0.self_attn.q_proj"].V.norm(p=2, dim=0)

px.histogram(detach(u_norms * v_norms), nbins=1000).show(renderer="browser")
# %%

for _ in range(10):
    _batch = extract_batch_data(next(data_loader_iter)).to(device)[0:1]

    _target_output = model(_batch, cache_type="input")

    _ci = model.calc_causal_importances(
        pre_weight_acts=_target_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    _, _, _pgd_mask_infos, _, _adv_sources = pgd_masked_recon_loss_update(
        model=model,
        batch=_batch,
        ci=_ci.lower_leaky,
        weight_deltas=model.calc_weight_deltas(),
        target_out=_target_output.output,
        output_loss_type=config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
    )

    run_batch(
        batch=_batch,
        target_output=_target_output.output,
        pgd_mask_infos=_pgd_mask_infos,
        # do_seq_heatmap=DoSeqHeatmap(title="all"),
        do_pw_rep_sim=pw_cfg(),
        # do_patterns=True,
    )
    break

# %%


_batch = extract_batch_data(next(data_loader_iter)).to(device)[0:1]

_target_output = model(_batch, cache_type="input")

_ci = model.calc_causal_importances(
    pre_weight_acts=_target_output.cache,
    detach_inputs=False,
    sampling=config.sampling,
)

_, _, _pgd_mask_infos, _, _ = pgd_masked_recon_loss_update(
    model=model,
    batch=_batch,
    ci=_ci.lower_leaky,
    weight_deltas=model.calc_weight_deltas(),
    target_out=_target_output.output,
    output_loss_type=config.output_loss_type,
    routing="all",
    pgd_config=pgd_config,
)

_pgd_outputs = model(_batch, mask_infos=_pgd_mask_infos)


# %%


def plot_aligned_logit_heatmaps(
    target_logits: Tensor,
    pgd_logits: Tensor,
    batch: Int[Tensor, "1 seq"],
    tokenizer: Any,
    title: str = "Top-K Token Logits Comparison",
    top_k: int = 10,
) -> go.Figure:
    target_ids = batch[0, 1:]
    t_logits_shifted = target_logits[0, :-1]
    p_logits_shifted = pgd_logits[0, :-1]

    # 1. Get "Correct" Token Logits (Target)
    # Shape: (seq_len,)
    t_correct = t_logits_shifted.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    p_correct = p_logits_shifted.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Shape: (1, seq_len)
    t_correct_np = detach(t_correct).reshape(1, -1)
    p_correct_np = detach(p_correct).reshape(1, -1)

    # 2. Get Top-K Logits
    t_topk = torch.topk(t_logits_shifted, k=top_k, dim=-1)
    p_topk = torch.topk(p_logits_shifted, k=top_k, dim=-1)

    # Shape: (top_k, seq_len)
    t_vals_np = detach(t_topk.values).T
    p_vals_np = detach(p_topk.values).T

    # 3. Combine Values (Correct row first)
    t_vals_final = np.vstack([t_correct_np, t_vals_np])
    p_vals_final = np.vstack([p_correct_np, p_vals_np])

    # 4. Prepare Text Labels
    # Correct token strings
    actual_tokens = tokenizer.batch_decode(target_ids)
    correct_text_row = np.array(actual_tokens).reshape(1, -1)

    # Top-K token strings
    t_indices = t_topk.indices
    p_indices = p_topk.indices
    t_tokens_flat = tokenizer.batch_decode(t_indices.transpose(0, 1).flatten())
    p_tokens_flat = tokenizer.batch_decode(p_indices.transpose(0, 1).flatten())

    t_text_matrix = np.array(t_tokens_flat).reshape(top_k, -1)
    p_text_matrix = np.array(p_tokens_flat).reshape(top_k, -1)

    # Combine Text
    t_text_final = np.vstack([correct_text_row, t_text_matrix])
    p_text_final = np.vstack([correct_text_row, p_text_matrix])

    # Axis Labels
    x_labels = [f"{i}: {tok}" for i, tok in enumerate(actual_tokens)]
    y_labels_topk = [str(i + 1) for i in range(top_k)]
    y_labels_correct = ["Target", "PGD"]

    # Data for the top heatmap (Correct Logits Comparison)
    # Row 1: Target Correct Logits
    # Row 2: PGD Correct Logits
    correct_vals_combined = np.vstack([t_correct_np, p_correct_np])
    correct_text_combined = np.vstack([correct_text_row, correct_text_row])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        # Row 1: Comparison of Correct Logits
        # Row 2: Target Top-K
        # Row 3: PGD Top-K
        subplot_titles=(
            "Logits of Ground Truth Token (Target vs PGD)",
            "Target Output Top-K Logits",
            "PGD Masked Top-K Logits",
        ),
        row_heights=[0.15, 0.425, 0.425],
    )

    # Global min/max for color scaling across all plots if desired,
    # or specific to each section. Let's use global for consistent color meaning.
    zmin = min(float(t_vals_final.min()), float(p_vals_final.min()))
    zmax = max(float(t_vals_final.max()), float(p_vals_final.max()))

    # 1. Correct Logits Heatmap
    fig.add_trace(
        go.Heatmap(
            z=correct_vals_combined,
            text=correct_text_combined,
            texttemplate="%{text}",
            x=x_labels,
            y=y_labels_correct,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="Logit", x=1.02, y=0.9, len=0.25),
        ),
        row=1,
        col=1,
    )

    # 2. Target Top-K Heatmap
    fig.add_trace(
        go.Heatmap(
            z=t_vals_np,
            text=t_text_matrix,
            texttemplate="%{text}",
            x=x_labels,
            y=y_labels_topk,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="Logit", x=1.02, y=0.5, len=0.35),
        ),
        row=2,
        col=1,
    )

    # 3. PGD Top-K Heatmap
    fig.add_trace(
        go.Heatmap(
            z=p_vals_np,
            text=p_text_matrix,
            texttemplate="%{text}",
            x=x_labels,
            y=y_labels_topk,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="Logit", x=1.02, y=0.1, len=0.35),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=max(800, (top_k * 2 + 4) * 40),
        width=max(1200, len(x_labels) * 30),
    )

    # Reverse Y-axes
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", title_text="Target Rank", row=2, col=1)
    fig.update_yaxes(autorange="reversed", title_text="PGD Rank", row=3, col=1)

    return fig


def analyze_logit_variation(
    target_logits: Tensor,
    pgd_logits: Tensor,
) -> None:
    """
    Analyzes the variation of logits across the sequence dimension.
    Hypothesis: PGD logits are less varied (more static) than target logits.
    Metrics:
    1. Mean Variance: Average variance of logit values across sequence (per vocab token).
    2. Entropy Std: Standard deviation of the entropy of the distribution over time.
    3. Temporal Smoothness: Mean cosine similarity between adjacent time steps.
    """
    # Use logits aligned with the sequence (ignoring the last shift for prediction alignment for now,
    # just treating them as a stream of vectors)
    # Shape: (seq, vocab)
    t_logits = target_logits[0]
    p_logits = pgd_logits[0]

    print("\n--- Logit Variation Analysis ---")

    # 1. Variance across sequence (averaged over vocabulary)
    # var(dim=0) gives variance for each token in vocab across the sequence
    t_seq_var = t_logits.var(dim=0).mean()
    p_seq_var = p_logits.var(dim=0).mean()
    print(
        f"Mean Logit Variance (seq dim):    Target={t_seq_var.item():.4f} | PGD={p_seq_var.item():.4f}"
    )

    # 2. Entropy Stats
    t_probs = t_logits.softmax(dim=-1)
    p_probs = p_logits.softmax(dim=-1)
    # Entropy = -sum(p * log(p))
    t_entropy = -(t_probs * t_probs.log()).sum(dim=-1)
    p_entropy = -(p_probs * p_probs.log()).sum(dim=-1)

    print(
        f"Entropy Mean:                     Target={t_entropy.mean().item():.4f} | PGD={p_entropy.mean().item():.4f}"
    )
    print(
        f"Entropy Std (Variation over seq): Target={t_entropy.std().item():.4f}  | PGD={p_entropy.std().item():.4f}"
    )

    # 3. Temporal Cosine Similarity (Auto-correlation lag 1)
    t_sim = F.cosine_similarity(t_logits[:-1], t_logits[1:], dim=-1).mean()
    p_sim = F.cosine_similarity(p_logits[:-1], p_logits[1:], dim=-1).mean()
    print(f"Adjacent Cosine Sim (Smoothness): Target={t_sim.item():.4f} | PGD={p_sim.item():.4f}")
    print("--------------------------------\n")

    # Plot Entropy and Max Logit trace
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Entropy over Time", "Max Logit Value over Time"),
    )

    fig.add_trace(
        go.Scatter(y=detach(t_entropy), name="Target Entropy", line=dict(color="blue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=detach(p_entropy), name="PGD Entropy", line=dict(color="red")), row=1, col=1
    )

    t_max = t_logits.max(dim=-1).values
    p_max = p_logits.max(dim=-1).values

    fig.add_trace(
        go.Scatter(y=detach(t_max), name="Target Max Logit", line=dict(color="blue", dash="dot")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=detach(p_max), name="PGD Max Logit", line=dict(color="red", dash="dot")),
        row=2,
        col=1,
    )

    fig.update_layout(title="Logit Dynamics across Sequence", height=600)
    fig.show(renderer="browser")


def plot_top_k_logit_scatter(
    target_logits: Tensor,
    pgd_logits: Tensor,
    batch: Int[Tensor, "1 seq"],
    top_k: int = 10,
    normalization: Literal["none", "centered", "softmax", "log_softmax"] = "log_softmax",
) -> go.Figure:
    """
    Scatter plot comparing logits for the Top-K tokens predicted by the Target model.
    X-axis: Target Logits
    Y-axis: PGD Logits (for the same tokens)

    normalization:
        - 'none': Raw logits.
        - 'centered': Logits minus mean over vocab (default). comparison invariant to shift.
        - 'softmax': Probabilities.
        - 'log_softmax': Log-probabilities.
    """
    target_ids = batch[0, 1:]
    t_logits_shifted = target_logits[0, :-1]
    p_logits_shifted = pgd_logits[0, :-1]

    # Apply normalization
    if normalization == "centered":
        t_logits_shifted = t_logits_shifted - t_logits_shifted.mean(dim=-1, keepdim=True)
        p_logits_shifted = p_logits_shifted - p_logits_shifted.mean(dim=-1, keepdim=True)
        val_label = "Centered Logits"
    elif normalization == "softmax":
        t_logits_shifted = t_logits_shifted.softmax(dim=-1)
        p_logits_shifted = p_logits_shifted.softmax(dim=-1)
        val_label = "Probabilities"
    elif normalization == "log_softmax":
        t_logits_shifted = t_logits_shifted.log_softmax(dim=-1)
        p_logits_shifted = p_logits_shifted.log_softmax(dim=-1)
        val_label = "Log Probabilities"
    else:
        val_label = "Raw Logits"

    # 1. Identify Top-K tokens according to Target model
    # Shape: (seq_len, top_k)
    t_topk = torch.topk(t_logits_shifted, k=top_k, dim=-1)
    topk_indices = t_topk.indices

    # 2. Gather logits for these specific tokens from both models
    # Shape: (seq_len, top_k)
    target_vals = t_logits_shifted.gather(-1, topk_indices)
    pgd_vals = p_logits_shifted.gather(-1, topk_indices)

    # 3. Flatten for scatter plot
    x_vals = detach(target_vals).flatten()
    y_vals = detach(pgd_vals).flatten()

    # Create colors based on rank (0 to top_k-1 repeated seq_len times)
    # We want to see if higher ranked tokens (e.g. rank 1) behave differently
    ranks = np.tile(np.arange(1, top_k + 1), (len(target_ids), 1)).flatten()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=6,
                color=ranks,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Target Rank"),
                opacity=0.6,
            ),
            text=[f"Rank {r}" for r in ranks],
            hovertemplate=f"<b>Target {val_label}:</b> %{{x:.4f}}<br><b>PGD {val_label}:</b> %{{y:.4f}}<br><b>Rank:</b> %{{text}}<extra></extra>",
        )
    )

    # Add y=x line for reference
    min_val = min(float(x_vals.min()), float(y_vals.min()))
    max_val = max(float(x_vals.max()), float(y_vals.max()))

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", dash="dash"),
    )

    fig.update_layout(
        title=f"Top-{top_k} Target Tokens: {val_label} Comparison",
        xaxis_title=f"Target {val_label}",
        yaxis_title=f"PGD {val_label}",
        height=800,
        width=800,
    )

    return fig


# plot_aligned_logit_heatmaps(
#     target_logits=_target_output.output,
#     pgd_logits=_pgd_outputs,
#     batch=_batch,
#     tokenizer=tokenizer,
# ).show(renderer="browser")

# analyze_logit_variation(_target_output.output, _pgd_outputs)

plot_top_k_logit_scatter(
    target_logits=_target_output.output,
    pgd_logits=_pgd_outputs,
    batch=_batch,
).show(renderer="browser")

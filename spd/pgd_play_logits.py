# ruff: noqa: E402, I001
# %%
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from collections import defaultdict
from collections.abc import Generator
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
from spd.utils.general_utils import extract_batch_data, runtime_cast
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


def get_logits_lens_toks_and_projections(
    vectors: dict[str, Float[Tensor, "s d"]], topk: int = 10
) -> dict[str, list[list[tuple[int, float]]]]:
    assert next(iter(vectors.values())).ndim == 2, "Expected 2D tensor"
    logits: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
    for layer, seq_vec in vectors.items():
        for tok_vec in seq_vec:
            # raise ValueError("check this is correct")
            top_logits = runtime_cast(Tensor, tmodel.lm_head(tmodel.model.norm(tok_vec))).topk(  # pyright: ignore[reportCallIssue, reportArgumentType]
                topk, dim=-1
            )  # pyright: ignore[reportCallIssue, reportArgumentType]  # noqa: F821
            logits[layer].append(
                list(zip(top_logits.indices.tolist(), top_logits.values.tolist(), strict=True))
            )
    return logits


def plot_pairwise_heatmap(
    title: str,
    target_matrix: Tensor,
    pgd_matrix: Tensor,
    loss_per_position: list[float],
    labels: list[str],
    imshow_kwargs: dict[str, Any],
) -> None:
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
        matrix_np = matrix.detach().cpu().numpy()
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

    if eos_x_axis_labels or eos_y_axis_labels:
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
                    line_color="orange",
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
    fig.show()


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

        layer_heads.append(attention_weights.detach().cpu().numpy())

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


@dataclass
class pw_cfg:
    filter_fn: Callable[[str], bool] | None = None


def run_batch(
    batch: Int[Tensor, "1 seq"],
    do_seq_heatmap: bool = False,
    do_pw: pw_cfg | Literal[False] = False,
    do_patterns: bool = False,
) -> None:
    actual_seq_toks = [tokenizer.decode(tok) for tok in batch[0]]

    target_output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    _, _, pgd_mask_infos, pgd_loss = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=model.calc_weight_deltas(),
        target_out=target_output.output,
        output_loss_type=config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
    )
    assert pgd_loss.shape == (1, len(actual_seq_toks)), "pgd_loss must be of shape (1, seq_len)"
    pgd_loss_per_token = pgd_loss[0].detach().cpu().numpy()

    resid_paths = get_target_module_paths(
        model.target_model,
        [
            "model.layers.*.self_attn.o_proj",
            "model.layers.*.mlp.down_proj",
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
            "model.layers.*.mlp.down_proj",
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
        target_outputs = {layer: all_target_outputs[layer] for layer in resid_paths}
        target_logits = get_logits_lens_toks_and_projections(target_outputs, topk=5)

        pgd_outputs = {layer: all_pgd_outputs[layer] for layer in resid_paths}
        pgd_logits = get_logits_lens_toks_and_projections(pgd_outputs, topk=5)

        diffs = {layer: pgd_outputs[layer] - target_outputs[layer] for layer in resid_paths}
        diff_logits = get_logits_lens_toks_and_projections(diffs, topk=5)

        hover_data = defaultdict[str, list[str]](list)
        for layer in diff_logits:
            for diff_data, target_data, pgd_data in zip(
                diff_logits[layer], target_logits[layer], pgd_logits[layer], strict=False
            ):
                lines = []
                for name, logit_lens_topk in [
                    ("target", target_data),
                    ("pgd", pgd_data),
                    ("diff", diff_data),
                ]:
                    lines.append(f"{name}:")
                    for tok_idx, prob in logit_lens_topk:
                        lines.append(f"{tokenizer.decode(tok_idx):<10} {prob:.2f}")
                    lines.append("")
                hover_data[layer].append("<br>".join(lines))

        # # Create a matrix of hover text strings
        # hover_matrix: list[list[str]] = []
        # for layer in sorted_resid_paths:
        #     assert layer in hover_data
        #     hover_matrix.append(hover_data[layer])
        # customdata = hover_matrix
        # hovertemplate = (
        #     "<b>Layer:</b> %{y}<br>"
        #     "<b>Position:</b> %{x}<br>"
        #     f"<b>{val_name}:</b> %{{z:.4f}}<br>"
        #     "<br>%{customdata}<extra></extra>"
        # )

        sorted_resid_paths = sorted(resid_paths, key=sort_layer)

        matrix_np = np.stack(
            [
                F.cosine_similarity(all_target_outputs[layer], all_pgd_outputs[layer], dim=-1)
                .detach()
                .cpu()
                .numpy()
                for layer in sorted_resid_paths
            ]
        )

        x_axis_labels = [f"{i}: [{token}]" for i, token in enumerate(actual_seq_toks)]
        heatmap = go.Heatmap(
            z=matrix_np,
            y=sorted_resid_paths,
            x=x_axis_labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            # customdata=customdata,
            # hovertemplate=hovertemplate,
            hoverongaps=False,
            showscale=False,
        )

        # loss_per_position = pgd_loss_per_token.tolist()

        # add this stuff as line plots at the top
        attn_weights = get_layer_attn_patterns(batch, mask_infos=None)
        pgd_attn_weights = get_layer_attn_patterns(batch, mask_infos=pgd_mask_infos)

        top_rows = np.stack(
            [
                pgd_loss_per_token,
                *[
                    head.mean(dim=0).detach().cpu().numpy()
                    for _layer, weights in attn_weights.items()
                    for head in weights[0]  # not sure, 0 or 1?
                ],
                *[
                    head.mean(dim=0).detach().cpu().numpy()
                    for _layer, weights in pgd_attn_weights.items()
                    for head in weights[0]  # not sure, 0 or 1?
                ],
            ]
        )
        print(top_rows.shape)

        top_row_labels = [
            "pgd loss",
            *[f"{layer} head {head} attn weight" for layer in attn_weights for head in range(4)],
            *[
                f"{layer} head {head} attn weight (pgd)"
                for layer in pgd_attn_weights
                for head in range(4)
            ],
        ]

        assert top_rows.shape[1] == len(actual_seq_toks)

        fig = make_subplots(
            rows=len(top_rows) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[(0.75 / len(top_rows)) for _ in range(len(top_rows))] + [0.25],
        )
        for idx, (row, label) in enumerate(zip(top_rows, top_row_labels, strict=True)):
            fig.add_trace(
                go.Scatter(
                    x=x_axis_labels,
                    y=row,
                    mode="lines+markers",
                    name=label,
                ),
                row=idx + 1,
                col=1,
            )

        fig.add_trace(heatmap, row=2, col=1)
        fig.update_yaxes(title_text="PGD Loss", row=1, col=1)
        fig.update_yaxes(title_text="Layer", row=2, col=1)
        fig.update_xaxes(title_text="Token Position", row=2, col=1)
        fig.update_layout(
            width=4000,
            height=2000,
        )

        fig.show()

    pairwise_cos_sim_val = {
        layer: (pw_cos(all_target_outputs[layer]), pw_cos(all_pgd_outputs[layer]))
        for layer in hidden_paths
    }

    stride = 1
    if do_pw:
        target_matrices = {k: v[::stride, ::stride] for k, (v, _) in pairwise_cos_sim_val.items()}
        pgd_matrices = {k: v[::stride, ::stride] for k, (_, v) in pairwise_cos_sim_val.items()}
        labels = actual_seq_toks[::stride]
        assert set(target_matrices) == set(pgd_matrices)
        sorted_keys = sorted(target_matrices.keys(), key=sort_layer)
        loss_per_position = pgd_loss_per_token[::stride].tolist()
        for layer in sorted_keys:
            if do_pw.filter_fn is not None and not do_pw.filter_fn(layer):
                continue

            plot_pairwise_heatmap(
                title=f"Pairwise cos_similarity across sequence - {layer}",
                target_matrix=target_matrices[layer],
                pgd_matrix=pgd_matrices[layer],
                loss_per_position=loss_per_position,
                labels=labels,
                imshow_kwargs={"cmap": "RdBu", "vmin": -1, "vmax": 1},
            )

    if do_patterns:
        # raise NotImplementedError("Not implemented")
        target_attn_patternss_d = get_layer_attn_patterns(batch, mask_infos=None)
        pgd_attn_patternss_d = get_layer_attn_patterns(batch, mask_infos=pgd_mask_infos)
        tokens = [tokenizer.decode(tok) for tok in batch[0]]

        start, end = 0, 512
        tokens = tokens[start:end]
        target_attn_patternss = [
            [
                pattern[start:end, start:end].detach().cpu().numpy()
                for pattern in patterns[0].unbind()
            ]
            for _layer, patterns in target_attn_patternss_d.items()
        ]
        pgd_attn_patternss = [
            [
                pattern[start:end, start:end].detach().cpu().numpy()
                for pattern in patterns[0].unbind()
            ]
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
        ).show()


# %%

batch = extract_batch_data(next(data_loader_iter)).to(device)[0:1]

# %%

run_batch(
    batch=batch,
    do_seq_heatmap=True,
    # do_pw=pw_cfg(filter_fn=lambda x: "self_attn" in x),
    # do_patterns=True,
)

# %%

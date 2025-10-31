# %%
# ruff: noqa: E402, I001
import math
from spd.settings import REPO_ROOT
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.configs import Config, PGDReconLossConfig, SamplingType
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.metrics.pgd_utils import (
    _interpolate_component_mask,
    pgd_masked_recon_loss_update,
)
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    calc_kl_divergence_lm,
    extract_batch_data,
    replace_pydantic_model,
    runtime_cast,
    set_seed,
)
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.module_utils import get_target_module_paths

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class ModelSetup:
    """Container for loaded model and config."""

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        device: str,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device


def load_model(run_info_path: str) -> ModelSetup:
    """Load model. Call once per model type."""
    device = get_device()

    # Load run info and validate config
    run_info = SPDRunInfo.from_path(run_info_path)
    config = run_info.config
    task_config = config.task_config

    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Load model
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.target_model.requires_grad_(False)

    return ModelSetup(model, config, device)


LOSS_TYPE = "kl"


def build_standard_mask_infos(
    *,
    weight_deltas: dict[str, Tensor],
    ci: dict[str, Tensor],
    sampling: SamplingType,
    rounding_threshold: float,
) -> dict[str, dict[str, ComponentsMaskInfo]]:
    """Build a set of masking strategies consistent with ce_and_kl_losses.

    Returns mapping: strategy_name -> mask_infos (output of make_mask_infos / stochastic builder)
    Strategies: ci_masked, stoch_masked, random_masked, rounded_masked
    """
    # Build per-layer delta masks using CI leading dims so shapes match (batch, seq_len)

    ci_sample = next(iter(ci.values()))
    random_infos = make_mask_infos(
        component_masks={k: torch.rand_like(ci_sample) for k in ci},
        weight_deltas_and_masks={
            layer: (weight_deltas[layer], torch.rand_like(ci_sample[..., 0])) for layer in ci
        },
    )

    ci_masked_infos = make_mask_infos(component_masks=ci)

    rounded_ci = {k: (mod_ci > rounding_threshold).float() for k, mod_ci in ci.items()}
    rounded_infos = make_mask_infos(component_masks=rounded_ci)

    stoch_infos = calc_stochastic_component_mask_info(
        causal_importances=ci,
        component_mask_sampling=sampling,
        weight_deltas=weight_deltas,
        routing="all",
    )

    return {
        "ci_masked": ci_masked_infos,
        "stoch_masked": stoch_infos,
        "random_masked": random_infos,
        "rounded_masked": rounded_infos,
    }


def get_norms_vs_tokenwise_loss(
    *,
    model: ComponentModel,
    batch: Tensor,
    mask_infos: dict[str, ComponentsMaskInfo],
    target_out: Tensor,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    norm_paths = get_target_module_paths(model.target_model, [NORM_GLOB])  # ["*norm"]
    with model.cache_inputs(norm_paths) as cache, torch.no_grad():
        logits = model(batch, mask_infos=mask_infos)

    assert target_out is not None, "target_out required for KL loss"
    loss_per_token = calc_kl_divergence_lm(pred=logits, target=target_out, reduce=False)

    norm_scales: dict[str, Tensor] = {}
    for module_path, input_act in cache.items():
        variance = input_act.pow(2).mean(-1)
        norm_scales[module_path] = (variance + 1e-6).rsqrt()

    total_component_norm_tok = {}

    for module_path, mask_info in mask_infos.items():
        U_norms = model.components[module_path].U.norm(dim=1, p=2)  # (C,)
        V_norms = model.components[module_path].V.norm(dim=0, p=2)  # (C,)
        total_component_norms = U_norms * V_norms  # (C,)
        assert mask_info.routing_mask == "all", f"sanity check failed, got {mask_info.routing_mask}"
        engaged_component_norms = mask_info.component_mask * total_component_norms  # (B, S, C)
        engaged_component_norms_sum = engaged_component_norms.sum(dim=-1)  # (B, S)
        assert engaged_component_norms_sum.shape == loss_per_token.shape, (
            f"sanity check failed, got {engaged_component_norms_sum.shape} != {loss_per_token.shape}"
        )
        total_component_norm_tok[module_path] = engaged_component_norms_sum

    return loss_per_token, norm_scales, total_component_norm_tok


def plot_scales_vs_tokenwise_loss_grid(
    *,
    results_by_strategy: dict[str, tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]],
    output_dir: Path,
    file_name: str = "norm_vs_tokenwise_loss_grid.png",
    logx: bool = False,
    logy: bool = False,
    epsilon: float = 1e-8,
) -> None:
    """Grid of scatter plots with columns = norm layers, rows = strategies.

    Each subplot shows a single strategy's token-wise loss vs the input norm scale for one layer.
    Regression is computed in linear space; lines will appear bent on log axes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(results_by_strategy.keys())
    first_key = strategies[0]
    _ref_loss, ref_norms, _ref_norm_total = results_by_strategy[first_key]
    layer_names = list(ref_norms.keys())

    n_rows = len(strategies)
    n_cols = len(layer_names)
    fig_width = max(18, 5 * n_cols)
    fig_height = max(10, 4 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)

    for r_idx, strategy in enumerate(strategies):
        loss_tensor, scales, _ = results_by_strategy[strategy]
        y_t = loss_tensor.flatten().detach().cpu().numpy()
        y_plot = np.clip(y_t, epsilon, None) if logy else y_t

        for c_idx, layer_name in enumerate(layer_names):
            ax = axs[r_idx, c_idx]
            x_t = scales[layer_name].flatten().detach().cpu().numpy()
            x_plot = np.clip(x_t, epsilon, None) if logx else x_t

            ax.scatter(x_plot, y_plot, s=6, alpha=0.1, color="#1f77b4", linewidths=0)

            # if np.std(x_t) > 0 and np.std(y_t) > 0:
            #     m, b = np.polyfit(x_t, y_t, 1)
            #     x_min = float(np.min(x_plot)) if logx else float(np.min(x_t))
            #     x_max = float(np.max(x_plot)) if logx else float(np.max(x_t))
            #     xx = np.linspace(x_min, x_max, 100)
            #     yy = m * xx + b
            #     if logy:
            #         yy = np.where(yy > epsilon, yy, np.nan)
            #     ax.plot(xx, yy, color="black", linestyle="--", linewidth=1.2)

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            # Column titles on top row
            if r_idx == 0:
                ax.set_title(layer_name)
            # Row labels on first column
            if c_idx == 0:
                ax.set_ylabel(f"{strategy}\nToken-wise loss")
            # X label on bottom row
            if r_idx == n_rows - 1:
                ax.set_xlabel("Input norm scale")

            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_engaged_norms_vs_tokenwise_loss_grid(
    *,
    results_by_strategy: dict[str, tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]],
    output_dir: Path,
    file_name: str = "norm_vs_tokenwise_loss_grid.png",
    # logx: bool = False,
    # logy: bool = False,
    # epsilon: float = 1e-8,
) -> None:
    """Grid of scatter plots with columns = norm layers, rows = strategies.

    Each subplot shows a single strategy's token-wise loss vs the input norm scale for one layer.
    Regression is computed in linear space; lines will appear bent on log axes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(results_by_strategy.keys())
    first_key = strategies[0]
    _, _, ref_engaged_norm_tok = results_by_strategy[first_key]
    layer_names = list(ref_engaged_norm_tok.keys())

    n_rows = len(strategies)
    n_cols = len(layer_names)
    fig_width = max(18, 5 * n_cols)
    fig_height = max(10, 4 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)

    for r_idx, strategy in enumerate(strategies):
        loss_tensor, _, engaged_norm_tok = results_by_strategy[strategy]

        for c_idx, (layer_name, engaged_norm_tok_layer) in enumerate(engaged_norm_tok.items()):
            x_t = engaged_norm_tok_layer.flatten().detach().cpu().numpy()
            y_t = loss_tensor.flatten().detach().cpu().numpy()
            ax = axs[r_idx, c_idx]
            ax.scatter(x_t, y_t, s=6, alpha=0.1, color="#1f77b4", linewidths=0)
            ax.set_title(layer_name)
            ax.set_xlabel("Engaged component norm")
            ax.set_ylabel(f"Token-wise loss {strategy}")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)



def get_pgd_mask_infos(
    *,
    model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    pgd_config: PGDReconLossConfig,
    target_out: Tensor,
) -> dict[str, ComponentsMaskInfo]:
    _, _, adv_sources = pgd_masked_recon_loss_update(
        model=model,
        output_loss_type=LOSS_TYPE,
        routing="all",
        pgd_config=pgd_config,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
    )

    adv_sources = adv_sources.expand(len(model.target_module_paths), *batch.shape, model.C + 1)

    adv_sources_components = adv_sources[..., :-1]

    weight_deltas_and_masks = {
        k: (weight_deltas[k], adv_sources[i, ..., -1]) for i, k in enumerate(weight_deltas)
    }

    return make_mask_infos(
        component_masks=_interpolate_component_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks="all",
    )


# %%

if __name__ == "__main__":
    model_config = "wandb:goodfire/spd/runs/lxs77xye"

    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.04,
        n_steps=100,
        mask_scope="shared_across_batch",
        step_type="sign",
    )

    batch_size = 32

    max_seq_len = 512  # Adjust as needed

    print("Loading model")
    # Load model once per model type
    model_setup = load_model(model_config)
    model = model_setup.model
    config = model_setup.config
    assert config.use_delta_component, "use_delta_component must be True"

    print("Loading data and running experiments")
    # Update task config with desired max_seq_len
    task_config = runtime_cast(LMTaskConfig, config.task_config)
    task_config = replace_pydantic_model(task_config, {"max_seq_len": max_seq_len})
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
    set_seed(0)
    data_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    print("setting up for experiments")
    data_loader_iter = iter(data_loader)
    weight_deltas = model.calc_weight_deltas()

    batch = extract_batch_data(next(data_loader_iter)).to(model_setup.device)
    target_model_output: OutputWithCache = model(batch, cache_type="input")

    ci = model.calc_causal_importances(
        pre_weight_acts=target_model_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    print("building standard mask infos")
    strategies = build_standard_mask_infos(
        ci=ci.lower_leaky,
        sampling=config.sampling,
        rounding_threshold=0.01,
        weight_deltas=weight_deltas,
    )

    print("building pgd mask infos")
    strategies["pgd_masked"] = get_pgd_mask_infos(
        model=model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=weight_deltas,
        pgd_config=pgd_config,
        target_out=target_model_output.output,
    )

    print("harvesting norms")
    results_by_strategy: dict[str, tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]] = {}
    for name, mi in strategies.items():
        loss_tok, scales_tok, norm_total_tok = get_norms_vs_tokenwise_loss(
            model=model_setup.model,
            batch=batch,
            mask_infos=mi,
            target_out=target_model_output.output,
        )
        results_by_strategy[name] = (loss_tok, scales_tok, norm_total_tok)
        print(
            f" - {name}: mean loss: {loss_tok.mean().item():.4f}, std loss: {loss_tok.std().item():.4f}"
        )

    # %%

    print("plotting...")
    plot_scales_vs_tokenwise_loss_grid(
        results_by_strategy=results_by_strategy,
        output_dir=REPO_ROOT / "plots",
        file_name="norm_vs_tokenwise_loss_grid.png",
        logy=True,
    )

    # %%

    plot_engaged_norms_vs_tokenwise_loss_grid(
        results_by_strategy=results_by_strategy,
        output_dir=REPO_ROOT / "plots",
        file_name="engaged_norms_vs_tokenwise_loss_grid.png",
    )

# %%


# def plot_norms_vs_pgd_loss_scatter(
#     results: tuple[Tensor, dict[str, Tensor]],
#     output_dir: Path,
#     *,
#     logx: bool = False,
#     logy: bool = False,
#     epsilon: float = 1e-8,
# ) -> None:
#     """Scatter plots of input norm scales vs observed PGD loss per norm module.

#     Args:
#         results: (loss, norm_scales) where loss has shape (batch, seq_len)
#                  and norm_scales maps layer_name -> (batch, seq_len)
#         output_dir: Directory to save the figure
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)

#     loss_tensor, norm_scales = results
#     y = loss_tensor.flatten().detach().cpu().numpy()
#     y_plot = np.clip(y, epsilon, None) if logy else y

#     n = len(norm_scales)
#     cols = 3 if n >= 3 else n
#     rows = int(math.ceil(n / cols)) if cols > 0 else 1
#     fig_width = max(18, 6 * cols)
#     fig_height = max(10, 5 * rows)
#     fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)

#     # Ensure axs is 2D
#     for idx, (layer_name, vals) in enumerate(norm_scales.items()):
#         r = idx // cols
#         c = idx % cols
#         ax = axs[r, c]

#         x = vals.flatten().detach().cpu().numpy()
#         x_plot = np.clip(x, epsilon, None) if logx else x
#         ax.scatter(x_plot, y_plot, s=10, alpha=0.07)
#         # Pearson correlation (sequence positions are the set of points)
#         assert np.std(x) > 0 and np.std(y) > 0, f"{np.std(x)=} {np.std(y)=}"
#         r = float(np.corrcoef(x, y)[0, 1])
#         # Best fit line
#         m, b = np.polyfit(x, y, 1)
#         x_min = float(np.min(x_plot)) if logx else float(np.min(x))
#         x_max = float(np.max(x_plot)) if logx else float(np.max(x))
#         xx = np.linspace(x_min, x_max, 100)
#         yy = m * xx + b
#         if logy:
#             yy = np.where(yy > epsilon, yy, np.nan)
#         ax.plot(xx, yy, color="black", linestyle="--", linewidth=1.5, zorder=3)
#         if logx:
#             ax.set_xscale("log")
#         if logy:
#             ax.set_yscale("log")
#         ax.set_xlabel("Input norm scale")
#         ax.set_ylabel("Observed PGD loss")
#         ax.set_title(f"{layer_name} (r={r:.3f})")
#         ax.grid(True, alpha=0.3)

#     # Hide any unused subplots
#     for idx in range(n, rows * cols):
#         r = idx // cols
#         c = idx % cols
#         axs[r, c].set_visible(False)

#     fig.tight_layout()
#     out_path = output_dir / "norm_vs_pgd_loss_scatter.png"
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     print(f"Plot saved to {out_path}")
#     plt.close(fig)


# def plot_norms_vs_tokenwise_loss_scatter(
#     *,
#     results_by_strategy: dict[str, tuple[Tensor, dict[str, Tensor]]],
#     output_dir: Path,
#     file_name: str = "norm_vs_tokenwise_loss_scatter.png",
#     logx: bool = False,
#     logy: bool = False,
#     epsilon: float = 1e-8,
# ) -> None:
#     """Scatter plots comparing norm scales vs token-wise loss across masking strategies.

#     Args:
#         results_by_strategy: mapping from strategy name -> (loss_tensor, norm_scales)
#         output_dir: directory to save the figure
#         file_name: output PNG file name
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Use first strategy as reference for layers
#     first_key = next(iter(results_by_strategy))
#     _ref_loss, ref_norms = results_by_strategy[first_key]
#     layer_names = list(ref_norms.keys())

#     n = len(layer_names)
#     cols = 3 if n >= 3 else n
#     rows = int(math.ceil(n / cols)) if cols > 0 else 1
#     fig_width = max(18, 6 * cols)
#     fig_height = max(10, 5 * rows)
#     fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)

#     color_map = plt.colormaps["tab10"]
#     strategy_colors = {
#         name: color_map(i % color_map.N) for i, name in enumerate(results_by_strategy.keys())
#     }

#     for idx, layer_name in enumerate(layer_names):
#         r = idx // cols
#         c = idx % cols
#         ax = axs[r, c]

#         # Per strategy overlay: points (faint) + best-fit line
#         line_labels = []
#         line_handles = []
#         for strategy, (loss_tensor, norms) in results_by_strategy.items():
#             x_t = norms[layer_name].flatten().detach().cpu().numpy()
#             y_t = loss_tensor.flatten().detach().cpu().numpy()
#             x_plot = np.clip(x_t, epsilon, None) if logx else x_t
#             y_plot = np.clip(y_t, epsilon, None) if logy else y_t

#             color = strategy_colors[strategy]
#             ax.scatter(x_plot, y_plot, s=6, alpha=0.02, color=color, linewidths=0)

#             if np.std(x_t) > 0 and np.std(y_t) > 0:
#                 m, b = np.polyfit(x_t, y_t, 1)
#                 x_min = float(np.min(x_plot)) if logx else float(np.min(x_t))
#                 x_max = float(np.max(x_plot)) if logx else float(np.max(x_t))
#                 xx = np.linspace(x_min, x_max, 100)
#                 yy = m * xx + b
#                 if logy:
#                     yy = np.where(yy > epsilon, yy, np.nan)
#                 (line,) = ax.plot(xx, yy, color=color, linestyle="--", linewidth=1.5)
#                 rxy = float(np.corrcoef(x_t, y_t)[0, 1])
#                 line_handles.append(line)
#                 line_labels.append(f"{strategy} (r={rxy:.3f})")

#         ax.set_xlabel("Input norm scale")
#         ax.set_ylabel("Token-wise loss")
#         ax.set_title(layer_name)
#         ax.grid(True, alpha=0.3)
#         if logx:
#             ax.set_xscale("log")
#         if logy:
#             ax.set_yscale("log")
#         if line_handles:
#             ax.legend(line_handles, line_labels, loc="best", fontsize=8)

#     # Hide any unused subplots
#     for idx in range(n, rows * cols):
#         r = idx // cols
#         c = idx % cols
#         axs[r, c].set_visible(False)

#     fig.tight_layout()
#     out_path = output_dir / file_name
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     print(f"Plot saved to {out_path}")
#     plt.close(fig)

# def plot_norm_scales_and_pgd_loss(
#     results: tuple[Tensor, dict[str, Tensor]],
#     output_dir: Path,
# ) -> None:
#     """Plot per-sequence norm scales and observed PGD loss.

#     Args:
#         results: tuple(loss, norm_scales), where
#             - loss: Tensor with shape (batch, seq_len)
#             - norm_scales: dict of {layer_name: Tensor(batch, seq_len)}
#         output_dir: Directory to save plots
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)

#     colormap = plt.colormaps["tab20"]

#     loss_tensor, norm_scales = results
#     loss_np = loss_tensor.flatten().detach().cpu().numpy()
#     seq_len = int(loss_np.shape[-1])
#     xs = list(range(seq_len))

#     # Make the figure very wide to visualize high-fidelity values across positions
#     width_inches = max(24, seq_len / 16)  # e.g., 512 -> 32 inches
#     height_inches = 6
#     fig, ax1 = plt.subplots(figsize=(width_inches, height_inches), dpi=200)

#     # Plot norm scales per layer on primary y-axis
#     for i, (layer_name, vals) in enumerate(norm_scales.items()):
#         ys = vals.flatten().detach().cpu().numpy()
#         ax1.plot(
#             xs,
#             ys,
#             label=layer_name,
#             color=colormap(i % colormap.N),
#             alpha=0.9,
#             linewidth=1.5,
#         )

#     ax1.set_xlabel("Sequence position")
#     ax1.set_ylabel("Input norm scale")
#     ax1.grid(True, alpha=0.3)

#     # Plot observed PGD loss on secondary y-axis for emphasis
#     ax2 = ax1.twinx()
#     ax2.plot(xs, loss_np, label="Observed PGD loss", color="red", linewidth=2.5)
#     ax2.set_ylabel("PGD loss")

#     # Combine legends from both axes
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

#     plt.title("Norm scales and observed PGD loss")
#     fig.tight_layout()

#     out_path = output_dir / "norm_scales_vs_pgd_loss.png"
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"Plot saved to {out_path}")
#     plt.close(fig)

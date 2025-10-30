# %%
# ruff: noqa: E402
import math
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.metrics.pgd_utils import (
    _interpolate_component_mask,
    pgd_masked_recon_loss_update,
)
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    calc_kl_divergence_lm,
    extract_batch_data,
    replace_pydantic_model,
    set_seed,
)
from spd.utils.module_utils import get_target_module_paths

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


ModelType = Literal["lm1", "lm2", "lm3", "lm4"]


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


def run_experiment(
    model_setup: ModelSetup,
    seed: int,
    pgd_config: PGDReconLossConfig,
    batch_size: int,
    max_seq_len: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run PGD reconstruction loss experiment for a single seed using pre-loaded model."""
    model = model_setup.model
    config = model_setup.config
    device = model_setup.device

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    set_seed(seed)

    # Update task config with desired max_seq_len
    updated_task_config = replace_pydantic_model(
        task_config, {"max_seq_len": max_seq_len, "train_data_split": "train"}
    )

    train_data_config = DatasetConfig(
        name=updated_task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=updated_task_config.train_data_split,
        n_ctx=updated_task_config.max_seq_len,
        is_tokenized=updated_task_config.is_tokenized,
        streaming=updated_task_config.streaming,
        column_name=updated_task_config.column_name,
        shuffle_each_epoch=updated_task_config.shuffle_each_epoch,
        seed=None,
    )

    set_seed(0)

    data_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=updated_task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    data_loader_iter = iter(data_loader)
    weight_deltas = model.calc_weight_deltas()

    batch = extract_batch_data(next(data_loader_iter)).to(device)
    target_model_output: OutputWithCache = model(batch, cache_type="input")

    ci = model.calc_causal_importances(
        pre_weight_acts=target_model_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    assert config.use_delta_component, "use_delta_component must be True"

    tloss, n_ex, adv_sources = pgd_masked_recon_loss_update(
        model=model,
        output_loss_type=LOSS_TYPE,
        routing="all",
        pgd_config=pgd_config,
        batch=batch,
        target_out=target_model_output.output,
        ci=ci.lower_leaky,
        weight_deltas=weight_deltas,
    )

    adv_sources = adv_sources.expand(len(model.target_module_paths), *batch.shape, model.C + 1)

    adv_sources_components = adv_sources[..., :-1]

    weight_deltas_and_masks = {
        k: (weight_deltas[k], adv_sources[i, ..., -1]) for i, k in enumerate(weight_deltas)
    }

    mask_infos = make_mask_infos(
        component_masks=_interpolate_component_mask(ci.lower_leaky, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks="all",
    )

    norm_paths = get_target_module_paths(model.target_model, ["*norm"])
    with model.cache_inputs(norm_paths) as cache:
        out = model(batch, mask_infos=mask_infos)

    loss = calc_kl_divergence_lm(pred=out, target=target_model_output.output, reduce=False)

    # == Sanity check ==
    total_observed_loss = loss.mean()
    print(f"{total_observed_loss=} VS {tloss / n_ex=}")
    # == EOF Sanity check ==

    assert loss.shape == batch.shape, "what?"

    norm_scales = {}
    for module_path, input in cache.items():
        variance = (input**2).mean(-1)
        norm_scales[module_path] = (variance + 1e-6).sqrt()

    assert next(iter(norm_scales.values())).shape == batch.shape, "what 2?"

    return loss, norm_scales


def plot_norms_vs_pgd_loss_scatter(
    results: tuple[Tensor, dict[str, Tensor]],
    output_dir: Path,
) -> None:
    """Scatter plots of input norm scales vs observed PGD loss per norm module.

    Args:
        results: (loss, norm_scales) where loss has shape (batch, seq_len)
                 and norm_scales maps layer_name -> (batch, seq_len)
        output_dir: Directory to save the figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_tensor, norm_scales = results
    y = loss_tensor.flatten().detach().cpu().numpy()

    n = len(norm_scales)
    cols = 3 if n >= 3 else n
    rows = int(math.ceil(n / cols)) if cols > 0 else 1
    fig_width = max(18, 6 * cols)
    fig_height = max(10, 5 * rows)
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)

    # Ensure axs is 2D
    for idx, (layer_name, vals) in enumerate(norm_scales.items()):
        r = idx // cols
        c = idx % cols
        ax = axs[r, c]

        x = vals.flatten().detach().cpu().numpy()
        ax.scatter(x, y, s=10, alpha=0.01)
        # Pearson correlation (sequence positions are the set of points)
        assert np.std(x) > 0 and np.std(y) > 0, f"{np.std(x)=} {np.std(y)=}"
        r = float(np.corrcoef(x, y)[0, 1])
        # Best fit line
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xx, m * xx + b, color="black", linestyle="--", linewidth=1.5, zorder=3)
        ax.set_xlabel("Input norm scale")
        ax.set_ylabel("Observed PGD loss")
        ax.set_title(f"{layer_name} (r={r:.3f})")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axs[r, c].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "norm_vs_pgd_loss_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_norm_scales_and_pgd_loss(
    results: tuple[Tensor, dict[str, Tensor]],
    output_dir: Path,
) -> None:
    """Plot per-sequence norm scales and observed PGD loss.

    Args:
        results: tuple(loss, norm_scales), where
            - loss: Tensor with shape (batch, seq_len)
            - norm_scales: dict of {layer_name: Tensor(batch, seq_len)}
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    colormap = plt.colormaps["tab20"]

    loss_tensor, norm_scales = results
    loss_np = loss_tensor.flatten().detach().cpu().numpy()
    seq_len = int(loss_np.shape[-1])
    xs = list(range(seq_len))

    # Make the figure very wide to visualize high-fidelity values across positions
    width_inches = max(24, seq_len / 16)  # e.g., 512 -> 32 inches
    height_inches = 6
    fig, ax1 = plt.subplots(figsize=(width_inches, height_inches), dpi=200)

    # Plot norm scales per layer on primary y-axis
    for i, (layer_name, vals) in enumerate(norm_scales.items()):
        ys = vals.flatten().detach().cpu().numpy()
        ax1.plot(
            xs,
            ys,
            label=layer_name,
            color=colormap(i % colormap.N),
            alpha=0.9,
            linewidth=1.5,
        )

    ax1.set_xlabel("Sequence position")
    ax1.set_ylabel("Input norm scale")
    ax1.grid(True, alpha=0.3)

    # Plot observed PGD loss on secondary y-axis for emphasis
    ax2 = ax1.twinx()
    ax2.plot(xs, loss_np, label="Observed PGD loss", color="red", linewidth=2.5)
    ax2.set_ylabel("PGD loss")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    plt.title("Norm scales and observed PGD loss")
    fig.tight_layout()

    out_path = output_dir / "norm_scales_vs_pgd_loss.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)

# %%

if __name__ == "__main__":
    """Run experiments for all models and seeds, generate report."""
    model_config = "wandb:goodfire/spd/runs/lxs77xye"

    pdg_config = PGDReconLossConfig(
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

    print("Running experiments\n")
    results = run_experiment(
        model_setup=model_setup,
        seed=0,
        pgd_config=pdg_config,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # %%

    # Plot and save figures
    plot_norms_vs_pgd_loss_scatter(results, output_dir=Path("logs") / "pgd_lm_plots")
    # plot_norm_scales_and_pgd_loss(results, output_dir=Path("logs") / "pgd_lm_plots")



    # %%
    list(results[1])
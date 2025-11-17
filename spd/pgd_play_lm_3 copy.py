# ruff: noqa: E402, I001
# %%
from jaxtyping import Int
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, set_seed
from torch import Tensor

# %%

device = get_device()

# %%

OLD = "wandb:goodfire/spd/runs/lxs77xye"
NEW = "wandb:goodfire/spd/runs/9gf5ud48"
# MLP_UP_0 = "model.layers.0.mlp.up_proj"
MLP_DOWN_1 = "model.layers.1.mlp.down_proj"

_config = SPDRunInfo.from_path(OLD).config
_task_config = _config.task_config
assert isinstance(_task_config, LMTaskConfig), "task_config not LMTaskConfig"

train_data_config = DatasetConfig(
    name=_task_config.dataset_name,
    hf_tokenizer_path=_config.tokenizer_name,
    split=_task_config.train_data_split,
    n_ctx=_task_config.max_seq_len,
    is_tokenized=_task_config.is_tokenized,
    streaming=_task_config.streaming,
    column_name=_task_config.column_name,
    shuffle_each_epoch=_task_config.shuffle_each_epoch,
    seed=None,
)


_tokenizer: Any
data_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=_task_config.buffer_size,
    global_seed=_config.seed,
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


def run_for_layer(model: ComponentModel, config: Config, batch: Int[Tensor, "1 seq"]):
    target_output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    pgd_loss, n_ex, _ = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=model.calc_weight_deltas(),
        target_out=target_output.output,
        output_loss_type=config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
    )

    return pgd_loss / n_ex


# %%

tokenizer: Any
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=4,
    buffer_size=_task_config.buffer_size,
    global_seed=_config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_loader_iter = iter(data_loader)

losses = []
from itertools import islice

opts_per_layer = 8

for batch_ in islice(data_loader_iter, opts_per_layer):
    batch = extract_batch_data(batch_).to(device)
    # assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"
    for run in [OLD, NEW]:
        print(f"Run: {run}")
        for layer in ComponentModel.from_run_info(SPDRunInfo.from_path(run)).target_module_paths:
            set_seed(0)
            spd_run_info = SPDRunInfo.from_path(run)
            config = spd_run_info.config
            model = ComponentModel.from_run_info(spd_run_info, module_patterns=[layer]).to(device)
            model.requires_grad_(False)
            loss = run_for_layer(model, config, batch)
            print(f"  Layer: {layer}, Loss: {loss}")
            losses.append({"run": run, "layer": layer, "loss": loss.item()})

# %%
from collections import defaultdict

# Aggregate losses per run and layer
losses_by_run_layer: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
layers_order_by_run: dict[str, list[str]] = defaultdict(list)

for rec in losses:
    if "identity" in rec["layer"]:
        continue
    r = rec["run"]
    layer_name = rec["layer"]
    v = rec["loss"]
    if layer_name not in layers_order_by_run[r]:
        layers_order_by_run[r].append(layer_name)
    losses_by_run_layer[r][layer_name].append(v)


def layer_type_sort(lname: str):
    if "q_proj" in lname:
        return 0
    if "k_proj" in lname:
        return 1
    if "v_proj" in lname:
        return 2
    if "o_proj" in lname:
        return 3
    if "up_proj" in lname:
        return 4
    if "gate_proj" in lname:
        return 5
    if "down_proj" in lname:
        return 6
    return 9_999


def layer_idx_sort(lname: str):
    if lname == "lm_head":
        return 10_000
    if lname == "model.norm":
        return 9_999

    assert lname.startswith("model.layers."), f"Unknown layer: {lname}"
    lname = lname.removeprefix("model.layers.")
    l_idx = int(lname.split(".")[0])
    return l_idx


def get_model_all_layers(run_name: str) -> ComponentModel:
    spd_run_info = SPDRunInfo.from_path(run_name)
    model = ComponentModel.from_run_info(spd_run_info).to(device)
    model.requires_grad_(False)
    return model


# Create one combined visualization:
# left = scatter of PGD losses, right = per-layer component norm histograms (both runs)
runs = list(layers_order_by_run.keys())
assert runs
# Build a unified layer order
layers_all: list[str] = []
for r in runs:
    for layer in layers_order_by_run[r]:
        if layer not in layers_all:
            layers_all.append(layer)
assert layers_all

layers_all.sort(key=lambda x: (layer_idx_sort(x), layer_type_sort(x)))

models_by_run = {run: get_model_all_layers(run) for run in runs}
component_norms_by_run: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
max_norm = 0.0
for run, model in models_by_run.items():
    run_norms = component_norms_by_run[run]
    for layer in layers_all:
        if layer not in model.components:
            continue
        comp = model.components[layer]
        U_norms = comp.U.norm(dim=1)
        V_norms = comp.V.norm(dim=0)
        layer_norms = (U_norms * V_norms).detach().cpu().numpy()
        if layer_norms.size == 0:
            continue
        run_norms[layer] = layer_norms
        max_norm = max(max_norm, float(layer_norms.max()))

assert max_norm > 0

bins = np.linspace(0, max_norm, 120).tolist()

# %%

fig_height = len(layers_all)
fig = plt.figure(figsize=(20, fig_height))
gs = fig.add_gridspec(len(layers_all), 2, width_ratios=[3, 2], wspace=0.25)

ax_scatter = fig.add_subplot(gs[:, 0])
hist_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(layers_all))]

palette = ["C0", "C1"]
run_colors = {r: palette[i % len(palette)] for i, r in enumerate(runs)}
legend_handles: dict[str, Any] = {}
legend_done = {r: False for r in runs}

for layer_idx, layer in enumerate(layers_all):
    for r in runs:
        vals = losses_by_run_layer[r][layer][:opts_per_layer]
        for loss_val in vals:
            handle = ax_scatter.scatter(
                loss_val,
                layer_idx,
                s=20,
                color=run_colors[r],
                alpha=0.7,
                label=(r if not legend_done[r] else "_nolegend_"),
            )
            if not legend_done[r]:
                legend_handles[r] = handle
                legend_done[r] = True

ax_scatter.set_yticks(list(range(len(layers_all))), layers_all, fontsize=8)
ax_scatter.set_ylim(-0.5, len(layers_all) - 0.5)
ax_scatter.set_xlabel("PGD loss")
ax_scatter.set_ylabel("Layer")
ax_scatter.set_title(f"PGD loss when optimizing a single layer ({opts_per_layer} runs per layer)")

for idx, (layer, ax_hist) in enumerate(zip(layers_all, reversed(hist_axes), strict=True)):
    for r in runs:
        layer_norms = component_norms_by_run[r][layer]
        assert layer_norms.size != 0
        ax_hist.hist(
            layer_norms,
            bins=bins,
            alpha=0.5,
            color=run_colors[r],
            label=r if idx == 0 and not legend_done[r] else None,
            log=True,
        )
    ax_hist.set_yticks([])
    ax_hist.set_xlim(bins[0], bins[-1])
    ax_hist.set_title(layer, loc="left", fontsize=8)
    if idx == len(layers_all) - 1:
        ax_hist.set_xlabel("Component norm")

ax_scatter.legend(legend_handles.values(), legend_handles.keys(), loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %%
import torch
from tqdm import tqdm


def find_mean_ci(model: ComponentModel, n_batches: int):
    sum_ci = {module: torch.zeros(model.C, device=device) for module in model.target_module_paths}
    n_examples = 0
    for batch in tqdm(islice(data_loader_iter, n_batches)):
        target_output = model(extract_batch_data(batch).to(device), cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=config.sampling,
        ).lower_leaky
        for module_name, ci_val in ci.items():
            sum_ci[module_name] += ci_val.sum(dim=0).sum(dim=0)
        n_examples += ci_val.shape[:-1].numel()  # pyright: ignore[reportPossiblyUnboundVariable]
    return {module_name: sum_ci_val for module_name, sum_ci_val in sum_ci.items()}


# %%
old_all = get_model_all_layers(OLD)
new_all = get_model_all_layers(NEW)
# %%

print("Finding dead components for old model...")
old_mean_ci = find_mean_ci(old_all, n_batches=500)
del old_mean_ci["model.layers.0.mlp.gate_proj.pre_identity"]

print("Finding dead components for new model...")
new_mean_ci = find_mean_ci(new_all, n_batches=500)

# %%


def plot_ci_grid(ci: dict[str, Tensor]):
    ncols = 7
    nrows = (len(old_mean_ci) + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(30, 5 * nrows))
    for i, (lname, ci_val) in enumerate(ci.items()):
        row = i // ncols
        col = i % ncols
        axes[row, col].plot(ci_val[ci_val.argsort(descending=True)].cpu().detach().numpy())
        axes[row, col].set_yscale("log")
        axes[row, col].set_ylim(1e-13, 1)
        axes[row, col].set_title(f"CI for {lname}")
    plt.show()


# %%

plot_ci_grid(old_mean_ci)
plot_ci_grid(new_mean_ci)
# %%


def alive(x: Tensor):
    return x > 1e-5


old_alive_indices = {key: torch.where(alive(old_mean_ci[key]))[0] for key in old_mean_ci}
new_alive_indices = {key: torch.where(alive(new_mean_ci[key]))[0] for key in new_mean_ci}

# %%


def plot_mean_ci_vs_norm(
    mean_ci: dict[str, np.ndarray],
    norms: dict[str, np.ndarray],
    run_name: str,
    log_x: bool = True,
    log_y: bool = True,
):
    n_cols = 7
    n_rows = (len(mean_ci) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    for i, (key, mean_cis) in enumerate(mean_ci.items()):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].scatter(mean_cis, norms[key], alpha=0.2, s=10)
        if log_x: axes[row, col].set_xscale("log")  # noqa: E701
        if log_y: axes[row, col].set_yscale("log")  # noqa: E701
        axes[row, col].set_title(key, fontsize=8)
    fig.supxlabel("Mean CI")
    fig.supylabel("Component norm")
    plt.suptitle(f"Mean CI vs norm of components for {run_name}")
    plt.show()


# %%

plot_mean_ci_vs_norm(
    {key: old_mean_ci[key].cpu().detach().numpy() for key in old_mean_ci},
    component_norms_by_run[OLD],
    OLD,
)
plot_mean_ci_vs_norm(
    {key: old_mean_ci[key].cpu().detach().numpy() for key in old_mean_ci},
    component_norms_by_run[OLD],
    OLD,
    log_x=False,
    log_y=False,
)
plot_mean_ci_vs_norm(
    {key: new_mean_ci[key].cpu().detach().numpy() for key in new_mean_ci},
    component_norms_by_run[NEW],
    NEW,
)
plot_mean_ci_vs_norm(
    {key: new_mean_ci[key].cpu().detach().numpy() for key in new_mean_ci},
    component_norms_by_run[NEW],
    NEW,
    log_x=False,
    log_y=False,
)
# %%


plot_mean_ci_vs_norm(
    {key: old_mean_ci[key].cpu().detach().numpy() for key in old_mean_ci},
    component_norms_by_run[OLD],
    OLD,
    log_y=False,
)

plot_mean_ci_vs_norm(
    {key: new_mean_ci[key].cpu().detach().numpy() for key in new_mean_ci},
    component_norms_by_run[NEW],
    NEW,
    log_y=False,
)

# %%


# ruff: noqa: E402, I001
# %%
import torch
import warnings
from typing import Literal

import matplotlib.pyplot as plt

from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.utils.module_utils import get_target_module_paths

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    calc_kl_divergence_lm,
    extract_batch_data,
    runtime_cast,
    set_seed,
)

# %%

ModelType = Literal["lm1", "lm2", "lm3", "lm4"]


class ModelSetup:
    """Container for loaded model and config."""

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device


def load_model(run_info_path: str, module_patterns: list[str]) -> ModelSetup:
    """Load model. Call once per model type."""
    device = get_device()

    # Load run info and validate config
    run_info = SPDRunInfo.from_path(run_info_path)
    config = run_info.config
    task_config = config.task_config

    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Load model
    model = ComponentModel.from_run_info(run_info, module_patterns=module_patterns)
    model.to(device)
    model.target_model.requires_grad_(False)

    return ModelSetup(model, config, device)


from torch import Tensor
from spd.models.components import ComponentsMaskInfo, LinearComponents, make_mask_infos
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.configs import SamplingType


def build_mask_infos(
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

    ci_masked_infos = make_mask_infos(component_masks=ci)

    rounded_ci = {k: (mod_ci > rounding_threshold).float() for k, mod_ci in ci.items()}
    rounded_infos = make_mask_infos(component_masks=rounded_ci)

    stoch_infos = calc_stochastic_component_mask_info(
        causal_importances=ci,
        component_mask_sampling=sampling,
        weight_deltas=weight_deltas,
        routing="all",
    )

    target_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.items()},
        weight_deltas_and_masks={
            k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.items()
        },
    )

    return {
        "target": target_infos,
        "ci": ci_masked_infos,
        "stoch": stoch_infos,
        "rounded": rounded_infos,
    }


# %%

config_path = "wandb:goodfire/spd/runs/lxs77xye"
model_setup = load_model(config_path, module_patterns=["model.layers.0.mlp.up_proj"])

model = model_setup.model
config: Config = model_setup.config
device = model_setup.device

task_config = config.task_config
assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

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

# %%

set_seed(0)
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)

pgd_config = PGDReconLossConfig(
    init="random",
    step_size=0.1,
    n_steps=50,
    mask_scope="shared_across_batch",
)

data_loader_iter = iter(data_loader)
weight_deltas = model.calc_weight_deltas()

batch = extract_batch_data(next(data_loader_iter)).to(device)

target_model_output = model(batch, cache_type="input")

ci = model.calc_causal_importances(
    pre_weight_acts=target_model_output.cache,
    detach_inputs=False,
    sampling=config.sampling,
)

# %%

assert config.use_delta_component
mask_infos = build_mask_infos(
    weight_deltas=weight_deltas,
    ci=ci.lower_leaky,
    sampling=config.sampling,
    rounding_threshold=0.5,
)
_, _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
    model=model,
    batch=batch,
    ci=ci.lower_leaky,
    weight_deltas=weight_deltas,
    target_out=target_model_output.output,
    output_loss_type=config.output_loss_type,
    routing="all",
    pgd_config=pgd_config,
)
mask_infos["pgd"] = pgd_mask_infos

# %%


def tw_loss(mask_infos: dict[str, ComponentsMaskInfo]) -> Tensor:
    return calc_kl_divergence_lm(
        pred=model(batch, mask_infos=mask_infos),
        target=target_model_output.output,
        reduce=False,
    )


tw_losses = {mask_type: tw_loss(mask_infos=mi)[0] for mask_type, mi in mask_infos.items()}

# %%
(pgd_abv_10_idcs,) = torch.where(tw_losses["pgd"] > 10)
(first_pgd_abv_10_idx := pgd_abv_10_idcs[0])
# %%
(layer,) = model.target_module_paths
# %%
(pgd_mask := mask_infos["pgd"][layer].copy().component_mask)
# %%

# %%

new_pgd_mask_infos = {layer: mask_infos["pgd"][layer].copy()}
new_pgd_mask_infos[layer].component_mask[0, first_pgd_abv_10_idx] = pgd_mask[
    0, first_pgd_abv_10_idx
]

# %%

mask_infos["new_pgd"] = new_pgd_mask_infos
tw_losses["new_pgd"] = tw_loss(mask_infos=new_pgd_mask_infos)[0]

# %%


# bad_idx_snapped_ci_mask_infos = {layer: mask_infos["ci"][layer].copy()}
# ci_mask = mask_infos["ci"][layer].copy().component_mask
# snapped_ci_mask = torch.where(ci_mask > 0, 1.0, 0.0)
# bad_idx_snapped_ci_mask_infos[layer].component_mask[0, first_pgd_abv_10_idx] = snapped_ci_mask[
#     0, first_pgd_abv_10_idx
# ]
# mask_infos["bad_idx_snapped_ci"] = bad_idx_snapped_ci_mask_infos
# tw_losses["bad_idx_snapped_ci"] = tw_loss(mask_infos=bad_idx_snapped_ci_mask_infos)[0]

# Log differences between tw losses:

# for mask_type in ["ci", "bad_idx_snapped_ci"]:
# ci__ = tw_losses["ci"]
# bad_idx_snapped_ci__ = tw_losses["bad_idx_snapped_ci"]
# # assert tw_loss_vals.shape == (512,)
# tw_loss_vals_np = (ci__ - bad_idx_snapped_ci__).cpu().detach().numpy()
# plt.plot(tw_loss_vals_np, alpha=1, linestyle="--")
# plt.legend()
# # make plot way wider
# plt.gcf().set_size_inches(30, 5)
# plt.show()

# %%
output_acts = {
    k: model(batch, mask_infos=mi, cache_type="output").cache[layer][0, first_pgd_abv_10_idx]
    for k, mi in mask_infos.items()
}

# %%

labels = list(output_acts.keys())
output_acts_t = torch.stack(list(output_acts.values()))
pw_cosine_sims = torch.nn.functional.cosine_similarity(
    output_acts_t[None], output_acts_t[:, None], dim=-1
)
# pw_cosine_sims.fill_diagonal_(0)
plt.imshow(pw_cosine_sims.cpu().detach().numpy(), cmap="RdBu", vmin=-1, vmax=1)
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.title("output activation cos sim")
plt.colorbar()

# %%

# bar chart of norms:

plt.bar(range(len(labels)), [v.norm(dim=-1).item() for v in output_acts.values()])
plt.xticks(range(len(labels)), labels, rotation=45)
plt.show()
# %%


(lm_head_path,) = get_target_module_paths(model.target_model, ["*lm_head"])

(norm_path,) = get_target_module_paths(model.target_model, ["model.norm"])

(layer_3_path,) = get_target_module_paths(model.target_model, ["model.layers.3"])
(layer_2_path,) = get_target_module_paths(model.target_model, ["model.layers.2"])
(layer_1_path,) = get_target_module_paths(model.target_model, ["model.layers.1"])
(layer_0_path,) = get_target_module_paths(model.target_model, ["model.layers.0"])
(mlpup0,) = get_target_module_paths(model.target_model, ["model.layers.0.mlp.up_proj"])
(mlpdown0,) = get_target_module_paths(model.target_model, ["model.layers.0.mlp.down_proj"])
(mlpgate0,) = get_target_module_paths(model.target_model, ["model.layers.0.mlp.gate_proj"])

(
    paths := [
        lm_head_path,
        norm_path,
        layer_3_path,
        layer_2_path,
        layer_1_path,
        layer_0_path,
        mlpdown0,
        mlpgate0,
        mlpup0,
    ]
)

# %%
# %%


# Collect all layer data first
layer_data: dict[str, dict[str, Tensor]] = {}
for path in paths:
    output_caches = {}
    for mask_type, mask_info in mask_infos.items():
        with model.cache_modules([path]) as (_, output_cache):
            model(batch, mask_infos=mask_info)
        output_caches[mask_type] = output_cache[path][0, first_pgd_abv_10_idx]
    layer_data[path] = output_caches

# Create multi-panel plot
n_layers = len(paths)
labels = list(next(iter(layer_data.values())).keys())

fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))

# Ensure axes is 2D even for single layer
if n_layers == 1:
    axes = axes.reshape(1, -1)

for i, path in enumerate(paths):
    output_caches = layer_data[path]
    output_caches_t = torch.stack(list(output_caches.values()))

    # Left: Cosine similarity heatmap
    pw_cosine_sims = torch.nn.functional.cosine_similarity(
        output_caches_t[None], output_caches_t[:, None], dim=-1
    )
    cosine_sims_np = pw_cosine_sims.cpu().detach().numpy()
    im = axes[i, 0].imshow(cosine_sims_np, cmap="RdBu", vmin=-1, vmax=1)
    axes[i, 0].set_xticks(range(len(labels)))
    axes[i, 0].set_xticklabels(labels, rotation=45, ha="right")
    axes[i, 0].set_yticks(range(len(labels)))
    axes[i, 0].set_yticklabels(labels)

    # Annotate cells with values
    for row in range(len(labels)):
        for col in range(len(labels)):
            val = cosine_sims_np[row, col]
            # Choose text color based on value for readability
            text_color = "white" if abs(val) > 0.5 else "black"
            axes[i, 0].text(
                col, row, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8
            )

    # Add row label on the left
    layer_name = path.split(".")[-1] if "." in path else path
    axes[i, 0].set_ylabel(layer_name, fontsize=12, fontweight="bold")

    # Add colorbar only for top row
    if i == 0:
        axes[i, 0].set_title("Pairwise Cosine Similarity")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

    # Right: Magnitude bar chart
    magnitudes = [v.norm(dim=-1).item() for v in output_caches.values()]
    bars = axes[i, 1].bar(range(len(labels)), magnitudes)
    axes[i, 1].set_xticks(range(len(labels)))
    axes[i, 1].set_xticklabels(labels, rotation=45, ha="right")
    axes[i, 1].set_ylabel("L2 Norm")

    # Annotate bars with values
    for bar, mag in zip(bars, magnitudes, strict=True):
        axes[i, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mag:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if i == 0:
        axes[i, 1].set_title("Activation Magnitudes")

    # Only show x-labels on bottom row
    if i < n_layers - 1:
        axes[i, 0].set_xticklabels([])
        axes[i, 1].set_xticklabels([])

plt.tight_layout()
plt.show()
# %%

for mask_type, output_act in layer_data["lm_head"].items():
    assert output_act.shape == (4096,)
    sm = output_act.softmax(dim=0)
    topk_toks = sm.topk(k=10, dim=0)
    tokenizer_output = tokenizer.convert_ids_to_tokens(topk_toks.indices.cpu().detach().numpy())
    print(f"{mask_type}:")
    for token, probability in zip(
        tokenizer_output, topk_toks.values.cpu().detach().numpy(), strict=True
    ):
        print(f"  {token:>10}: {probability:.2f}")


# %%

from rich import print as rprint
from rich.table import Table

table = Table(title="Top 10 Tokens")
for mask_type in mask_infos:
    table.add_column(mask_type)
    table.add_column("P")

toks = {}
probs = {}
K = 10
for mask_type, output_act in layer_data["lm_head"].items():
    sm = output_act.softmax(dim=0)
    topk_toks = sm.topk(k=K, dim=0)
    toks[mask_type] = tokenizer.convert_ids_to_tokens(topk_toks.indices.cpu().detach().numpy())
    probs[mask_type] = topk_toks.values.cpu().detach().numpy()

for i in range(K):
    data = []
    for mask_type in mask_infos:
        data.append(toks[mask_type][i])
        data.append(f"{probs[mask_type][i]:.2f}")
    table.add_row(*data)
rprint(table)

# %%
comp = runtime_cast(LinearComponents, next(iter(model.components.values())))
U_norms = comp.U.norm(dim=1)  # (C,)


def get_effective_U_norms(*, mask_infos: dict[str, ComponentsMaskInfo]) -> tuple[Tensor, Tensor]:
    model(batch, mask_infos=mask_infos)
    assert (ia := comp.inner_acts) is not None
    inner_acts = ia[0, first_pgd_abv_10_idx]
    comp.inner_acts = None
    masks = next(iter(mask_infos.values())).component_mask[0, first_pgd_abv_10_idx]
    return inner_acts, masks


# %%

_, ci_masks = get_effective_U_norms(mask_infos=mask_infos["ci"])
inner_acts, pgd_masks = get_effective_U_norms(mask_infos=mask_infos["pgd"])

plt.bar(range(len(ci_masks)), (ci_masks * -1).cpu().detach().numpy(), alpha=0.5)
plt.bar(range(len(pgd_masks)), pgd_masks.cpu().detach().numpy(), alpha=0.5)
plt.bar(range(len(inner_acts)), inner_acts.cpu().detach().numpy(), alpha=0.5)
plt.legend(["ci", "pgd", "inner_acts"])
plt.gcf().set_size_inches(100, 20)
plt.savefig("inner_acts.png", dpi=150, bbox_inches="tight")
plt.show()
# %%

# scatter plots:

plt.scatter(ci_masks.cpu().detach().numpy(), inner_acts.cpu().detach().numpy())
plt.legend(["ci", "pgd inner"])
plt.gcf().set_size_inches(10, 10)
plt.savefig("inner_acts_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
# %%

# 3d scatter plot using plotly:
import plotly.express as px
fig = px.scatter_3d(
    x=ci_masks.cpu().detach().numpy(),
    y=pgd_masks.cpu().detach().numpy(),
    z=inner_acts.cpu().detach().numpy(),
)
fig.show()
# %%

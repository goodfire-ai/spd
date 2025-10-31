# ruff: noqa: E402, I001
# %%
from collections.abc import Callable
from dataclasses import dataclass
import torch
import warnings

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
    set_seed,
)

# %%


@dataclass
class ModelSetup:
    model: ComponentModel
    config: Config
    device: str | torch.device


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
from spd.models.components import ComponentsMaskInfo, make_mask_infos

# %%

config_path = "wandb:goodfire/spd/runs/lxs77xye"
model_setup = load_model(config_path, module_patterns=["model.layers.0.mlp.up_proj"])

model = model_setup.model
(layer,) = model.target_module_paths
EXPECTED_POS = 108

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
batch = extract_batch_data(next(data_loader_iter)).to(device)

target_model_output = model(batch, cache_type="input")

ci = model.calc_causal_importances(
    pre_weight_acts=target_model_output.cache,
    detach_inputs=False,
    sampling=config.sampling,
)

weight_deltas = model.calc_weight_deltas()


def tw_loss(mask_infos: dict[str, ComponentsMaskInfo]) -> Tensor:
    return calc_kl_divergence_lm(
        pred=model(batch, mask_infos=mask_infos),
        target=target_model_output.output,
        reduce=False,
    )


# %%

masks_by_seed: dict[int, ComponentsMaskInfo] = {}
cat_by_seed: dict[int, bool] = {}
for seed in range(50):
    set_seed(seed)
    assert config.use_delta_component

    target_mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        weight_deltas_and_masks={
            k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.lower_leaky.items()
        },
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

    pgd_tw_loss = tw_loss(mask_infos=pgd_mask_infos)[0]

    (pgd_abv_10_idcs,) = torch.where(pgd_tw_loss > 10)
    pgd_loss_vals = pgd_tw_loss[pgd_abv_10_idcs]
    print(f"pgd loss vals: {pgd_loss_vals}")

    (first_pgd_abv_10_idx := pgd_abv_10_idcs[0])
    if first_pgd_abv_10_idx != EXPECTED_POS:
        continue

    mask_w_single_bit_replaced = target_mask_infos[layer].copy()
    mask_w_single_bit_replaced.component_mask[0, first_pgd_abv_10_idx].copy_(
        pgd_mask_infos[layer].component_mask[0, first_pgd_abv_10_idx]
    )
    # mask_w_single_bit_replaced.weight_delta_and_mask = None

    masks_by_seed[seed] = mask_w_single_bit_replaced

    new_pgd_tw_loss = tw_loss(mask_infos={layer: mask_w_single_bit_replaced})[0]
    is_catastrophic = new_pgd_tw_loss.mean().item() > 1
    cat_by_seed[seed] = is_catastrophic
    # plt.plot(new_pgd_tw_loss.cpu().detach().numpy(), alpha=1, linestyle="--")
    # plt.show()

# %%

paths = get_target_module_paths(
    model.target_model,
    [
        "model.layers.*.mlp",
        "model.layers.*.mlp.*proj",
        # "model.layers.*.self_attn", # i think it just uses kwargs. Oh of course it does
        "model.layers.*.self_attn.*proj",
        "model.layers.*.*layernorm",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.3",
        "model.norm",
        "lm_head",
    ],
)

# %%

mask_infos = {
    f"{'cat_' if cat_by_seed[seed] else ''}seed_{seed}": masks_by_seed[seed]
    for seed in masks_by_seed
}

mask_infos = {k: mask_infos[k] for k in sorted(mask_infos.keys())}

# Collect all layer data first
layer_data: dict[str, dict[str, Tensor]] = {}
for path in paths:
    output_caches = {}
    for mask_type, mask_info in mask_infos.items():
        with model.cache_modules([path]) as (_, output_cache):
            model(batch, mask_infos={layer: mask_info})
        output_caches[mask_type] = output_cache[path][0, EXPECTED_POS]
    layer_data[path] = output_caches

# %%

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

    # Add row label on the left
    axes[i, 0].set_ylabel(path, fontsize=12, fontweight="bold")

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

# for mask_type, output_act in layer_data["lm_head"].items():
#     assert output_act.shape == (4096,)
#     sm = output_act.softmax(dim=0)
#     topk_toks = sm.topk(k=10, dim=0)
#     tokenizer_output = tokenizer.convert_ids_to_tokens(topk_toks.indices.cpu().detach().numpy())
#     print(f"{mask_type}:")
#     for token, probability in zip(
#         tokenizer_output, topk_toks.values.cpu().detach().numpy(), strict=True
#     ):
#         print(f"  {token:>10}: {probability:.2f}")


# # %%
# from rich import print as rprint
# from rich.table import Table

# def plot_table():
#     masks_by_seed[seed][layer].component_mask for seed, cat in cat_by_seed.items() if cat], dim=0)

#     table = Table(title="Top 10 Tokens")
#     for seed in masks_by_seed:
#         table.add_column(str(seed))
#         table.add_column("P")

#     toks = {}
#     probs = {}
#     K = 10
#     for mask_type, output_act in layer_data["lm_head"].items():
#         sm = output_act.softmax(dim=0)
#         topk_toks = sm.topk(k=K, dim=0)
#         toks[mask_type] = tokenizer.convert_ids_to_tokens(topk_toks.indices.cpu().detach().numpy())
#         probs[mask_type] = topk_toks.values.cpu().detach().numpy()

#     for i in range(K):
#         data = []
#         for mask_type in mask_infos:
#             data.append(toks[mask_type][i])
#             data.append(f"{probs[mask_type][i]:.2f}")
#         table.add_row(*data)
#     rprint(table)

# plot_table()

# %%

# subplots
fig, axes = plt.subplots(len(mask_infos), len(mask_infos), figsize=(10, 10))
noise_scale = 0.15
alpha = 0.01

for i, seed1 in enumerate(mask_infos):
    for j, seed2 in enumerate(mask_infos):
        c1 = mask_infos[seed1].component_mask[0, EXPECTED_POS]
        # jitter c1
        c1_noise = torch.randn_like(c1) * noise_scale
        c1 = c1 + c1_noise
        c2 = mask_infos[seed2].component_mask[0, EXPECTED_POS]
        c2_noise = torch.randn_like(c2) * noise_scale
        c2 = c2 + c2_noise
        axes[i, j].scatter(c1.cpu().detach().numpy(), c2.cpu().detach().numpy(), alpha=0.007)

plt.show()

# %%

avg_cat = torch.stack(
    [
        masks_by_seed[seed].component_mask[0, EXPECTED_POS]
        for seed in masks_by_seed
        if cat_by_seed[seed]
    ],
).mean(dim=0)

avg_cat_noise = torch.randn_like(avg_cat) * noise_scale  # meow
avg_cat = avg_cat + avg_cat_noise

avg_noncat = torch.stack(
    [
        masks_by_seed[seed].component_mask[0, EXPECTED_POS]
        for seed in masks_by_seed
        if not cat_by_seed[seed]
    ],
).mean(dim=0)

avg_noncat_noise = torch.randn_like(avg_noncat) * noise_scale  # woof
avg_noncat = avg_noncat + avg_noncat_noise

plt.scatter(avg_cat.cpu().detach().numpy(), avg_noncat.cpu().detach().numpy(), alpha=0.3)
plt.xlabel("Catastrophic average mask")
plt.ylabel("Noncatastrophic average mask")
plt.legend(["cat", "noncat"])
plt.show()

# %%


def soft_jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter = (mask_a * mask_b).sum(dim=-1)
    union = mask_a.sum(dim=-1) + mask_b.sum(dim=-1) - inter
    return (inter + eps) / (union + eps)


all_stacked = torch.stack([mi.component_mask[0, EXPECTED_POS] for mi in mask_infos.values()])
soft_jaccard_matrix = soft_jaccard(all_stacked[None], all_stacked[:, None])

plt.imshow(soft_jaccard_matrix.cpu().detach().numpy(), cmap="RdBu", vmin=0, vmax=1)
plt.xticks(range(len(mask_infos)), list(mask_infos.keys()), rotation=45, ha="right")
plt.yticks(range(len(mask_infos)), list(mask_infos.keys()))
plt.title("Soft Jaccard Similarity between different PGD masks")
plt.colorbar()
plt.show()


# %%


def get_hidden_states(mask_infos: ComponentsMaskInfo, paths: list[str]) -> dict[str, Tensor]:
    with model.cache_modules(paths) as (input_cache, output_cache):
        model(batch, mask_infos={layer: mask_infos})
    assert set(input_cache.keys()) == set(paths), (
        f"expected {set(paths)}, got {set(input_cache.keys())}"
    )
    assert set(output_cache.keys()) == set(paths), (
        f"expected {set(paths)}, got {set(output_cache.keys())}"
    )

    layers = {}
    for path in paths:
        layers[f"{path} input"] = input_cache[path][0]
        layers[f"{path} output"] = output_cache[path][0]
    return layers



def compare_mask_infos(
    mi_a: ComponentsMaskInfo,
    mi_b: ComponentsMaskInfo,
    title: str,
    fn: Callable[[Tensor, Tensor], Tensor],
) -> None:
    target_layers = get_hidden_states(mi_a, paths)
    cat_layers = get_hidden_states(mi_b, paths)
    mses: list[Tensor] = []
    for layer in target_layers:
        mses.append(fn(target_layers[layer], cat_layers[layer]))
    mses_matrix = torch.stack(mses)
    fig_height = len(target_layers) * 0.2
    plt.figure(figsize=(40, fig_height))
    plt.imshow(
        mses_matrix.cpu().detach().numpy()[::-1], cmap="Blues", vmin=0, vmax=1, aspect="auto"
    )
    yticks = list(target_layers.keys())[::-1]
    yticks = [y.removeprefix("model.") for y in yticks]
    yticks = [y.removeprefix("layers.") for y in yticks]

    plt.yticks(range(len(target_layers)), yticks)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# %%

mask_infos["target"] = make_mask_infos(
    component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
    weight_deltas_and_masks={
        k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.lower_leaky.items()
    },
)[layer]

# %%

mse = lambda a, b: (a - b).pow(2).mean(dim=-1)
cosine_similarity = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)


for fn in [mse, cosine_similarity]:
    for mi_a, mi_b in [
        ("target", "cat_seed_0"),
        ("cat_seed_0", "cat_seed_15"),
        ("cat_seed_0", "seed_14"),
    ]:
        compare_mask_infos(
            mi_a=mask_infos[mi_a],
            mi_b=mask_infos[mi_b],
            title=f"{fn.__name__} between {mi_a} and {mi_b}",
            fn=fn,
        )

# %%


# from spd.utils.component_utils import calc_stochastic_component_mask_info
# from spd.configs import SamplingType


# def build_mask_infos(
#     *,
#     weight_deltas: dict[str, Tensor],
#     ci: dict[str, Tensor],
#     sampling: SamplingType,
#     rounding_threshold: float,
# ) -> dict[str, dict[str, ComponentsMaskInfo]]:
#     """Build a set of masking strategies consistent with ce_and_kl_losses.

#     Returns mapping: strategy_name -> mask_infos (output of make_mask_infos / stochastic builder)
#     Strategies: ci_masked, stoch_masked, random_masked, rounded_masked
#     """
#     # Build per-layer delta masks using CI leading dims so shapes match (batch, seq_len)

#     # ci_masked_infos = make_mask_infos(component_masks=ci)

#     # rounded_ci = {k: (mod_ci > rounding_threshold).float() for k, mod_ci in ci.items()}
#     # rounded_infos = make_mask_infos(component_masks=rounded_ci)

#     # stoch_infos = calc_stochastic_component_mask_info(
#     #     causal_importances=ci,
#     #     component_mask_sampling=sampling,
#     #     weight_deltas=weight_deltas,
#     #     routing="all",
#     # )
#     return {
#         "target": target_infos,
#         # "ci": ci_masked_infos,
#         # "stoch": stoch_infos,
#         # "rounded": rounded_infos,
#     }

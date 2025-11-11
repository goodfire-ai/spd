# ruff: noqa: E402, I001
# %%

from collections import defaultdict
from jaxtyping import Float
from typing import Any
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.utils.module_utils import get_target_module_paths
from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data, set_seed
from torch import Tensor
from spd.models.components import ComponentsMaskInfo, make_mask_infos

# %%


@dataclass
class ModelSetup:
    path: str
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

    return ModelSetup(path=run_info_path, model=model, config=config, device=device)


# %%

OLD = "wandb:goodfire/spd/runs/lxs77xye"
NEW = "wandb:goodfire/spd/runs/9gf5ud48"

model_setups = {
    path: load_model(path, module_patterns=["model.layers.0.mlp.up_proj"]) for path in [OLD, NEW]
}


# model = model_setup.model
# (layer,) = model.target_module_paths

# config: Config = model_setup.config
# device = model_setup.device
_config = model_setups[OLD].config
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

# %%

data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=_task_config.buffer_size,
    global_seed=_config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)

device = get_device()

data_loader_iter = iter(data_loader)
batch = extract_batch_data(next(data_loader_iter)).to(device)

# %%

target_model_outputs = {
    path: setup.model(batch, cache_type="input") for path, setup in model_setups.items()
}

ci = {
    path: setup.model.calc_causal_importances(
        pre_weight_acts=target_model_outputs[path].cache,
        detach_inputs=False,
        sampling=_config.sampling,
    )
    for path, setup in model_setups.items()
}

weight_deltas = {path: setup.model.calc_weight_deltas() for path, setup in model_setups.items()}


def tw_loss(
    setup: ModelSetup, mask_infos: dict[str, ComponentsMaskInfo], batch: Tensor, target_out: Tensor
) -> Tensor:
    return calc_kl_divergence_lm(
        pred=setup.model(batch, mask_infos=mask_infos),
        target=target_out,
        reduce=False,
    )


# %%

for setup in model_setups.values():
    assert (
        next(iter(model_setups.values())).model.target_module_paths
        == setup.model.target_module_paths
    )

(layer,) = next(iter(model_setups.values())).model.target_module_paths

# %%


def catastrophe(
    setup: ModelSetup,
    batches: list[Tensor],
) -> None:
    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        mask_scope="shared_across_batch",
    )

    assert setup.config.use_delta_component

    for seed, batch in enumerate(batches):
        set_seed(seed)
        assert setup.config.use_delta_component

        target_out = setup.model(batch)
        _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
            model=setup.model,
            batch=batch,
            ci=ci[setup.path].lower_leaky,
            weight_deltas=weight_deltas[setup.path],
            target_out=target_out,
            output_loss_type=setup.config.output_loss_type,
            routing="all",
            pgd_config=pgd_config,
        )

        # pgd_tw_loss = tw_loss(
        #     setup=setup, mask_infos=pgd_mask_infos, batch=batch, target_out=target_out
        # )[0]
        # mean_loss = pgd_tw_loss.mean().item()
        # plt.plot(pgd_tw_loss.cpu().detach().numpy(), alpha=1, linestyle="--")
        # plt.ylim(0, 10)
        # plt.title(f"[{setup.path}] Tokenwise PGD Loss (seed: {seed}, mean: {mean_loss:.2f})")
        # plt.show()

        # seq_len = pgd_tw_loss.shape[0]
        positions = list(range(100, 200))
        n_cols = 10

        _, axes = plt.subplots(
            (len(positions) + 1) // n_cols, n_cols, sharex=True, figsize=(20, 20), squeeze=False
        )
        from tqdm import tqdm

        for ax_idx, i in tqdm(enumerate(positions)):
            pgd_mask = pgd_mask_infos[layer].component_mask
            surgeried_pgd_mask = torch.ones_like(pgd_mask)
            surgeried_pgd_mask[0, i].copy_(pgd_mask[0, i])

            mask_w_single_bit_replaced = ComponentsMaskInfo(
                component_mask=surgeried_pgd_mask,
                routing_mask="all",
                weight_delta_and_mask=pgd_mask_infos[layer].weight_delta_and_mask,
            )

            new_pgd_tw_loss = tw_loss(
                setup=setup,
                mask_infos={layer: mask_w_single_bit_replaced},
                batch=batch,
                target_out=target_out,
            )[0]

            # # =============
            # surgeried_pgd_mask = pgd_mask_infos[layer].component_mask.clone()
            # surgeried_pgd_mask[0, i] = 1

            # mask_w_single_bit_removed = ComponentsMaskInfo(
            #     component_mask=surgeried_pgd_mask,
            #     routing_mask="all",
            #     weight_delta_and_mask=pgd_mask_infos[layer].weight_delta_and_mask,
            # )

            # new_pgd_tw_loss = tw_loss(
            #     setup=setup,
            #     mask_infos={layer: mask_w_single_bit_removed},
            #     batch=batch,
            #     target_out=target_out,
            # )[0]

            # # =============

            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]
            ax.plot(
                new_pgd_tw_loss.cpu().detach().numpy(),
                alpha=1,
                linestyle="--",
                # (pgd_tw_loss - new_pgd_tw_loss).cpu().detach().numpy(), alpha=1, linestyle="--"
            )
            ax.set_ylim(0, 20)

        plt.suptitle(f"[{setup.path}] single token mask replaced PGDLoss (toks 100-200)")
        plt.tight_layout()
        plt.show()

        # print the tokens in the same 10 x 10 grid
        for row in range(n_cols):
            for col in range(n_cols):
                tok_str = tokenizer.decode(batch[0, row * n_cols + col].item())  # pyright: ignore[reportAttributeAccessIssue]
                print(f"{tok_str:<10}", end=" ")
            print()


# %%

batches = [extract_batch_data(next(data_loader_iter)).to(device) for _ in range(10)]

# %%

i = 2
catastrophe(setup=model_setups[OLD], batches=batches[i : i + 1])
catastrophe(setup=model_setups[NEW], batches=batches[i : i + 1])

# %%
pgd_config = PGDReconLossConfig(
    init="random",
    step_size=0.1,
    n_steps=20,
    mask_scope="shared_across_batch",
)

# %%


from tqdm import tqdm


def which_toks(setup: ModelSetup, batches: list[Tensor]) -> dict[int, list[float]]:
    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        mask_scope="shared_across_batch",
    )

    assert setup.config.use_delta_component

    toks_losses = defaultdict(list)

    for seed, batch in tqdm(list(enumerate(batches))):
        set_seed(seed)
        assert setup.config.use_delta_component

        target_out = setup.model(batch)
        _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
            model=setup.model,
            batch=batch,
            ci=ci[setup.path].lower_leaky,
            weight_deltas=weight_deltas[setup.path],
            target_out=target_out,
            output_loss_type=setup.config.output_loss_type,
            routing="all",
            pgd_config=pgd_config,
        )

        for i in range(len(batch[0])):
            pgd_mask = pgd_mask_infos[layer].component_mask
            surgeried_pgd_mask = torch.ones_like(pgd_mask)
            surgeried_pgd_mask[0, i].copy_(pgd_mask[0, i])

            mask_w_single_bit_replaced = ComponentsMaskInfo(
                component_mask=surgeried_pgd_mask,
                routing_mask="all",
                weight_delta_and_mask=pgd_mask_infos[layer].weight_delta_and_mask,
            )

            new_pgd_tw_loss = tw_loss(
                setup=setup,
                mask_infos={layer: mask_w_single_bit_replaced},
                batch=batch,
                target_out=target_out,
            )[0]

            mean_loss = new_pgd_tw_loss.mean().item()
            tok_idx = batch[0, i].item()
            toks_losses[tok_idx].append(mean_loss)

    return toks_losses


# %%
lotsa_batches = [extract_batch_data(next(data_loader_iter)).to(device) for _ in range(200)]
# old_toks = which_toks(setup=model_setups[OLD], batches=lotsa_batches)
new_toks_losses = which_toks(setup=model_setups[NEW], batches=lotsa_batches)

# %%


# def visualize_top_toks(toks_losses: dict[int, list[float]]) -> None:
#     means = {tok_str: sum(losses) / len(losses) for tok_str, losses in toks_losses.items()}

#     sorted_token_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
#     print("top tokens")
#     for tok_str, mean_loss in sorted_token_means[:30]:
#         print(f"{tok_str:<10} {mean_loss:.2f}")

#     print("bottom tokens")
#     for tok_str, mean_loss in sorted_token_means[-30:]:
#         print(f"{tok_str:<10} {mean_loss:.2f}")


# visualize_top_toks(old_toks)
# visualize_top_toks(new_toks)

# %%


# def plot_token_losses(toks_losses: dict[str, list[float]]) -> None:
#     # scatter plot of token losses
#     # x = token, order by mean loss
#     # y = observed losses
#     # Calculate mean for each token and sort by mean
#     # token_means = {tok_str: sum(losses) / len(losses) for tok_str, losses in toks_losses.items()}
#     sorted_tokens = sorted(toks_losses.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

#     # Plot all observed losses
#     x_positions = []
#     y_values = []
#     for idx, (tok_str, losses) in enumerate(sorted_tokens):
#         x_positions.extend([idx] * len(losses))
#         y_values.extend(losses)

#     plt.scatter(x_positions, y_values, alpha=0.2, s=10)

#     # Optionally plot mean line
#     # means = [mean for _, mean in sorted_tokens]
#     # plt.plot(range(len(sorted_tokens)), means, "r-", linewidth=2, label="Mean", alpha=0.2)

#     plt.xlabel("Token (ordered by mean loss)")
#     plt.ylabel("Observed Loss")
#     plt.title("Token Loss Distribution")
#     plt.legend()
#     plt.show()


# plot_token_losses(new_toks)


# %%
tokenizer: Any

# plot histogram of token losses for top 30 tokens
total_n_toks = sum(len(losses) for losses in new_toks_losses.values())
total_n_toks  # pyright: ignore[reportUnusedExpression]

# %%

sorted_new = sorted(new_toks_losses.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

# scatter plot of x: token (sorted by mean loss) and y: number of occurrences of token
x = list(range(len(sorted_new)))
y = [len(losses) for _, losses in sorted_new]

plt.scatter(x, y, alpha=0.4)
plt.xlabel("Token (sorted by mean loss)")
plt.ylabel("Number of occurrences")
plt.yscale("log")
plt.title("Number of occurrences of token by mean loss")
plt.show()

# %%


def top_token_loss_histograms(n: int) -> None:
    for tok_id, losses in sorted_new[:n]:
        plt.hist(losses, bins=100)
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.xlim(0, 2)
        plt.title(f'Token Loss Distribution for "{tokenizer.decode(tok_id)}"')
        plt.show()


top_token_loss_histograms(10)
# %%

# are the worst tokens similar in embedding space?


def pw_cosine_similarity(
    setup: ModelSetup, sorted_token_means: list[tuple[int, float]], indices: list[int]
):
    """indices indexes into the ordered list of tokens by mean loss"""
    token_ids = torch.tensor(
        [sorted_token_means[i][0] for i in indices], device=device, dtype=torch.long
    )
    token_embeddings = setup.model.target_model.model.embed_tokens.weight[token_ids]  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
    a = token_embeddings[None]
    b = token_embeddings[:, None]
    cosine_similarities = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cosine_similarities


# %%
# old_means = {tok_id: sum(losses) / len(losses) for tok_id, losses in old_toks.items()}
# sorted_old = sorted(old_means.items(), key=lambda x: x[1], reverse=True)

new_means = {tok_id: sum(losses) / len(losses) for tok_id, losses in new_toks_losses.items()}
sorted_new = sorted(new_means.items(), key=lambda x: x[1], reverse=True)

# %%

# pos = list(range(50)) + list(range(2000, 2050))

# old_pw = pw_cosine_similarity(model_setups[OLD], pos, sorted_old)
# plt.imshow(old_pw.cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
# plt.xlabel("Token")
# plt.ylabel("Token")
# plt.title("Pairwise Cosine Similarity")
# plt.show()

# # pos = list(range(0, 3000, 50))
# pos = list(range(0, 2000))
# new_pw = pw_cosine_similarity(model_setups[NEW], sorted_new, pos)
# plt.imshow(new_pw.cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
# plt.xlabel("Token")
# plt.ylabel("Token")
# plt.title("Pairwise absolute cosine similarity of top 2000 tokens ordered by pgd loss")
# plt.show()

pos = list(range(0, 100)) + list(range(2000, 2100))
new_pw = pw_cosine_similarity(model_setups[NEW], sorted_new, pos)
plt.imshow(new_pw.cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
# Show sorted indices, but only every 20th to avoid overcrowding
tick_positions = list(range(0, len(pos), 20))
tick_labels = [str(pos[i]) for i in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=90)
plt.yticks(tick_positions, tick_labels)
# Add divider lines between the two groups (after first 100 tokens)
divider_pos = 100 - 0.5  # Position between index 99 and 100
plt.axvline(x=divider_pos, color="black", linewidth=2, linestyle="--")
plt.axhline(y=divider_pos, color="black", linewidth=2, linestyle="--")
plt.xlabel("Sorted Index")
plt.ylabel("Sorted Index")
plt.title("Pairwise cosine similarity of top 100 and 2000-2100 tokens ordered by pgd loss")
plt.show()

# %%


def eval_pgd(setup: ModelSetup, pgd_config: PGDReconLossConfig, batches: list[Tensor]) -> None:
    assert setup.config.use_delta_component

    # 1
    sum_losses = 0
    sum_n_examples = 0

    # 2
    self_calculated_losses = []

    for batch in batches:
        # Compute target output for this batch
        target_out = setup.model(batch, cache_type="input").output

        loss, n_examples, pgd_mask_infos = pgd_masked_recon_loss_update(
            model=setup.model,
            batch=batch,
            ci=ci[setup.path].lower_leaky,
            weight_deltas=weight_deltas[setup.path],
            target_out=target_out,
            output_loss_type=setup.config.output_loss_type,
            routing="all",
            pgd_config=pgd_config,
        )
        sum_losses += loss
        sum_n_examples += n_examples

        self_calculated_losses.append(
            tw_loss(
                setup=setup, mask_infos=pgd_mask_infos, batch=batch, target_out=target_out
            ).mean()
        )

    print(f"Mean given loss: {sum_losses / sum_n_examples}")
    print(f"Mean self-calculated loss: {torch.stack(self_calculated_losses).mean()}")


# %%
eval_pgd(setup=model_setups[OLD], pgd_config=pgd_config, batches=batches)
# %%

# plot inner acts vs CI

# CI is { layer: (b, s, c)}
# inner acts are { layer: (b, s, c)}
# just flatten them all:
all_cis = torch.stack(list(ci.lower_leaky.values())).flatten().cpu().detach()
all_inner_acts = torch.stack(list(ci.inner_acts.values())).flatten().cpu().detach()

plt.scatter(all_inner_acts.numpy(), all_cis.numpy())
plt.xlabel("Inner Acts")
plt.ylabel("CI")
plt.title("CI vs Inner Acts")
plt.show()

# %%

plt.scatter(all_inner_acts.abs().numpy(), all_cis.numpy())
plt.xlabel("|Inner Acts|")
plt.ylabel("CI")
plt.title("CI vs Inner Acts")
plt.show()

# %%


def simple_divergence_pairwise_plots():
    simple_paths = get_target_module_paths(
        model.target_model,
        [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "lm_head",
        ],
    )

    # unnecesarily slow but fine (could just cache at the same time)
    output_caches_by_path: dict[str, dict[str, Tensor]] = {}
    for path in simple_paths:
        output_caches = {}
        for seed_label, mask_info in mask_infos.items():
            with model.cache_modules([path]) as (_, output_cache):
                model(batch, mask_infos={layer: mask_info})
            output_caches[seed_label] = output_cache[path][0, EXPECTED_POS]
        output_caches_by_path[path] = output_caches

    # Create multi-panel plot
    n_layers = len(simple_paths)
    labels = list(next(iter(output_caches_by_path.values())).keys())

    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))

    # Ensure axes is 2D even for single layer
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for i, path in enumerate(simple_paths):
        output_caches = output_caches_by_path[path]
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


simple_divergence_pairwise_plots()


# %%
def scatter_plots():
    # subplots
    _, axes = plt.subplots(len(mask_infos), len(mask_infos), figsize=(10, 10))
    noise_scale = 0.15

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
scatter_plots()
# %%


def avg_scatter_plots():
    noise_scale = 0.15

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
avg_scatter_plots()
# %%


def soft_jaccard(
    mask_a: Float[Tensor, "... d"],
    mask_b: Float[Tensor, "... d"],
    eps: float = 1e-7,
) -> Float[Tensor, "... d"]:
    inter = (mask_a * mask_b).sum(dim=-1)
    union = mask_a.sum(dim=-1) + mask_b.sum(dim=-1) - inter
    return (inter + eps) / (union + eps)


def plot_soft_jaccard_matrix():
    all_stacked = torch.stack([mi.component_mask[0, EXPECTED_POS] for mi in mask_infos.values()])
    soft_jaccard_matrix = soft_jaccard(all_stacked[None], all_stacked[:, None])

    plt.imshow(soft_jaccard_matrix.cpu().detach().numpy(), cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(mask_infos)), list(mask_infos.keys()), rotation=45, ha="right")
    plt.yticks(range(len(mask_infos)), list(mask_infos.keys()))
    plt.title("Soft Jaccard Similarity between different PGD masks")
    plt.colorbar()
    plt.show()


plot_soft_jaccard_matrix()
# %%


def plot_l1_norms():
    l1_norms = [
        mi.component_mask[0, EXPECTED_POS].norm(dim=-1).item() for mi in mask_infos.values()
    ]
    plt.bar(range(len(mask_infos)), l1_norms)
    plt.xticks(range(len(mask_infos)), list(mask_infos.keys()), rotation=45, ha="right")
    plt.ylabel("L1 norm")
    plt.title("L1 norm of different PGD masks")
    plt.show()


# %%
plot_l1_norms()
# %%


def get_layer_outputs(mask_infos: ComponentsMaskInfo, paths: list[str]) -> dict[str, Tensor]:
    with model.cache_modules(paths) as (_, output_cache):
        model(batch, mask_infos={layer: mask_infos})
    layers = {}
    for path in paths:
        layers[path] = output_cache[path][0]
    return layers


# %%
mask_infos["target"] = make_mask_infos(
    component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
    weight_deltas_and_masks={
        k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.lower_leaky.items()
    },
)[layer]

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


def visualize_seq_layer_metric(
    vals: dict[str, Tensor],
    title: str,
    imshow_kwargs: dict[str, Any] | None = None,
) -> None:
    layer_names = sorted(vals.keys(), key=sort_layer)
    matrix = torch.stack([vals[layer] for layer in layer_names])
    fig_height = len(layer_names) * 0.2
    plt.figure(figsize=(25, fig_height))
    plt.imshow(matrix.cpu().detach().numpy()[::-1], aspect="auto", **(imshow_kwargs or {}))
    plt.yticks(range(len(layer_names)), layer_names[::-1], fontsize=10)
    plt.title(title, fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# %%


def mse(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).pow(2).mean(dim=-1)


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def l2_norm_ratio(a: Tensor, b: Tensor) -> Tensor:
    return torch.norm(a, dim=-1) / torch.norm(b, dim=-1)


def plot_metrics():
    paths = get_target_module_paths(
        model.target_model,
        [
            # "model.layers.*.mlp",
            "model.layers.*.mlp.down_proj",
            # "model.layers.*.mlp.*proj",
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

    for fn, title, imshow_kwargs in [
        (mse, "mse", {"cmap": "Reds", "vmin": 0}),
        (cosine_similarity, "cosine similarity", {"cmap": "RdBu", "vmin": -1, "vmax": 1}),
        (l2_norm_ratio, "l2(a)/l2(b)", {"cmap": "RdBu", "vmin": 0, "vmax": 2}),
    ]:
        for mi_a, mi_b in [
            ("target", "cat_seed_0"),
            # ("target", "seed_20"),
            # ("cat_seed_0", "seed_20"),
            # ("cat_seed_0", "cat_seed_15"),
            # ("cat_seed_0", "seed_14"),
        ]:
            target_layers = get_layer_outputs(mask_infos[mi_a], paths)
            cat_layers = get_layer_outputs(mask_infos[mi_b], paths)
            vals = {layer: fn(target_layers[layer], cat_layers[layer]) for layer in paths}
            visualize_seq_layer_metric(
                vals,
                title=f"{title} between {mi_a} and {mi_b}",
                imshow_kwargs=imshow_kwargs,
            )


# %%
plot_metrics()

# %%
# hypothesis: PGD causes the model to just predict the current token mostly
# let's visualize this with a historgram of "current token probability"
target_output = model(batch)[0]
target_log_probs = target_output.log_softmax(dim=-1)

pgd_output = model(batch, mask_infos={layer: mask_infos["cat_seed_0"]})[0]
pgd_log_probs = pgd_output.log_softmax(dim=-1)


def current_token_probs_histogram():
    S = batch.shape[1]
    target_current_token_probs = target_log_probs[torch.arange(S), batch[0]]
    pgd_current_token_probs = pgd_log_probs[torch.arange(S), batch[0]]
    plt.hist(
        target_current_token_probs[108:150].cpu().detach().numpy(),
        alpha=0.5,
        label="target",
        bins=100,
        range=(-20, 0),
    )
    plt.title("Target current token log probs")
    plt.legend()
    plt.show()

    plt.hist(
        pgd_current_token_probs[108:150].cpu().detach().numpy(),
        alpha=0.5,
        label="pgd",
        bins=100,
        range=(-20, 0),
    )
    plt.title("PGD current token log probs")
    plt.legend()
    plt.show()


current_token_probs_histogram()
# %%

from circuitsvis import logits

logits.token_log_probs(
    token_indices=batch[0],
    log_probs=target_output.log_softmax(dim=-1),
    to_string=tokenizer.decode,  # pyright: ignore[reportAttributeAccessIssue]
    top_k=10,
)

# %%
logits.token_log_probs(
    token_indices=batch[0],
    log_probs=pgd_output.log_softmax(dim=-1),
    to_string=tokenizer.decode,  # pyright: ignore[reportAttributeAccessIssue]
    top_k=10,
)

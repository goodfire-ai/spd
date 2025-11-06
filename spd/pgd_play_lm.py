# ruff: noqa: E402, I001
# %%

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


def test_sparse_pgd():
    sparsity_coeffs = [None, 0.3, 0.52, 1, 1.7, 3, 5.2, 10, 17, 30, 52]

    l1s: list[list[float]] = []
    losses: list[list[float]] = []

    for sparsity_coeff in sparsity_coeffs:
        sparsity_l1s = []
        sparsity_losses = []
        for seed in range(10):
            set_seed(seed)
            pgd_config = PGDReconLossConfig(
                init="random",
                step_size=0.04,
                n_steps=50,
                mask_scope="shared_across_batch",
                step_type="signed-gradient",
                mask_sparsity_coeff=sparsity_coeff,
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

            l1 = pgd_mask_infos[layer].component_mask[0].sum(dim=-1).mean(dim=0).item()

            sparsity_l1s.append(l1)
            sparsity_losses.append(pgd_tw_loss.mean().item())

        l1s.append(sparsity_l1s)
        losses.append(sparsity_losses)

    # plot each sparsity coeff as a colored scatter group
    plt.scatter(
        [sum(l1s_) / len(l1s_) for l1s_ in l1s],
        [sum(losses_) / len(losses_) for losses_ in losses],
    )
    plt.xlabel("L1 norm")
    plt.ylabel("PGD loss")
    plt.title("PGD loss vs L1 norm")
    plt.show()


# %%
test_sparse_pgd()
# %%


def get_mask_infos_by_seed():
    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.04,
        n_steps=50,
        mask_scope="shared_across_batch",
        step_type="signed-gradient",
    )

    full_pgd_masks_by_seed: dict[int, ComponentsMaskInfo] = {}
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
        full_pgd_masks_by_seed[seed] = pgd_mask_infos[layer]

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
        mask_w_single_bit_replaced.weight_delta_and_mask = None

        masks_by_seed[seed] = mask_w_single_bit_replaced

        new_pgd_tw_loss = tw_loss(mask_infos={layer: mask_w_single_bit_replaced})[0]
        is_catastrophic = new_pgd_tw_loss.mean().item() > 1
        cat_by_seed[seed] = is_catastrophic
        plt.plot(new_pgd_tw_loss.cpu().detach().numpy(), alpha=1, linestyle="--")
        plt.title(
            f"Tokenwise PGD Loss with single (token, layer) mask replaced (seed: {seed}, catastrophic: {is_catastrophic})"
        )
        plt.show()

    return masks_by_seed, cat_by_seed


masks_by_seed, cat_by_seed = get_mask_infos_by_seed()

mask_infos = {
    f"{'cat_' if cat_by_seed[seed] else ''}seed_{seed}": masks_by_seed[seed]
    for seed in masks_by_seed
}

mask_infos = {k: mask_infos[k] for k in sorted(mask_infos.keys())}

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

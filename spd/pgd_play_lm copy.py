# ruff: noqa: E402, I001
# %%

import einops
import numpy as np
from collections import defaultdict
from collections.abc import Iterator
from jaxtyping import Float, Int
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import CIOutputs, ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data, set_seed
from torch import Tensor
from spd.models.components import ComponentsMaskInfo

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
MLP_UP_0 = "model.layers.0.mlp.up_proj"

model_setups = {path: load_model(path, module_patterns=[MLP_UP_0]) for path in [OLD, NEW]}

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

tokenizer: Any
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=_task_config.buffer_size,
    global_seed=_config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_loader_iter = iter(data_loader)

device = get_device()


# %%

pgd_config = PGDReconLossConfig(
    init="random",
    step_size=0.04,
    n_steps=50,
    mask_scope="shared_across_batch",
)

# %%


def catastrophe(setup: ModelSetup, n_batches: int):
    B = 1
    assert isinstance(_task_config, LMTaskConfig), "task_config not LMTaskConfig"

    data_loader, tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=32,
        buffer_size=_task_config.buffer_size,
        global_seed=_config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    data_loader_iter = iter(data_loader)

    assert setup.config.use_delta_component

    for i in range(n_batches):
        try:
            batch = extract_batch_data(next(data_loader_iter)).to(device)
        except StopIteration:
            for _ in range(10):
                print("=" * 100)
            print(f"STOPITER at batch {i}")
            for _ in range(10):
                print("=" * 100)
            break
        S = batch.shape[1]

        set_seed(i)

        assert setup.config.use_delta_component

        target_output = setup.model(batch, cache_type="input")

        ci = setup.model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=setup.config.sampling,
        )

        _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
            model=setup.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=setup.model.calc_weight_deltas(),
            target_out=target_output.output,
            output_loss_type=setup.config.output_loss_type,
            routing="all",
            pgd_config=pgd_config,
        )
        pgd_mask_info = pgd_mask_infos[MLP_UP_0]

        # plot the overall pgd loss
        full_pgd_output = setup.model(batch, mask_infos={MLP_UP_0: pgd_mask_infos[MLP_UP_0]})
        full_pgd_loss = calc_kl_divergence_lm(full_pgd_output, target_output.output, reduce=False)[
            0
        ]
        yield full_pgd_loss
        continue
        # plt.plot(full_pgd_loss.cpu().detach().numpy())
        # plt.title("Overall PGD Loss")
        # plt.xlabel("Sequence Index")
        # plt.ylabel("Loss")
        # plt.show()
        # return

        n_cols = 10
        start, stop = 100, 200
        _, axes = plt.subplots(10, 10, sharex=True, figsize=(20, 20), squeeze=False)
        for seq_idx in range(start, stop):
            routing_mask = torch.zeros(B, S, device=device, dtype=torch.bool)
            routing_mask[:, seq_idx] = True

            mask_w_single_bit_replaced = ComponentsMaskInfo(
                component_mask=pgd_mask_info.component_mask,
                routing_mask=routing_mask,
                weight_delta_and_mask=pgd_mask_info.weight_delta_and_mask,
            )

            output = setup.model(batch, mask_infos={MLP_UP_0: mask_w_single_bit_replaced})
            new_pgd_tw_loss = calc_kl_divergence_lm(output, target_output.output, reduce=False)

            ax_idx = seq_idx - start
            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]
            ax.plot(new_pgd_tw_loss[0].cpu().detach().numpy(), linestyle="--")
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

# catastrophe(setup=model_setups[OLD], n_batches=1)

for loss in catastrophe(setup=model_setups[NEW], n_batches=10):
    print(loss.mean().item())
    # plt.plot(loss.cpu().detach().numpy())
    # plt.show()

# %%


def which_toks(setup: ModelSetup, n_batches: int, batch_size: int) -> dict[int, list[float]]:
    assert isinstance(_task_config, LMTaskConfig), "task_config not LMTaskConfig"
    data_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=_task_config.buffer_size,
        global_seed=_config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )
    data_iter = iter(data_loader)

    assert setup.config.use_delta_component

    toks_losses = defaultdict(list)

    for i in tqdm(range(n_batches)):
        set_seed(i)  # <------------- NOTE THIS
        batch = extract_batch_data(next(data_iter)).to(device)


        target_output = setup.model(batch, cache_type="input")

        ci_batch = setup.model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=setup.config.sampling,
        ).lower_leaky

        _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
            model=setup.model,
            batch=batch,
            ci=ci_batch,
            weight_deltas=setup.model.calc_weight_deltas(),
            target_out=target_output.output,
            output_loss_type=setup.config.output_loss_type,
            routing="all",
            pgd_config=pgd_config,
        )

        _, S = batch.shape
        for seq_idx in tqdm(range(S), desc="Sequence index"):
            pgd_mask = pgd_mask_infos[MLP_UP_0].component_mask
            wdm = pgd_mask_infos[MLP_UP_0].weight_delta_and_mask
            surgeried_pgd_mask = torch.ones_like(pgd_mask)
            surgeried_pgd_mask[:, seq_idx].copy_(pgd_mask[:, seq_idx])
            mask_w_single_bit_replaced = ComponentsMaskInfo(
                component_mask=surgeried_pgd_mask,
                routing_mask="all",
                weight_delta_and_mask=wdm,
            )

            pred = setup.model(batch, mask_infos={MLP_UP_0: mask_w_single_bit_replaced})
            new_pgd_loss_BS = calc_kl_divergence_lm(pred, target_output.output, reduce=False)

            toks_replaced_B = batch[:, seq_idx]  # (B,)
            batch_means_B = new_pgd_loss_BS.mean(dim=1)

            for tok, batch_mean in zip(toks_replaced_B, batch_means_B, strict=True):
                toks_losses[tok.item()].append(batch_mean.item())

    return toks_losses


# %%

new_toks_losses = which_toks(setup=model_setups[NEW], n_batches=4, batch_size=32)
total_n_toks = sum(len(losses) for losses in new_toks_losses.values())


# %%


# %%
def n_samples_histogram(toks_losses: dict[int, list[float]]) -> None:
    n_samples = [len(losses) for losses in toks_losses.values()]
    plt.hist(n_samples, bins=100, log=True)
    plt.xlabel("Number of samples")
    plt.ylabel("Frequency")
    plt.title("Number of samples per token")
    plt.show()


n_samples_histogram(new_toks_losses)

# %%


def visualize_top_toks(toks_losses: dict[int, list[float]]) -> None:
    means = {tok_str: sum(losses) / len(losses) for tok_str, losses in toks_losses.items()}

    sorted_token_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
    print("top toks" + " " * 30 + "bottom toks")
    for (top_tok_idx, top_mean_loss), (bottom_tok_idx, bottom_mean_loss) in zip(
        sorted_token_means[:30], sorted_token_means[-30:], strict=True
    ):
        top_tok_str = tokenizer.decode(top_tok_idx)
        bottom_tok_str = tokenizer.decode(bottom_tok_idx)
        print(
            f"{top_tok_str:<10} {top_mean_loss:<30.2f} {bottom_tok_str:<10} {bottom_mean_loss:<10.2f}"
        )


visualize_top_toks(new_toks_losses)

# %%


def plot_token_losses(toks_losses: dict[int, list[float]]) -> None:
    # scatter plot of token losses
    # x = token, order by mean loss
    # y = observed losses
    # Calculate mean for each token and sort by mean
    # token_means = {tok_str: sum(losses) / len(losses) for tok_str, losses in toks_losses.items()}
    sorted_tokens = sorted(toks_losses.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

    # Plot all observed losses
    x_positions = []
    y_values = []
    for idx, (_, losses) in enumerate(sorted_tokens):
        x_positions.extend([idx] * len(losses))
        y_values.extend(losses)

    plt.scatter(x_positions, y_values, alpha=0.2, s=10)

    # Optionally plot mean line
    means = [sum(losses) / len(losses) for _, losses in sorted_tokens]
    plt.plot(range(len(sorted_tokens)), means, "r-", linewidth=2, label="Mean", alpha=0.2)

    plt.xlabel("Token (ordered by mean loss)")
    plt.ylabel("Observed Loss")
    plt.title("Token Loss Distribution")
    plt.legend()
    plt.show()


plot_token_losses(new_toks_losses)


# %%
# %%
def asdf():
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


# old_means = {tok_id: sum(losses) / len(losses) for tok_id, losses in old_toks.items()}
# sorted_old = sorted(old_means.items(), key=lambda x: x[1], reverse=True)


# %%
def something_else():
    new_means = {tok_id: sum(losses) / len(losses) for tok_id, losses in new_toks_losses.items()}
    sorted_new = sorted(new_means.items(), key=lambda x: x[1], reverse=True)

    # pos = list(range(0, 3000, 50))
    pos = list(range(0, 2000))
    new_pw = pw_cosine_similarity(model_setups[NEW], sorted_new, pos)
    plt.imshow(new_pw.cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.title("Pairwise absolute cosine similarity of top 2000 tokens ordered by pgd loss")
    plt.show()

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


something_else()

# %%


# train linear regression model to predict mean loss from embedding (basically just a linear probe)
# importantly we don't use sklearn here, dependency-light
def fit_linear_regression(
    embeddings: list[Tensor], mean_losses: list[float]
) -> tuple[Tensor, Tensor]:
    """
    Fit a linear regression y ≈ X @ w + b to predict mean_losses from embeddings.
    Returns a parameter vector of shape (d + 1,), where the last entry is the bias b.
    """
    assert len(embeddings) == len(mean_losses), "Mismatched lengths"
    assert len(embeddings) > 0, "Empty inputs"

    X = torch.stack(embeddings, dim=0)
    device = X.device
    dtype = X.dtype

    y = torch.as_tensor(mean_losses, device=device, dtype=dtype).reshape(-1, 1)
    ones = torch.ones((X.shape[0], 1), device=device, dtype=dtype)
    X_aug = torch.cat([X, ones], dim=1)

    solution = torch.linalg.lstsq(X_aug, y).solution  # (d+1, 1)
    assert solution.shape[1] == 1
    return (
        solution[:-1, 0],
        solution[-1, 0],
    )


def do_linear():
    sorted_new = sorted(new_toks_losses.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    mean_losses = [sum(losses) / len(losses) for _, losses in sorted_new]

    model_emb = model_setups[NEW].model.target_model.model.embed_tokens.weight  # pyright: ignore[reportAttributeAccessIssue]
    embeddings = [
        model_emb[tok_id]  # pyright: ignore[reportIndexIssue]
        for tok_id, _ in sorted_new
    ]

    W, b = fit_linear_regression(embeddings, mean_losses)

    # plot real against predicted loss
    t_embeddings = torch.stack(embeddings, dim=0)

    predicted_losses = einops.einsum(t_embeddings, W, "b d, d -> b") + b
    # Prepare arrays
    true_losses = np.array(mean_losses)
    pred_losses = predicted_losses.cpu().detach().numpy()

    # Scatter
    plt.scatter(true_losses, pred_losses, alpha=0.08, label="data")

    # R^2
    sst = ((true_losses - true_losses.mean()) ** 2).sum()
    ssr = ((true_losses - pred_losses) ** 2).sum()
    r2 = 1.0 - (ssr / sst) if sst > 0 else float("nan")
    ax = plt.gca()
    ax.text(
        0.02,
        0.98,
        f"R² = {r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.xlabel("True mean loss")
    plt.ylabel("Predicted mean loss")
    plt.title("Predicted vs true mean loss (linear probe)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()


# %%


def get_many_cis(setup: ModelSetup, data_loader_iter: Iterator[Tensor], nbatches: int) -> CIOutputs:
    batch_cis: list[CIOutputs] = []
    for _ in range(nbatches):
        batch = extract_batch_data(next(data_loader_iter)).to(device)
        target_output = setup.model(batch, cache_type="input")
        ci = setup.model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=setup.config.sampling,
        )
        batch_cis.append(ci)
    return CIOutputs(
        lower_leaky={
            k: torch.stack([ci.lower_leaky[k] for ci in batch_cis])
            for k in batch_cis[0].lower_leaky
        },
        upper_leaky={
            k: torch.stack([ci.upper_leaky[k] for ci in batch_cis])
            for k in batch_cis[0].upper_leaky
        },
        pre_sigmoid={
            k: torch.stack([ci.pre_sigmoid[k] for ci in batch_cis])
            for k in batch_cis[0].pre_sigmoid
        },
        inner_acts={
            k: torch.stack([ci.inner_acts[k] for ci in batch_cis]) for k in batch_cis[0].inner_acts
        },
    )


def plot_ci_against_inner_acts(ci: CIOutputs, title: str) -> None:
    all_cis = torch.stack(list(ci.lower_leaky.values())).flatten().cpu().detach()
    all_inner_acts_abs = torch.stack(list(ci.inner_acts.values())).abs().flatten().cpu().detach()
    plt.scatter(all_inner_acts_abs.numpy(), all_cis.numpy(), alpha=0.1)
    plt.xlabel("abs(Inner Acts)")
    plt.ylabel("CI")
    plt.title(f"{title}: abs(Inner Acts) vs CI")
    plt.show()


# %%
# %%
plot_ci_against_inner_acts(
    get_many_cis(model_setups[NEW], data_loader_iter, 4), title=model_setups[NEW].path
)

# %%

plot_ci_against_inner_acts(
    get_many_cis(model_setups[OLD], data_loader_iter, 4), title=model_setups[OLD].path
)
# %%


def plot_ci_histogram(ci: CIOutputs, title: str) -> None:
    all_cis = torch.stack(list(ci.lower_leaky.values())).flatten().cpu().detach()
    # hist of CI vals
    plt.hist(all_cis.numpy(), bins=100, log=True)
    plt.xlabel("CI")
    plt.ylabel("Frequency")
    plt.ylim(1, 10**7)
    plt.title(f"{title}: CI histogram")
    plt.show()


def soft_jaccard(
    mask_a: Float[Tensor, "... d"],
    mask_b: Float[Tensor, "... d"],
    eps: float = 1e-7,
) -> Float[Tensor, "... d"]:
    inter = (mask_a * mask_b).sum(dim=-1)
    union = mask_a.sum(dim=-1) + mask_b.sum(dim=-1) - inter
    return (inter + eps) / (union + eps)


# %%

"""
We want to optimize pgd masks without the batch positions that cause catastrophic cascading loss.

1. find positions
2. optimize pgd masks without those positions
"""


def find_catastrophic_positions(
    setup: ModelSetup, batch: Int[Tensor, "1 seq"], start_idx: int, end_idx: int
):
    target_output = setup.model(batch, cache_type="input")
    ci = setup.model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=setup.config.sampling,
    )

    full_pgd_mask_info = pgd_masked_recon_loss_update(
        model=setup.model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=setup.model.calc_weight_deltas(),
        target_out=target_output.output,
        output_loss_type=setup.config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
    )[2][MLP_UP_0]

    pgd_mask = full_pgd_mask_info.component_mask

    assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"

    for seq_idx in range(start_idx, end_idx):
        one_pgd_idx_mask = torch.ones_like(pgd_mask)
        one_pgd_idx_mask[:, seq_idx] = pgd_mask[:, seq_idx]

        mask_w_pos_replaced = ComponentsMaskInfo(
            component_mask=one_pgd_idx_mask,
            routing_mask="all",
            weight_delta_and_mask=None,  # full_pgd_mask_info.weight_delta_and_mask,
        )

        patched_output = setup.model(batch, mask_infos={MLP_UP_0: mask_w_pos_replaced})
        patched_loss = calc_kl_divergence_lm(patched_output, target_output.output, reduce=False)
        yield patched_loss[0]


# %%
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=_task_config.buffer_size,
    global_seed=_config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_loader_iter = iter(data_loader)

loss_means = []
for _ in tqdm(range(30)):
    batch = extract_batch_data(next(data_loader_iter)).to(device)
    loss_means.extend(
        losses.mean(dim=0).item()
        for losses in find_catastrophic_positions(model_setups[OLD], batch, 0, 512)
    )

plt.hist(loss_means, log=True)
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.title("Mean Loss Distribution")
plt.show()

# _, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True, sharey=True)

# for i, loss in enumerate(loss_sets):
#     row, col = i // 10, i % 10
#     axes[row, col].plot(loss.cpu().detach().numpy())

# plt.show()


# %%
def pgd_without_pos(setup: ModelSetup, batch: Int[Tensor, "1 seq"], pos_idx: int):
    target_output = setup.model(batch, cache_type="input")
    ci = setup.model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=setup.config.sampling,
    )

    def callback(sources: Tensor) -> None:
        assert sources.shape == (1, 1, 512, 1201), (
            f"sources must be of shape (1, 1, 512, 1201), got {sources.shape}"
        )
        out = sources.clone()
        out[0, 0, pos_idx, :] = 1.0
        return out

    sum_loss, n_ex, _ = pgd_masked_recon_loss_update(
        model=setup.model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=setup.model.calc_weight_deltas(),
        target_out=target_output.output,
        output_loss_type=setup.config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
        step_callback=callback,
    )

    return sum_loss / n_ex


# %%
loss_means = []

bef_aft: list[tuple[float, float]] = []

for _ in tqdm(range(30)):
    batch = extract_batch_data(next(data_loader_iter)).to(device)
    for pos_idx, losses in enumerate(
        find_catastrophic_positions(model_setups[OLD], batch, 0, 512),
    ):
        mean_loss = losses.mean(dim=0).item()
        pos_is_catastrophic = mean_loss > 0.3
        if not pos_is_catastrophic:
            continue

        new_mean_loss = pgd_without_pos(model_setups[OLD], batch, pos_idx)
        print(f"loss: {mean_loss}, excluding pos {pos_idx}: {new_mean_loss}")

        bef_aft.append((mean_loss, new_mean_loss.item()))

# %%

plt.scatter(
    [bef for bef, _ in bef_aft],
    [aft for _, aft in bef_aft],
    alpha=0.1,
)
plt.xlabel("Before")
plt.ylabel("After")
plt.title("Before and After PGD Loss")
plt.show()

# %%

# def get_layer_outputs(setup: ModelSetup, mask_infos: ComponentsMaskInfo, paths: list[str]) ->
# dict[str, Tensor]:

#     with model.cache_modules(paths) as (_, output_cache):
#         model(batch, mask_infos={MLP_UP_0: mask_infos})
#     layers = {}
#     for path in paths:
#         layers[path] = output_cache[path][0]
#     return layers


# # %%
# mask_infos["target"] = make_mask_infos(
#     component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
#     weight_deltas_and_masks={
#         k: (weight_deltas[k], torch.ones_like(v[..., 0])) for k, v in ci.lower_leaky.items()
#     },
# )[MLP_UP_0]

# # %%


# def sort_layer(lname: str):
#     if lname == "lm_head":
#         return 10_000
#     if lname == "model.norm":
#         return 9_999

#     assert lname.startswith("model.layers."), f"Unknown layer: {lname}"
#     lname = lname.removeprefix("model.layers.")

#     if "." in lname:
#         layer_str, rest = lname.split(".", maxsplit=1)
#         layer = int(layer_str)
#         assert layer in [0, 1, 2, 3], f"Unknown layer: {lname}"
#         if rest == "input_layernorm":
#             return (layer * 1000) + 1
#         if rest == "self_attn.q_proj":
#             return (layer * 1000) + 2
#         if rest == "self_attn.k_proj":
#             return (layer * 1000) + 3
#         if rest == "self_attn.v_proj":
#             return (layer * 1000) + 4
#         if rest == "self_attn.o_proj":
#             return (layer * 1000) + 5
#         if rest == "post_attention_layernorm":
#             return (layer * 1000) + 6
#         if rest == "mlp.down_proj":
#             return (layer * 1000) + 7

#     return (int(lname) * 1000) + 8


# def visualize_seq_layer_metric(
#     vals: dict[str, Tensor],
#     title: str,
#     imshow_kwargs: dict[str, Any] | None = None,
# ) -> None:
#     layer_names = sorted(vals.keys(), key=sort_layer)
#     matrix = torch.stack([vals[layer] for layer in layer_names])
#     fig_height = len(layer_names) * 0.2
#     plt.figure(figsize=(25, fig_height))
#     plt.imshow(matrix.cpu().detach().numpy()[::-1], aspect="auto", **(imshow_kwargs or {}))
#     plt.yticks(range(len(layer_names)), layer_names[::-1], fontsize=10)
#     plt.title(title, fontsize=20)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()


# # %%


# def mse(a: Tensor, b: Tensor) -> Tensor:
#     return (a - b).pow(2).mean(dim=-1)


# def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
#     return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# def l2_norm_ratio(a: Tensor, b: Tensor) -> Tensor:
#     return torch.norm(a, dim=-1) / torch.norm(b, dim=-1)


# def plot_metrics():
#     paths = get_target_module_paths(
#         model.target_model,
#         [
#             # "model.layers.*.mlp",
#             "model.layers.*.mlp.down_proj",
#             # "model.layers.*.mlp.*proj",
#             # "model.layers.*.self_attn", # i think it just uses kwargs. Oh of course it does
#             "model.layers.*.self_attn.*proj",
#             "model.layers.*.*layernorm",
#             "model.layers.0",
#             "model.layers.1",
#             "model.layers.2",
#             "model.layers.3",
#             "model.norm",
#             "lm_head",
#         ],
#     )

#     for fn, title, imshow_kwargs in [
#         (mse, "mse", {"cmap": "Reds", "vmin": 0}),
#         (cosine_similarity, "cosine similarity", {"cmap": "RdBu", "vmin": -1, "vmax": 1}),
#         (l2_norm_ratio, "l2(a)/l2(b)", {"cmap": "RdBu", "vmin": 0, "vmax": 2}),
#     ]:
#         for mi_a, mi_b in [
#             ("target", "cat_seed_0"),
#             # ("target", "seed_20"),
#             # ("cat_seed_0", "seed_20"),
#             # ("cat_seed_0", "cat_seed_15"),
#             # ("cat_seed_0", "seed_14"),
#         ]:
#             target_layers = get_layer_outputs(mask_infos[mi_a], paths)
#             cat_layers = get_layer_outputs(mask_infos[mi_b], paths)
#             vals = {layer: fn(target_layers[layer], cat_layers[layer]) for layer in paths}
#             visualize_seq_layer_metric(
#                 vals,
#                 title=f"{title} between {mi_a} and {mi_b}",
#                 imshow_kwargs=imshow_kwargs,
#             )


# # %%
# plot_metrics()

# # %%
# # hypothesis: PGD causes the model to just predict the current token mostly
# # let's visualize this with a historgram of "current token probability"
# target_output = model(batch)[0]
# target_log_probs = target_output.log_softmax(dim=-1)

# pgd_output = model(batch, mask_infos={MLP_UP_0: mask_infos["cat_seed_0"]})[0]
# pgd_log_probs = pgd_output.log_softmax(dim=-1)


# def current_token_probs_histogram():
#     S = batch.shape[1]
#     target_current_token_probs = target_log_probs[torch.arange(S), batch[0]]
#     pgd_current_token_probs = pgd_log_probs[torch.arange(S), batch[0]]
#     plt.hist(
#         target_current_token_probs[108:150].cpu().detach().numpy(),
#         alpha=0.5,
#         label="target",
#         bins=100,
#         range=(-20, 0),
#     )
#     plt.title("Target current token log probs")
#     plt.legend()
#     plt.show()

#     plt.hist(
#         pgd_current_token_probs[108:150].cpu().detach().numpy(),
#         alpha=0.5,
#         label="pgd",
#         bins=100,
#         range=(-20, 0),
#     )
#     plt.title("PGD current token log probs")
#     plt.legend()
#     plt.show()


# current_token_probs_histogram()
# # %%

# from circuitsvis import logits

# logits.token_log_probs(
#     token_indices=batch[0],
#     log_probs=target_output.log_softmax(dim=-1),
#     to_string=tokenizer.decode,  # pyright: ignore[reportAttributeAccessIssue]
#     top_k=10,
# )

# # %%
# logits.token_log_probs(
#     token_indices=batch[0],
#     log_probs=pgd_output.log_softmax(dim=-1),
#     to_string=tokenizer.decode,  # pyright: ignore[reportAttributeAccessIssue]
#     top_k=10,
# )

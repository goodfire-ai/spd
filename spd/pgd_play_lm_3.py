# ruff: noqa: E402, I001
# %%
import numpy as np
from jaxtyping import Int
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data, set_seed
from torch import Tensor
from spd.models.components import ComponentsMaskInfo

# %%


device = get_device()



# %%

def run_for_layer(layer: str):



OLD = "wandb:goodfire/spd/runs/lxs77xye"
NEW = "wandb:goodfire/spd/runs/9gf5ud48"
# MLP_UP_0 = "model.layers.0.mlp.up_proj"
MLP_DOWN_1 = "model.layers.1.mlp.down_proj"

model_setups = {path: load_model(path) for path in [OLD, NEW]}

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
    step_size=0.1,
    n_steps=20,
    mask_scope="shared_across_batch",
)

"""
We want to optimize pgd masks without the batch positions that cause catastrophic cascading loss.

1. find positions
2. optimize pgd masks without those positions
"""


def get_losses_by_pos(setup: ModelSetup, batch: Int[Tensor, "1 seq"]):
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
    )[2][MLP_DOWN_1]

    assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"

    for seq_idx in range(batch.shape[1]):
        routing_mask = torch.zeros_like(batch, dtype=torch.bool)
        routing_mask[:, seq_idx] = True

        mask_w_pos_replaced = ComponentsMaskInfo(
            component_mask=full_pgd_mask_info.component_mask,
            routing_mask=routing_mask,
            weight_delta_and_mask=None,  # full_pgd_mask_info.weight_delta_and_mask,
        )

        patched_output = setup.model(batch, mask_infos={MLP_DOWN_1: mask_w_pos_replaced})
        patched_loss = calc_kl_divergence_lm(patched_output, target_output.output, reduce=False)
        yield patched_loss[0]


# %%


def pgd_without_positions(setup: ModelSetup, batch: Int[Tensor, "1 seq"], pos_indices: list[int]):
    target_output = setup.model(batch, cache_type="input")
    ci = setup.model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=setup.config.sampling,
    )

    def one_out_positions(exp_sources: Tensor) -> Tensor:
        assert exp_sources.shape == (1, 1, 512, 1201), (
            f"exp_sources must be of shape (1, 1, 512, 1201), got {exp_sources.shape}"
        )
        out = exp_sources.clone()
        out[0, 0, pos_indices, :] = 1.0

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
        exp_callback=one_out_positions,
    )

    return (sum_loss / n_ex).item()


# %%


def grid_of_line_plots(lines: list[Tensor], width: int = 10, height: int = 10):
    _, axes = plt.subplots(width, height, sharex=True, sharey=True, figsize=(20, 20))
    for i, line in enumerate(lines):
        row, col = i // width, i % width
        axes[row, col].plot(line.detach().cpu().numpy())
        # add mean annotation in top left
        axes[row, col].text(0.02, 0.98, f"{line.mean().item():.2f}", ha="left", va="top", fontsize=8, transform=axes[row, col].transAxes)
    plt.show()


from itertools import islice
def vis_loss_dist_single(setup: ModelSetup):
    def generator():
        for batch in data_loader_iter:
            batch = extract_batch_data(batch).to(device)
            assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"
            for x in get_losses_by_pos(setup, batch):
                yield x
                # if x.mean().item() > 0.03:
                #     yield x
    grid_of_line_plots(list(tqdm(islice(generator(), 100))))

vis_loss_dist_single(model_setups[OLD])

# %%


def vis_loss_dist(nb: int):
    new_means: list[tuple[float, int]] = []
    old_means: list[tuple[float, int]] = []
    for i in tqdm(range(nb)):
        set_seed(i)
        batch = extract_batch_data(next(data_loader_iter)).to(device)
        assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"
        for pos_idx, losses in enumerate(get_losses_by_pos(model_setups[NEW], batch)):
            mean_loss = losses.mean(dim=0).item()
            new_means.append((mean_loss, int(batch[0, pos_idx].item())))
        for pos_idx, losses in enumerate(get_losses_by_pos(model_setups[OLD], batch)):
            mean_loss = losses.mean(dim=0).item()
            old_means.append((mean_loss, int(batch[0, pos_idx].item())))
    return new_means, old_means


n, o = vis_loss_dist(20)
# %%

bins = np.linspace(0, 0.15, 100).tolist()

fst = lambda x: x[0]
plt.hist(list(map(fst, n)), alpha=0.3, bins=bins)
plt.hist(list(map(fst, o)), alpha=0.3, bins=bins)
plt.yscale("log")
plt.legend(["New", "Old"])
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.title("Loss Distribution")
plt.show()


# %%

o_t = torch.tensor(list(map(fst, o)), device=device)
worst_100_idcs = o_t.argsort().tolist()[::-1][:100]
for idx in worst_100_idcs:
    print(idx)
    loss, token_idx = o[idx]
    print(f"loss: {loss}, token: {tokenizer.decode(token_idx)}")

# %%
o_top_100
# %%

loss_means = []

bef = []
aft = []

for _ in tqdm(range(30)):
    batch = extract_batch_data(next(data_loader_iter)).to(device)
    cat_pos_indices = []
    for pos_idx, losses in enumerate(get_losses_by_pos(model_setups[OLD], batch)):
        mean_loss = losses.mean(dim=0).item()
        if mean_loss > 0.3:
            cat_pos_indices.append(pos_idx)

    # mean_loss_before
    bef.append(pgd_without_positions(model_setups[OLD], batch, []))
    aft.append(pgd_without_positions(model_setups[OLD], batch, cat_pos_indices))

# %%

plt.scatter(bef, aft)
plt.xlabel("Before")
plt.ylabel("After")
plt.title("Before and After PGD Loss")
plt.show()

# %%

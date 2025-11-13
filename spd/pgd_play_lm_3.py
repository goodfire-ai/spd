# ruff: noqa: E402, I001
# %%

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
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data
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
    )[2][MLP_UP_0]

    assert batch.shape[0] == 1, "batch must be of shape (1, seq_len)"

    for seq_idx in range(batch.shape[1]):
        routing_mask = torch.ones_like(batch, dtype=torch.bool)
        routing_mask[:, seq_idx] = False

        mask_w_pos_replaced = ComponentsMaskInfo(
            component_mask=full_pgd_mask_info.component_mask,
            routing_mask="all",
            weight_delta_and_mask=None,  # full_pgd_mask_info.weight_delta_and_mask,
        )

        patched_output = setup.model(batch, mask_infos={MLP_UP_0: mask_w_pos_replaced})
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

def vis_loss_dist():
loss_means = []
for _ in tqdm(range(30)):
    batch = extract_batch_data(next(data_loader_iter)).to(device)
    for losses in get_losses_by_pos(model_setups[OLD], batch):
        mean_loss = losses.mean(dim=0).item()
        loss_means.append(mean_loss)
plt.hist(loss_means, log=True)
plt.show()


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

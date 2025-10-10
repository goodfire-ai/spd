# %%
import json

import matplotlib.pyplot as plt
from pathlib import Path

import fire
from torch import Tensor, nn
import torch.nn.functional as F
import wandb
from simple_stories_train.run_info import RunInfo as SSRunInfo

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.run_spd import optimize
from spd.settings import REPO_ROOT
from spd.utils.distributed_utils import (
    call_on_rank0_then_broadcast,
    ensure_cached_and_call,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import (
    calc_kl_divergence_lm,
    load_config,
    resolve_class,
    save_pre_run_info,
    set_seed,
)
from spd.utils.run_utils import get_output_dir
from spd.utils.wandb_utils import init_wandb

# %%

cfg = load_config(REPO_ROOT / "spd/experiments/lm/ss_llama_config.yaml", Config)

# %%

target_model: nn.Module = resolve_class(cfg.pretrained_model_class).from_pretrained(cfg.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
# %%
device = get_device()
target_model.to(device)


# %%

BATCH_SIZE = 8

assert isinstance(cfg.task_config, LMTaskConfig)
train_data_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=cfg.tokenizer_name,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    is_tokenized=cfg.task_config.is_tokenized,
    streaming=cfg.task_config.streaming,
    column_name=cfg.task_config.column_name,
    shuffle_each_epoch=cfg.task_config.shuffle_each_epoch,
    seed=None,
)

train_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=BATCH_SIZE,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)


# %%

dl = iter(train_loader)

# %%

target_model.requires_grad_(False)

model = ComponentModel(
    target_model=target_model,
    target_module_patterns=["model.embed_tokens"],
    C=8000,
    ci_fn_type="mlp",
    ci_fn_hidden_dims=[16],
    pretrained_model_output_attr=cfg.pretrained_model_output_attr,
)
# %%

model.to(device)

ex: Tensor = next(dl)["input_ids"].to(device)

target_out_logits, module_cache = model(ex, cache_type="input")

# TODO maybe using the wrong leaky one
ci, _ = model.calc_causal_importances(module_cache, sigmoid_type=cfg.sigmoid_type, detach_inputs=False, sampling=cfg.sampling)
assert len(ci) == 1

mask_infos = make_mask_infos(ci)

out_logits = model(ex, mask_infos=mask_infos)

kl = calc_kl_divergence_lm(pred=out_logits, target=target_out_logits)

# def bwd_hook(grad):

# %%
kl.backward()
# %%

# %%
U_grad = model.components["model.embed_tokens"].U.grad
V_grad = model.components["model.embed_tokens"].V.grad
assert U_grad is not None and V_grad is not None
# %%
import torch
print(V_grad.shape) # (C, d_model)
# %%
import gc
gc.collect()
torch.cuda.empty_cache()

a = V_grad.unsqueeze(1) # (C, 1, d_model)
b = V_grad.unsqueeze(0) # (1, C, d_model)

pw_cs = torch.zeros(V_grad.shape[0], V_grad.shape[0], device=V_grad.device)
for chunk in range(0, V_grad.shape[0], 200):
    for chunk2 in range(0, V_grad.shape[0], 200):
        a = V_grad[chunk:chunk+200].unsqueeze(1)
        b = V_grad[chunk2:chunk2+200].unsqueeze(0)
        cosine_similarity = F.cosine_similarity(a, b, dim=-1)
        pw_cs[chunk:chunk+200, chunk2:chunk2+200] = cosine_similarity
# %%

naive_max_pwcs = pw_cs.amax(dim=0)
print(naive_max_pwcs.shape)
idcs, = torch.where(naive_max_pwcs > 0.9)
plt.hist(naive_max_pwcs.cpu().detach().numpy())
# %%

sanitized_pw_cs = pw_cs.clone()
sanitized_pw_cs.fill_diagonal_(0)
max_pwcs = sanitized_pw_cs.amax(dim=0)
plt.hist(max_pwcs[ex.unique()].cpu().detach().numpy())
# %%

# idcs, = torch.where( > 0.9)
# idcs

# max_pwcs = pw_cs.amax(dim=0)
# pwcs[:6, :6]
# # max_pwcs = pwcs.amax(dim=0)
# # plt.hist(max_pwcs.cpu().detach().numpy())

# %%

max_pwcs = pwcs.amax(dim=0)

ind, = torch.where(max_pwcs > 0.9)

# %%

m = max_pwcs.cpu().detach().numpy()
print(m.shape)
plt.hist(m)

# %%
def kl_divergence(p, q):
# %%


@with_distributed_cleanup
def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    dist_state = init_distributed()

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    # Use the same seed across all ranks for deterministic data loading
    set_seed(config.seed)

    if is_main_process():
        if config.wandb_project:
            tags = ["lm"]
            if evals_id:
                tags.append(evals_id)
            if sweep_id:
                tags.append(sweep_id)
            config = init_wandb(config, config.wandb_project, tags=tags)
            assert wandb.run
            if config.wandb_run_name:
                wandb.run.name = config.wandb_run_name

        if config.out_dir is not None:
            out_dir = config.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = get_output_dir(use_wandb_id=config.wandb_project is not None)
        logger.info(f"Output directory: {out_dir}")
        logger.info(config)
        if dist_state.world_size > 1:
            logger.info(f"Running distributed training with {dist_state.world_size} processes")
    else:
        out_dir = None

    device = get_device()
    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None

    ln_stds: dict[str, float] | None = None
    if config.pretrained_model_class.startswith("simple_stories_train"):
        # Handle differently in case run has layernorm ablations (we'd need to collect ln_stds)
        # Avoid concurrent wandb API requests on each rank
        run_info = call_on_rank0_then_broadcast(SSRunInfo.from_path, config.pretrained_model_name)
        if run_info.config_dict["enable_ln_ablation"]:
            ln_stds = run_info.ln_stds
            assert ln_stds is not None, "Run had enable_ln_ablation set to True but no ln_stds"
        assert hasattr(pretrained_model_class, "from_run_info")
        # Just loads from local file
        target_model = pretrained_model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        # Avoid concurrent wandb API requests by first calling from_pretrained on rank 0 only
        target_model = ensure_cached_and_call(
            pretrained_model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            config.pretrained_model_name,
        )
    target_model.eval()

    if is_main_process():
        assert out_dir is not None
        save_pre_run_info(
            save_to_wandb=config.wandb_project is not None,
            out_dir=out_dir,
            spd_config=config,
            sweep_params=sweep_params,
            target_model=None,
            train_config=None,
            task_name=None,
        )

    # --- Load Data --- #
    if is_main_process():
        logger.info("Loading dataset...")
    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    # Keep per-process batch size constant to maintain scale of all metrics so we can simply average
    # them across processes.
    assert config.microbatch_size % dist_state.world_size == 0 and config.microbatch_size > 0, (
        f"Microbatch size {config.microbatch_size} is not divisible by world size {dist_state.world_size}. "
    )
    train_rank_microbatch_size = config.microbatch_size // dist_state.world_size

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_rank_microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=dist_state.rank,
        ddp_world_size=dist_state.world_size,
    )

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    assert config.eval_batch_size % dist_state.world_size == 0 and config.eval_batch_size > 0, (
        f"Eval batch size {config.eval_batch_size} is not divisible by world size {dist_state.world_size}. "
    )
    eval_rank_batch_size = config.eval_batch_size // dist_state.world_size

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
        ddp_rank=dist_state.rank,
        ddp_world_size=dist_state.world_size,
    )

    if is_main_process():
        logger.info("Starting optimization...")

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        ln_stds=ln_stds,
    )

    if is_main_process():
        logger.info("Optimization finished.")
        if config.wandb_project:
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

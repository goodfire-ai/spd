"""Language Model decomposition script."""

import json
import os
from pathlib import Path

import fire
import wandb
from simple_stories_train.run_info import RunInfo as SSRunInfo

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.distributed_utils import (
    DistributedState,
    call_on_rank0_then_broadcast,
    ensure_cached_and_call,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import resolve_class, save_pre_run_info, set_seed
from spd.utils.run_utils import get_output_dir
from spd.utils.wandb_utils import init_wandb


@with_distributed_cleanup
def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )
    if config_path is not None:
        config = Config.from_file(config_path)
    else:
        assert config_json is not None
        config = Config(**json.loads(config_json.removeprefix("json:")))

    dist_state = init_distributed(backend="nccl")
    logger.info(f"Distributed state: {dist_state}")

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
            slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", None)
            if slurm_array_job_id is not None:
                logger.info(f"Running on slurm array job id: {slurm_array_job_id}")
                tags.append(f"slurm-array-job-id_{slurm_array_job_id}")

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
        if dist_state is not None:
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

    match dist_state:
        case DistributedState(world_size=world_size):
            # Keep per-process batch size constant to maintain scale of all metrics so we can simply average
            # them across processes.
            assert config.microbatch_size % world_size == 0 and config.microbatch_size > 0, (
                f"Microbatch size {config.microbatch_size} is not divisible by world size {world_size}. "
            )
            train_rank_microbatch_size = config.microbatch_size // world_size
        case None:
            train_rank_microbatch_size = config.microbatch_size

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_rank_microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        dist_state=dist_state,
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

    match dist_state:
        case DistributedState(world_size=world_size):
            assert config.eval_batch_size % world_size == 0 and config.eval_batch_size > 0, (
                f"Eval batch size {config.eval_batch_size} is not divisible by world size {world_size}. "
            )
            eval_rank_batch_size = config.eval_batch_size // world_size
        case None:
            eval_rank_batch_size = config.eval_batch_size

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
        dist_state=dist_state,
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

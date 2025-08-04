"""Language Model decomposition script."""

import json
from pathlib import Path

import fire
import wandb
from transformers import PreTrainedModel

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_device,
    init_distributed,
    is_main_process,
)
from spd.utils.general_utils import (
    load_config,
    resolve_class,
    save_pre_run_info,
    set_seed,
)
from spd.utils.run_utils import get_output_dir
from spd.utils.wandb_utils import init_wandb


def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    # Load config first to get DDP settings
    config = load_config(config_path_or_obj, config_model=Config)

    # Initialize distributed training with backend from config if specified
    backend = config.ddp_backend if config.ddp_enabled else None
    rank, world_size, _local_rank = init_distributed(backend=backend)

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    # Only initialize wandb on main process
    if config.wandb_project and is_main_process():
        tags = ["lm"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    if is_main_process():
        out_dir = get_output_dir(use_wandb_id=config.wandb_project is not None)
        logger.info(f"Output directory: {out_dir}")
    else:
        out_dir = None

    # Adjust seed per rank to ensure different data sampling
    set_seed(config.seed + rank)
    if is_main_process():
        logger.info(config)
        if world_size > 1:
            logger.info(f"Running distributed training with {world_size} processes")

    device = get_device()
    if is_main_process():
        logger.info(f"Rank {rank} using device: {device}")
    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for LM decomposition."
    )

    # --- Load Model --- #
    if is_main_process():
        logger.info("Loading base language model ...")

    hf_model_class = resolve_class(config.pretrained_model_class)
    assert issubclass(hf_model_class, PreTrainedModel), (
        f"Model class {hf_model_class} should be a subclass of PreTrainedModel which "
        "defines a `from_pretrained` method"
    )
    assert config.pretrained_model_name_hf is not None
    target_model = hf_model_class.from_pretrained(config.pretrained_model_name_hf)
    target_model.eval()

    if config.wandb_project and is_main_process():
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    # Only save pre-run info on main process
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
        hf_tokenizer_path=config.pretrained_model_name_hf,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=config.task_config.column_name,
    )

    # Adjust batch size for distributed training
    # Keep per-process batch size constant to maintain gradient scale
    train_batch_size = config.microbatch_size // world_size
    if train_batch_size == 0:
        raise ValueError(
            f"Microbatch size {config.microbatch_size} is smaller than world size {world_size}. "
            "Please increase the microbatch size or decrease the world size."
        )

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=rank,  # Use actual rank
        ddp_world_size=world_size,  # Use actual world_size
    )

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name_hf,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=config.task_config.column_name,
    )

    # For evaluation, we can use full batch size on rank 0 only
    # TODO: Update to split eval across all ranks
    eval_batch_size = config.eval_batch_size if is_main_process() else 1

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=rank,  # Use actual rank
        ddp_world_size=world_size,  # Use actual world_size
    )

    if is_main_process():
        logger.info("Dataset and tokenizer loaded.")

    # TODO: Below not needed when TMS supports config.n_eval_steps
    assert config.n_eval_steps is not None, "n_eval_steps must be set"
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
    )

    if is_main_process():
        logger.info("Optimization finished.")

    if config.wandb_project and is_main_process():
        wandb.finish()

    # Clean up distributed process group
    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(main)

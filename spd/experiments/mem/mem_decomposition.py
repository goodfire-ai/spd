"""Mem model decomposition script using SPD."""

import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config, MemTaskConfig
from spd.experiments.mem.mem_dataset import MemDataset
from spd.experiments.mem.models import MemTargetRunInfo, MemTransformer
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    """Run SPD decomposition on a trained MemTransformer model.

    Args:
        config_path: Path to config YAML file
        config_json: JSON string containing config (alternative to config_path)
        evals_id: Optional evaluation ID for tagging
        sweep_id: Optional sweep ID for tagging
        sweep_params_json: Optional JSON string with sweep parameters
    """
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )
    if config_path is not None:
        config = Config.from_file(config_path)
    else:
        assert config_json is not None
        config = Config(**json.loads(config_json.removeprefix("json:")))

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    out_dir, run_id, tags = setup_decomposition_run(
        experiment_tag="mem", evals_id=evals_id, sweep_id=sweep_id
    )

    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=config.wandb_run_name,
            tags=tags,
        )

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    task_config = config.task_config
    assert isinstance(task_config, MemTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = MemTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = MemTransformer.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_run_info.config,
        task_name=config.task_config.task_name,
    )

    # Create dataset with same parameters as training
    dataset = MemDataset(
        n_facts=target_run_info.n_facts,
        vocab_size=target_model.config.vocab_size,
        seq_len=target_model.config.seq_len,
        device=device,
        seed=target_run_info.config.seed,  # Use same seed for reproducibility
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    assert config.n_eval_steps is not None, "n_eval_steps must be set"
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

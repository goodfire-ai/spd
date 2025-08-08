import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config
from spd.experiments.ih.configs import IHTaskConfig
from spd.experiments.ih.model import InductionModelTargetRunInfo, InductionTransformer
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    load_config,
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
    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["ih"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    # Get output directory (automatically uses wandb run ID if available)
    out_dir = get_output_dir(use_wandb_id=config.wandb_project is not None)

    task_config = config.task_config
    assert isinstance(task_config, IHTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = InductionModelTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = InductionTransformer.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
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

    prefix_window = task_config.prefix_window or target_model.config.seq_len - 3

    dataset = InductionDataset(
        vocab_size=target_model.config.vocab_size,
        seq_len=target_model.config.seq_len,
        prefix_window=prefix_window,
        device=device,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

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

import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config, IHTaskConfig
from spd.experiments.ih.model import InductionModelTargetRunInfo, InductionTransformer
from spd.log import logger
from spd.models.batch_and_loss_fns import recon_loss_kl
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    save_pre_run_info,
    set_seed,
)
from spd.utils.run_utils import ExecutionStamp
from spd.utils.wandb_utils import init_wandb


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

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    execution_stamp = ExecutionStamp.create(run_type="spd", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = ["ih"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=execution_stamp.run_id,
            name=config.wandb_run_name,
            tags=tags,
        )

    task_config = config.task_config
    assert isinstance(task_config, IHTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = InductionModelTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = InductionTransformer.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

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
    extract_input = lambda batch: batch[0]
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, transform=extract_input
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, transform=extract_input
    )

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        reconstruction_loss=recon_loss_kl,
        out_dir=out_dir,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import json
from pathlib import Path

import fire
import wandb

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.log import logger
from spd.models.batch_and_loss_fns import recon_loss_mse
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
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

    set_seed(config.seed)

    out_dir, run_id, tags = setup_decomposition_run(
        experiment_tag="tms", evals_id=evals_id, sweep_id=sweep_id
    )
    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=config.wandb_run_name,
            tags=tags,
        )
    logger.info(config)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = TMSTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = TMSModel.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_model.config,
        task_name=config.task_config.task_name,
    )

    synced_inputs = target_run_info.config.synced_inputs
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        reconstruction_loss=recon_loss_mse,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

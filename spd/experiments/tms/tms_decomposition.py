from __future__ import annotations

"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
import yaml

from spd.configs import Config, TMSTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.experiments.tms.models import TMSModel, TMSModelConfig
from spd.log import logger
from spd.plotting import create_toy_model_plot_results
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, tms_model_config: TMSModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{tms_model_config.n_features}_"
        run_suffix += f"hid{tms_model_config.n_hidden}"
        run_suffix += f"hid-layers{tms_model_config.n_hidden_layers}"
    return config.wandb_run_name_prefix + run_suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    tms_model: TMSModel,
    tms_model_train_config_dict: dict[str, Any],
) -> None:
    torch.save(tms_model.state_dict(), out_dir / "tms.pth")

    with open(out_dir / "tms_train_config.yaml", "w") as f:
        yaml.dump(tms_model_train_config_dict, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "tms.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "tms_train_config.yaml"), base_path=out_dir, policy="now")


def main(config_path_or_obj: Path | str | Config) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        config.pretrained_model_path,
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, tms_model_config=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        tms_model=target_model,
        tms_model_train_config_dict=target_model_train_config_dict,
    )

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        plot_results_fn=create_toy_model_plot_results,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

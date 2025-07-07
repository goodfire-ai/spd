from __future__ import annotations

"""Residual Linear decomposition script."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, ResidualMLPTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.resid_mlp.models import ResidualMLP
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.log import logger
from spd.plotting import create_toy_model_plot_results
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    n_features: int,
    n_layers: int,
    d_resid: int,
    d_mlp: int,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
    return config.wandb_run_name_prefix + run_suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    resid_mlp: ResidualMLP,
    resid_mlp_train_config_dict: dict[str, Any],
    label_coeffs: Float[Tensor, " n_features"],
) -> None:
    torch.save(resid_mlp.state_dict(), out_dir / "resid_mlp.pth")

    with open(out_dir / "resid_mlp_train_config.yaml", "w") as f:
        yaml.dump(resid_mlp_train_config_dict, f, indent=2)

    with open(out_dir / "label_coeffs.json", "w") as f:
        json.dump(label_coeffs.detach().cpu().tolist(), f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "resid_mlp.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "resid_mlp_train_config.yaml"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")


def main(config_path_or_obj: Path | str | Config) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualMLPTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict, label_coeffs = ResidualMLP.from_pretrained(
        config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(
        config,
        n_features=target_model.config.n_features,
        n_layers=target_model.config.n_layers,
        d_resid=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        resid_mlp=target_model,
        resid_mlp_train_config_dict=target_model_train_config_dict,
        label_coeffs=label_coeffs,
    )

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = ResidualMLPDataset(
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=synced_inputs,
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
        plot_results_fn=create_toy_model_plot_results,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

"""Run SPD on a memorization model."""

from pathlib import Path
from typing import Any

import fire
import wandb
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, MemorizationTaskConfig
from spd.experiments.memorization.memorization_dataset import KeyValueMemorizationDataset
from spd.experiments.memorization.models import SingleLayerMemorizationMLP
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    memorization_model: SingleLayerMemorizationMLP,
    memorization_train_config_dict: dict[str, Any],
    key_value_pairs: tuple[Float[Tensor, "n_pairs d_model"], Float[Tensor, "n_pairs d_model"]],
) -> None:
    save_file(memorization_model.state_dict(), out_dir / "memorization.pth")
    save_file(memorization_train_config_dict, out_dir / "memorization_train_config.yaml")

    # Save key-value pairs as JSON
    keys, values = key_value_pairs
    kv_data = {
        "keys": keys.cpu().tolist(),
        "values": values.cpu().tolist(),
    }
    save_file(kv_data, out_dir / "key_value_pairs.json")

    if save_to_wandb:
        wandb.save(str(out_dir / "memorization.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "memorization_train_config.yaml"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "key_value_pairs.json"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["memorization"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    # Get output directory (automatically uses wandb run ID if available)
    out_dir = get_output_dir()

    task_config = config.task_config
    assert isinstance(task_config, MemorizationTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict, key_value_pairs = (
        SingleLayerMemorizationMLP.from_pretrained(
            config.pretrained_model_path,
        )
    )
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    save_file(config.model_dump(mode="json"), out_dir / "final_config.yaml")
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        memorization_model=target_model,
        memorization_train_config_dict=target_model_train_config_dict,
        key_value_pairs=key_value_pairs,
    )

    # Create dataset with the same key-value pairs used for training
    keys, values = key_value_pairs
    dataset = KeyValueMemorizationDataset(
        n_pairs=target_model.config.n_pairs,
        d_model=target_model.config.d_model,
        device=device,
        key_value_pairs=(keys.to(device), values.to(device)),
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

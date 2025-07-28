import json
from pathlib import Path
from typing import Any

import fire
import wandb

from spd.configs import Config, IHTaskConfig
from spd.experiments.ih.model import InductionTransformer
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    induction_model: InductionTransformer,
    induction_model_train_config_dict: dict[str, Any],
) -> None:
    save_file(induction_model.state_dict(), out_dir / "ih.pth")
    save_file(induction_model_train_config_dict, out_dir / "ih_train_config.yaml")

    if save_to_wandb:
        wandb.save(str(out_dir / "ih.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "ih_train_config.yaml"), base_path=out_dir, policy="now")


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
    out_dir = get_output_dir()

    task_config = config.task_config
    assert isinstance(task_config, IHTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict = InductionTransformer.from_pretrained(
        config.pretrained_model_path,
    )
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = config.wandb_run_name

    save_file(config.model_dump(mode="json"), out_dir / "final_config.yaml")
    if sweep_params:
        save_file(sweep_params, out_dir / "sweep_params.yaml")
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")
        if sweep_params:
            wandb.save(str(out_dir / "sweep_params.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        induction_model=target_model,
        induction_model_train_config_dict=target_model_train_config_dict,
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

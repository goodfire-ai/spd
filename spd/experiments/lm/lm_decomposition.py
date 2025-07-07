from __future__ import annotations

"""Language Model decomposition script."""

from datetime import datetime
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.plotting import plot_mean_component_activation_counts
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, load_pretrained, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    pretrained_model_name: str | None,
    max_seq_len: int,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        if pretrained_model_name:
            run_suffix += f"_pretrained{pretrained_model_name}"
        run_suffix += f"_seq{max_seq_len}"
    return config.wandb_run_name_prefix + run_suffix


def plot_lm_results(
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]],
) -> plt.Figure:
    """Plotting function for LM decomposition."""

    return plot_mean_component_activation_counts(
        mean_component_activation_counts=mean_component_activation_counts,
    )


def main(config_path_or_obj: Path | str | Config) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for LM decomposition."
    )

    # --- Load Model --- #
    logger.info("Loading base language model ...")

    target_model = load_pretrained(
        path_to_class=config.pretrained_model_class,
        model_path=None,
        model_name_hf=config.pretrained_model_name_hf,
    )

    # --- Setup Run Name and Output Dir --- #
    run_name = get_run_name(
        config,
        pretrained_model_name=config.pretrained_model_name_hf,
        max_seq_len=config.task_config.max_seq_len,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Save Config --- #
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    # --- Load Data --- #
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

    train_loader, tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
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
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    logger.info("Dataset and tokenizer loaded.")

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

    logger.info("Optimization finished.")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

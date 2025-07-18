"""
Vizualises the components of the model.
"""

import torch

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.plotting import plot_mean_component_activation_counts
from spd.spd_types import ModelPath
from spd.utils.component_utils import component_activation_statistics


def main(path: ModelPath) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ss_model, config, checkpoint_path = ComponentModel.from_pretrained(path)
    ss_model.to(device)

    out_dir = checkpoint_path

    assert isinstance(config.task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name_hf,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=config.task_config.column_name,
    )

    dataloader, _tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )
    # logger.info(ss_model)
    logger.values(config.model_dump(), msg="Model config")

    mean_n_active_components_per_token, mean_component_activation_counts = (
        component_activation_statistics(
            model=ss_model,
            dataloader=dataloader,
            n_steps=100,
            device=device,
        )
    )
    logger.values(
        {
            "n_components": str(ss_model.C),
            "mean_n_active_components_per_token": str(mean_n_active_components_per_token),
            "mean_component_activation_counts": str(mean_component_activation_counts),
        }
    )
    fig = plot_mean_component_activation_counts(
        mean_component_activation_counts=mean_component_activation_counts,
    )
    # Save the entire figure once
    save_path = out_dir / "modules_mean_component_activation_counts.png"
    fig.savefig(save_path)
    logger.info(f"Saved combined plot to {save_path}")


if __name__ == "__main__":
    path = "wandb:spd-gf-lm/runs/151bsctx"
    main(path)

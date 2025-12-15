"""CLI for autointerp pipeline.

Usage:
    # Harvest (GPU)
    python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> [options]

    # Interpret (CPU)
    python -m spd.autointerp.scripts.run_autointerp interpret <run_id> [options]
"""

import os

import fire
from transformers import AutoTokenizer

from spd.autointerp.harvest import HarvestConfig, harvest, save_harvest
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import ArchitectureInfo
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.wandb_utils import parse_wandb_run_path


def harvest_cmd(
    wandb_path: str,
    n_batches: int,
    batch_size: int = 32,
    context_length: int = 256,
    ci_threshold: float = 0.01,
    activation_examples_per_component: int = 20,
    activation_context_tokens_per_side: int = 10,
) -> None:
    """Harvest correlations and activation contexts."""
    device = get_device()
    print(f"Device: {device}")

    entity, project, run_id = parse_wandb_run_path(wandb_path)
    clean_path = f"{entity}/{project}/{run_id}"
    print(f"Loading: {clean_path}")

    run_info = SPDRunInfo.from_path(clean_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    assert spd_config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)

    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)

    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=context_length,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
    )

    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed,
    )

    harvest_config = HarvestConfig(
        wandb_path=clean_path,
        n_batches=n_batches,
        batch_size=batch_size,
        context_length=context_length,
        ci_threshold=ci_threshold,
        activation_examples_per_component=activation_examples_per_component,
        activation_context_tokens_per_side=activation_context_tokens_per_side,
    )

    result = harvest(harvest_config, model, tokenizer, train_loader, spd_config)  # pyright: ignore[reportArgumentType]

    out_dir = save_harvest(result, run_id)
    print(f"Saved {len(result.components)} components to {out_dir}")


def interpret_cmd(
    run_id: str,
    model_name: str,
    dataset_name: str,
    dataset_description: str,
    n_layers: int,
    c_per_layer: int,
    model: str = "claude-haiku-4-5-20251001",
    max_concurrent: int = 50,
) -> None:
    """Interpret harvested components."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY not set"

    arch = ArchitectureInfo(
        n_layers=n_layers,
        c_per_layer=c_per_layer,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_description=dataset_description,
    )

    run_interpret(run_id, arch, api_key, model, max_concurrent)


if __name__ == "__main__":
    fire.Fire({"harvest": harvest_cmd, "interpret": interpret_cmd})

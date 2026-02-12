"""Standalone eval script to run PGDReconLoss on an existing SPD run.

Uses PGDMultiBatch with gradient accumulation to fit batch_size=128 in memory.
"""

import torch

from spd.configs import LMTaskConfig, PGDMultiBatchReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.metrics.pgd_utils import calc_multibatch_pgd_masked_recon_loss
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.routing import AllLayersRouter
from spd.utils.general_utils import bf16_autocast


def main() -> None:
    wandb_path = "wandb:goodfire/spd/runs/s-275c8f21"

    logger.info(f"Loading SPD run from {wandb_path}")
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    device = "cuda"
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()

    assert isinstance(config.task_config, LMTaskConfig)

    microbatch_size = 16
    gradient_accumulation_steps = 8  # 16 * 8 = 128 effective batch size

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
    )

    # Deterministic iterator: recreated each PGD step so each step sees the same data
    def create_data_iter():
        eval_loader.generator.manual_seed(config.seed + 1)
        return iter(eval_loader)

    pgd_config = PGDMultiBatchReconLossConfig(
        coeff=None,
        init="random",
        step_size=0.05,
        n_steps=500,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    batch_dims = (microbatch_size, config.task_config.max_seq_len)

    logger.info(
        f"Running PGDMultiBatchReconLoss: n_steps={pgd_config.n_steps}, "
        f"step_size={pgd_config.step_size}, "
        f"microbatch_size={microbatch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"effective_batch_size={microbatch_size * gradient_accumulation_steps}"
    )

    with bf16_autocast(enabled=config.autocast_bf16):
        weight_deltas = model.calc_weight_deltas() if config.use_delta_component else None

        result = calc_multibatch_pgd_masked_recon_loss(
            pgd_config=pgd_config,
            model=model,
            weight_deltas=weight_deltas,
            create_data_iter=create_data_iter,
            output_loss_type=config.output_loss_type,
            router=AllLayersRouter(),
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            batch_dims=batch_dims,
            device=device,
        )

    logger.info(f"PGDMultiBatchReconLoss result: {result.item():.8f}")


if __name__ == "__main__":
    main()

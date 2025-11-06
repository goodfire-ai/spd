import time

import wandb

from spd.configs import PGDGlobalReconLossConfig, PGDInitStrategy
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.metrics.pgd_utils import calc_pgd_global_masked_recon_loss
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed


def main(seed: int, name: str, tags: list[str]):
    logger.info(f"Running PGD saturation test with seed {seed}, name {name}, and tags {tags}")
    device = get_device()

    logger.info("Loading run info")
    run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/pwghbtr2")
    logger.info("Loading config")
    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    init: PGDInitStrategy = "random"
    step_size = 0.1
    n_steps = 20

    bs_grad_accum_steps_n_ctx_list = [
        # (1, 1, 1),  # 1
        # (1, 1, 2),  # 2
        # (1, 1, 4),  # 4
        (1, 1, 8),  # 8
        (1, 1, 16),  # 16
        (1, 1, 32),  # 32
        (1, 1, 64),  # 64
        (1, 1, 128),  # 128
        (1, 1, 256),  # 256
        (1, 1, 521),  # 521
        (2, 1, 512),  # 1024
        (4, 1, 512),  # 2048
        (8, 1, 512),  # 4096
        (16, 1, 512),  # 8192
        (32, 1, 512),  # 16384
        (64, 1, 512),  # 32768
        (128, 1, 512),  # 65536
        # (128, 2, 512), # 131072
        # (128, 4, 512), # 262144
        # (128, 8, 512), # 524288
        # (128, 16, 512), # 1048576
        # (128, 32, 512), # 2097152
        # (128, 64, 512), # 4194304
        # (128, 128, 512), # 8388608
    ]

    logger.info("Initializing WandB run")
    # run = wandb.init(project="pgd-saturation", name=f"varying_seed-base_seed_{seed}")
    run = wandb.init(project="pgd-saturation", name=name, tags=tags)

    for _, (batch_size, grad_accum_steps, n_ctx) in enumerate(bs_grad_accum_steps_n_ctx_list):
        logger.info(f"{batch_size=}, {grad_accum_steps=}, {n_ctx=}")
        logger.info("Loading model")
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.target_model.requires_grad_(False)

        logger.info("Loading train data config")
        train_data_config = DatasetConfig(
            name=config.task_config.dataset_name,
            hf_tokenizer_path=config.tokenizer_name,
            split=config.task_config.train_data_split,
            n_ctx=n_ctx,
            is_tokenized=config.task_config.is_tokenized,
            streaming=config.task_config.streaming,
            column_name=config.task_config.column_name,
            shuffle_each_epoch=config.task_config.shuffle_each_epoch,
            seed=None,
        )

        logger.info("Setting seed, loading data loader")
        set_seed(seed)
        data_loader, _ = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=config.task_config.buffer_size,
            global_seed=config.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

        pgd_global_config = PGDGlobalReconLossConfig(
            init=init,
            step_size=step_size,
            n_steps=n_steps,
            gradient_accumulation_steps=grad_accum_steps,
        )

        logger.info("Calculating PGD loss")
        start_time = time.time()
        loss = calc_pgd_global_masked_recon_loss(
            pgd_config=pgd_global_config,
            model=model,
            dataloader=data_loader,
            output_loss_type=config.output_loss_type,
            routing="all",
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            batch_dims=(batch_size, n_ctx),
            device=device,
        )
        taken = time.time() - start_time

        run.log(
            {
                "pgd_init_seed": seed,  #  * 1000 + i,
                "pgd_loss": loss,
                "micro_batch_size": batch_size,
                "effective_batch_size": batch_size * grad_accum_steps,
                "grad_accum_steps": grad_accum_steps,
                "time_taken": taken,
            }
        )
    run.finish()


def slurm_entrypoint():
    import os

    array_job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
    array_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    seed = array_idx
    name = f"seed_{seed}"
    tags = ["fixed-seed", "varying_ctx", f"sweep_{array_job_id}"]
    main(seed=seed, name=name, tags=tags)


if __name__ == "__main__":
    slurm_entrypoint()

# if __name__ == "__main__":
#     main(
#         seed=0,
#         name="seed_0",
#         tags=["fixed-seed", "varying_ctx", "sweep_0"],
#     )

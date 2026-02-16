"""PGDReconLoss eval inlining the EXACT training eval logic from the snapshot.

Replicates: create one PGDReconLoss, call update() N_EVAL_STEPS times, compute() once.
No bf16 autocast (didn't exist in snapshot).
"""

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed.distributed_c10d as c10d

from spd.configs import LMTaskConfig, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.log import logger
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.run_info import PretrainRunInfo
from spd.utils.distributed_utils import (
    DistributedState,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import extract_batch_data, set_seed

# ── Eval knobs ──────────────────────────────────────────────────────
RUN_ID = "s-892f140b"
N_EVAL_STEPS = 5
N_STEPS_LIST = [5, 20]
STEP_SIZE = 0.1
# ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd")


def pre_cache_on_rank0() -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    logger.info("Pre-caching model checkpoint on LOCAL_RANK=0...")
    run_info = SPDRunInfo.from_path(f"wandb:goodfire/spd/{RUN_ID}")
    assert run_info.config.pretrained_model_name is not None
    PretrainRunInfo.from_path(run_info.config.pretrained_model_name)
    logger.info("Model cached.")


@with_distributed_cleanup
def main() -> None:
    pre_cache_on_rank0()

    c10d.default_pg_nccl_timeout = timedelta(minutes=30)
    dist_state = init_distributed()
    device = get_device()

    run_info = SPDRunInfo.from_path(f"wandb:goodfire/spd/{RUN_ID}")
    config = run_info.config

    set_seed(config.seed)

    assert isinstance(config.task_config, LMTaskConfig)
    task_config = config.task_config

    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
        seed=None,
    )

    match dist_state:
        case DistributedState(world_size=world_size):
            assert config.eval_batch_size % world_size == 0
            eval_rank_batch_size = config.eval_batch_size // world_size
        case None:
            eval_rank_batch_size = config.eval_batch_size

    if is_main_process():
        logger.info(f"Run: {RUN_ID} (inlined training eval path)")
        logger.info(f"PGD n_steps: {N_STEPS_LIST}, step_size: {STEP_SIZE}")
        logger.info(f"N_EVAL_STEPS: {N_EVAL_STEPS}")
        logger.info(f"Eval batch size: {config.eval_batch_size} (per-rank: {eval_rank_batch_size})")

    component_model = ComponentModel.from_run_info(run_info)
    component_model.to(device)
    component_model.eval()

    for n_steps in N_STEPS_LIST:
        eval_loader, _ = create_data_loader(
            dataset_config=eval_data_config,
            batch_size=eval_rank_batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=config.seed + 1,
            dist_state=dist_state,
        )
        eval_iterator = loop_dataloader(eval_loader)

        # Exactly like training: one metric, N_EVAL_STEPS updates, one compute
        metric = PGDReconLoss(
            model=component_model,
            device=device,
            output_loss_type=config.output_loss_type,
            pgd_config=PGDReconLossConfig(
                init="random",
                step_size=STEP_SIZE,
                n_steps=n_steps,
                mask_scope="shared_across_batch",
            ),
            use_delta_component=config.use_delta_component,
        )

        with torch.no_grad():
            weight_deltas = component_model.calc_weight_deltas()

            for eval_step in range(N_EVAL_STEPS):
                batch_raw = next(eval_iterator)
                batch = extract_batch_data(batch_raw).to(device)

                target_output = component_model(batch, cache_type="input")
                ci = component_model.calc_causal_importances(
                    pre_weight_acts=target_output.cache,
                    detach_inputs=False,
                    sampling=config.sampling,
                )

                metric.update(
                    batch=batch,
                    target_out=target_output.output,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )

                if is_main_process():
                    logger.info(f"  n_steps={n_steps:>2d} | eval_step {eval_step} done")

        loss = metric.compute().item()
        if is_main_process():
            logger.info(f"  n_steps={n_steps:>2d} | RESULT: {loss:.6f}")

        torch.cuda.empty_cache()

    if is_main_process():
        logger.info("Done.")


if __name__ == "__main__":
    main()

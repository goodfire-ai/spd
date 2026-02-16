"""PGDReconLoss eval for an SPD model across PGD step counts using real DDP.

Mimics exactly what happens during a regular eval step in run_spd.py.
For each n_steps value, runs PGDReconLoss with shared_across_batch.
"""

import json
import os
from datetime import datetime, timedelta
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
from spd.utils.general_utils import bf16_autocast, extract_batch_data, set_seed

# ── Eval knobs ──────────────────────────────────────────────────────
RUN_ID = "s-892f140b"
N_BATCHES = 15
N_STEPS_LIST = [5, 20, 50]
STEP_SIZE = 0.1
# ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd")


def pre_cache_on_rank0() -> None:
    """Download model artifacts on LOCAL_RANK=0 before distributed init.

    Large SPD checkpoints can take >10min to download, exceeding the NCCL default timeout
    (600s). By downloading before init_distributed, we avoid NCCL timeouts entirely --
    other ranks simply wait at init_process_group's rendezvous instead.
    """
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
        logger.info(f"Run: {RUN_ID}, PGD n_steps: {N_STEPS_LIST}, step_size: {STEP_SIZE}")
        logger.info(f"Eval batch size: {config.eval_batch_size} (per-rank: {eval_rank_batch_size})")

    component_model = ComponentModel.from_run_info(run_info)
    component_model.to(device)
    component_model.eval()

    per_batch: dict[int, list[float]] = {}

    for n_steps in N_STEPS_LIST:
        eval_loader, _ = create_data_loader(
            dataset_config=eval_data_config,
            batch_size=eval_rank_batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=config.seed + 1,
            dist_state=dist_state,
        )
        eval_iterator = loop_dataloader(eval_loader)

        batch_losses: list[float] = []

        with torch.no_grad(), bf16_autocast(enabled=config.autocast_bf16):
            weight_deltas = component_model.calc_weight_deltas()

            for batch_idx in range(N_BATCHES):
                batch_raw = next(eval_iterator)
                batch = extract_batch_data(batch_raw).to(device)

                target_output = component_model(batch, cache_type="input")
                ci = component_model.calc_causal_importances(
                    pre_weight_acts=target_output.cache,
                    detach_inputs=False,
                    sampling=config.sampling,
                )

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
                metric.update(
                    batch=batch,
                    target_out=target_output.output,
                    ci=ci,
                    weight_deltas=weight_deltas,
                )
                loss = metric.compute().item()
                batch_losses.append(loss)

                if is_main_process():
                    logger.info(
                        f"  n_steps={n_steps:>2d} | batch {batch_idx}: {loss:.6f}"
                    )

        per_batch[n_steps] = batch_losses
        if is_main_process():
            mean = sum(batch_losses) / len(batch_losses)
            logger.info(
                f"  n_steps={n_steps:>2d} | "
                f"mean: {mean:.6f}  batches: {[f'{v:.4f}' for v in batch_losses]}"
            )

        torch.cuda.empty_cache()

    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"=== Summary ({RUN_ID}, mean over {N_BATCHES} batches) ===")
        header = "".join(f"{'n=' + str(n):>12s}" for n in N_STEPS_LIST)
        logger.info(header)
        means = {n: sum(per_batch[n]) / len(per_batch[n]) for n in N_STEPS_LIST}
        row = "".join(f"{means[n]:>12.6f}" for n in N_STEPS_LIST)
        logger.info(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"pgd_recon_eval_{RUN_ID}_{timestamp}.json"
        results_json = {
            "run_id": RUN_ID,
            "n_batches": N_BATCHES,
            "step_size": STEP_SIZE,
            "n_steps_list": N_STEPS_LIST,
            "per_n_steps": {
                str(n): {
                    "batch_losses": per_batch[n],
                    "mean": sum(per_batch[n]) / len(per_batch[n]),
                }
                for n in N_STEPS_LIST
            },
        }
        out_path.write_text(json.dumps(results_json, indent=2))
        logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

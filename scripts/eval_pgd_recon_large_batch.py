"""PGDReconLoss eval with large batch size (1024) across SPD models and PGD step counts.

Same as eval_pgd_recon.py but with eval_batch_size overridden to 1024 (vs 128)
and only 3 batches for a quick comparison.
"""

import gc
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed.distributed_c10d as c10d
import wandb

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
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
    parse_wandb_run_path,
)

BASE_RUN_ID = "s-275c8f21"
FINETUNE_RUN_IDS = ["s-ae03d45a", "s-c37766d7", "s-6e4ac7ae"]
ALL_RUN_IDS = [BASE_RUN_ID, *FINETUNE_RUN_IDS]
RUN_LABELS = {
    "s-275c8f21": "s-275c8f21 (base, imp_min=0.0005)",
    "s-ae03d45a": "s-ae03d45a (imp_min=0.0008)",
    "s-c37766d7": "s-c37766d7 (imp_min=0.001)",
    "s-6e4ac7ae": "s-6e4ac7ae (imp_min=0.0005)",
}
N_STEPS_LIST = [5, 20, 50]
EVAL_BATCH_SIZE = 768
N_BATCHES = 3


def download_checkpoint(run_id: str) -> Path:
    """Download a checkpoint from wandb, returning the local path."""
    entity, project, rid = parse_wandb_run_path(f"wandb:goodfire/spd/{run_id}")
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{rid}")
    run_dir = fetch_wandb_run_dir(run.id)
    checkpoint_file = fetch_latest_wandb_checkpoint(run, prefix="model")
    return download_wandb_file(run, run_dir, checkpoint_file.name)


def pre_cache_on_rank0() -> None:
    """Download all model artifacts on LOCAL_RANK=0 before distributed init."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    logger.info("Pre-caching model checkpoints on LOCAL_RANK=0...")
    base_run_info = SPDRunInfo.from_path(f"wandb:goodfire/spd/{BASE_RUN_ID}")
    assert base_run_info.config.pretrained_model_name is not None
    PretrainRunInfo.from_path(base_run_info.config.pretrained_model_name)
    for run_id in FINETUNE_RUN_IDS:
        download_checkpoint(run_id)
    logger.info("All models cached.")


@with_distributed_cleanup
def main() -> None:
    pre_cache_on_rank0()

    c10d.default_pg_nccl_timeout = timedelta(minutes=30)
    dist_state = init_distributed()
    device = get_device()

    base_run_info = SPDRunInfo.from_path(f"wandb:goodfire/spd/{BASE_RUN_ID}")
    config = base_run_info.config

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
            assert EVAL_BATCH_SIZE % world_size == 0
            eval_rank_batch_size = EVAL_BATCH_SIZE // world_size
        case None:
            eval_rank_batch_size = EVAL_BATCH_SIZE

    if is_main_process():
        logger.info(f"Models: {ALL_RUN_IDS}, PGD n_steps: {N_STEPS_LIST}")
        logger.info(f"Eval batch size: {EVAL_BATCH_SIZE} (per-rank: {eval_rank_batch_size})")

    checkpoint_paths: dict[str, Path] = {BASE_RUN_ID: base_run_info.checkpoint_path}
    for run_id in FINETUNE_RUN_IDS:
        checkpoint_paths[run_id] = download_checkpoint(run_id)

    # per_batch[run_id][n_steps] = list of per-batch losses
    per_batch: dict[str, dict[int, list[float]]] = {}

    for run_id in ALL_RUN_IDS:
        if is_main_process():
            logger.info(f"\nLoading model {RUN_LABELS[run_id]}...")

        component_model = ComponentModel.from_run_info(base_run_info)
        weights = torch.load(checkpoint_paths[run_id], map_location="cpu", weights_only=True)
        component_model.load_state_dict(weights)
        component_model.to(device)
        component_model.eval()

        per_batch[run_id] = {}

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
                            step_size=0.1,
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
                            f"  {RUN_LABELS[run_id]} | n_steps={n_steps:>2d} | "
                            f"batch {batch_idx}: {loss:.6f}"
                        )

            per_batch[run_id][n_steps] = batch_losses
            if is_main_process():
                mean = sum(batch_losses) / len(batch_losses)
                logger.info(
                    f"  {RUN_LABELS[run_id]} | n_steps={n_steps:>2d} | "
                    f"mean: {mean:.6f}  batches: {[f'{v:.4f}' for v in batch_losses]}"
                )

            torch.cuda.empty_cache()

        del component_model
        torch.cuda.empty_cache()
        gc.collect()

    if is_main_process():
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"=== Summary (mean over {N_BATCHES} batches, batch_size={EVAL_BATCH_SIZE}) ==="
        )
        header = f"{'run':<40s}" + "".join(f"{'n=' + str(n):>12s}" for n in N_STEPS_LIST)
        logger.info(header)
        for run_id in ALL_RUN_IDS:
            label = RUN_LABELS[run_id]
            means = {n: sum(per_batch[run_id][n]) / len(per_batch[run_id][n]) for n in N_STEPS_LIST}
            row = f"{label:<40s}" + "".join(f"{means[n]:>12.6f}" for n in N_STEPS_LIST)
            logger.info(row)

        out_path = Path(
            "/mnt/polished-lake/artifacts/mechanisms/spd/pgd_recon_eval_large_batch_results.json"
        )
        results_json = {
            "eval_batch_size": EVAL_BATCH_SIZE,
            "n_batches": N_BATCHES,
            "n_steps_list": N_STEPS_LIST,
            "runs": {
                run_id: {
                    "label": RUN_LABELS[run_id],
                    "per_n_steps": {
                        str(n): {
                            "batch_losses": per_batch[run_id][n],
                            "mean": sum(per_batch[run_id][n]) / len(per_batch[run_id][n]),
                        }
                        for n in N_STEPS_LIST
                    },
                }
                for run_id in ALL_RUN_IDS
            },
        }
        out_path.write_text(json.dumps(results_json, indent=2))
        logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

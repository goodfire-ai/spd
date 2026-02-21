"""Test DP eval invariance: compare eval metrics across dp=1, dp=8, dp=16.

Sub-test 1 (Broadcast): All ranks process identical broadcast data. If all-reduced results
differ from rank-0-only results, the reduction logic is broken.

Sub-test 2 (Sharded): Normal DDP eval pipeline with each rank processing its own data shard.
Compare means/stds across different dp configurations.

Usage:
  python scripts/test_dp_eval_invariance.py           # dp=1
  torchrun --standalone --nproc_per_node=8 scripts/test_dp_eval_invariance.py  # dp=8
"""

import torch
from torch.types import Number

from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.eval import clean_metric_output, evaluate
from spd.metrics.base import Metric
from spd.metrics.ce_and_kl_losses import CEandKLLosses
from spd.metrics.ci_l0 import CI_L0
from spd.metrics.faithfulness_loss import FaithfulnessLoss
from spd.metrics.stochastic_hidden_acts_recon_loss import StochasticHiddenActsReconLoss
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.utils.distributed_utils import (
    broadcast_tensor,
    ensure_cached_and_call,
    get_device,
    get_distributed_state,
    init_distributed,
    is_main_process,
    print0,
    sync_across_processes,
    with_distributed_cleanup,
)
from spd.utils.general_utils import extract_batch_data, set_seed

WANDB_PATH = "wandb:goodfire/spd/runs/s-84ca0717"
N_BROADCAST_BATCHES = 10
N_EVAL_STEPS = 50
# Global batch size split across ranks. dp=1 gets the full batch on 1 GPU,
# so keep small enough to fit in memory with CEandKLLosses (multiple forward passes).
EVAL_BATCH_SIZE = 16
# Broadcast test: each rank processes the FULL batch
BROADCAST_BATCH_SIZE = 16


def load_model_and_config(device: str) -> tuple[ComponentModel, SPDRunInfo]:
    run_info = ensure_cached_and_call(SPDRunInfo.from_path, WANDB_PATH)
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()
    return model, run_info


def init_metrics(model: ComponentModel, config, device: str) -> list[Metric]:
    return [
        FaithfulnessLoss(model=model, device=device),
        CI_L0(
            model=model,
            device=device,
            ci_alive_threshold=config.ci_alive_threshold,
            groups=None,
        ),
        StochasticHiddenActsReconLoss(
            model=model,
            device=device,
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            n_mask_samples=config.n_mask_samples,
        ),
        CEandKLLosses(
            model=model,
            device=device,
            sampling=config.sampling,
            rounding_threshold=0.5,
        ),
    ]


def run_broadcast_test(model: ComponentModel, config, device: str) -> None:
    """Sub-test 1: All ranks process identical broadcast data."""
    print0("\n" + "=" * 80)
    print0("SUB-TEST 1: BROADCAST TEST (identical data across ranks)")
    print0("=" * 80)

    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state else 1

    task_config = config.task_config
    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=42,
    )

    # Only rank 0 loads data; other ranks receive via broadcast
    if is_main_process():
        loader, _ = create_data_loader(
            dataset_config=eval_data_config,
            batch_size=BROADCAST_BATCH_SIZE,
            buffer_size=task_config.buffer_size,
            global_seed=42,
            dist_state=None,
        )
        data_iter = iter(loader)

    for batch_idx in range(N_BROADCAST_BATCHES):
        if is_main_process():
            batch = extract_batch_data(next(data_iter)).to(device)
        else:
            batch = torch.zeros(
                BROADCAST_BATCH_SIZE, task_config.max_seq_len, dtype=torch.long, device=device
            )

        broadcast_tensor(batch)

        # All ranks process the FULL batch with same seed
        set_seed(batch_idx + 1000)

        with torch.no_grad():
            weight_deltas = model.calc_weight_deltas()
            target_output: OutputWithCache = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=target_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )

            dist_metrics = init_metrics(model, config, device)
            for metric in dist_metrics:
                metric.update(
                    batch=batch,
                    target_out=target_output.output,
                    pre_weight_acts=target_output.cache,
                    ci=ci,
                    current_frac_of_training=1.0,
                    weight_deltas=weight_deltas,
                )

            dist_results: dict[str, float] = {}
            for metric in dist_metrics:
                computed_raw = metric.compute()
                computed = clean_metric_output(
                    section=metric.metric_section,
                    metric_name=type(metric).__name__,
                    computed_raw=computed_raw,
                )
                for k, v in computed.items():
                    if isinstance(v, Number):
                        dist_results[k] = float(v)

        if is_main_process():
            print(f"\n--- Batch {batch_idx} ---")
            for k in sorted(dist_results.keys()):
                if "bar_chart" in k:
                    continue
                print(f"  {k}: {dist_results[k]:.10f}")

    if is_main_process():
        print("\nBroadcast test complete. Compare values across dp=1/8/16 runs.")
        print("FaithfulnessLoss should be IDENTICAL across all dp values.")
        print("All metrics should be IDENTICAL since all ranks see same data + same seed.")


def run_sharded_eval_test(model: ComponentModel, config, device: str) -> None:
    """Sub-test 2: Normal DDP eval with sharded data."""
    print0("\n" + "=" * 80)
    print0("SUB-TEST 2: SHARDED EVAL TEST (normal DDP eval)")
    print0("=" * 80)

    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state else 1
    task_config = config.task_config

    eval_rank_batch_size = EVAL_BATCH_SIZE // world_size
    assert EVAL_BATCH_SIZE % world_size == 0, (
        f"EVAL_BATCH_SIZE={EVAL_BATCH_SIZE} not divisible by world_size={world_size}"
    )

    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=42,
    )

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_rank_batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=42,
        dist_state=dist_state,
    )
    eval_iterator = loop_dataloader(eval_loader)

    # Use the evaluate function directly
    from spd.configs import (
        CEandKLLossesConfig,
        CI_L0Config,
        FaithfulnessLossConfig,
        StochasticHiddenActsReconLossConfig,
    )

    eval_metric_configs = [
        FaithfulnessLossConfig(),
        CI_L0Config(groups=None),
        StochasticHiddenActsReconLossConfig(),
        CEandKLLossesConfig(rounding_threshold=0.5),
    ]

    set_seed(42)
    with torch.no_grad():
        metrics = evaluate(
            eval_metric_configs=eval_metric_configs,
            model=model,
            eval_iterator=eval_iterator,
            device=device,
            run_config=config,
            slow_step=False,
            n_eval_steps=N_EVAL_STEPS,
            current_frac_of_training=1.0,
        )

    # Metrics already all_reduce in their compute() methods, no extra averaging needed
    if is_main_process():
        print(f"\nSharded eval results (dp={world_size}, {N_EVAL_STEPS} steps):")
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if isinstance(v, Number):
                print(f"  {k}: {float(v):.10f}")

    sync_across_processes()


@with_distributed_cleanup
def main() -> None:
    dist_state = init_distributed()
    device = get_device()

    world_size = dist_state.world_size if dist_state else 1
    print0(f"Running DP eval invariance test with dp={world_size}")
    print0(f"Model: {WANDB_PATH}")

    model, run_info = load_model_and_config(device)
    config = run_info.config
    print0(f"Config loaded: sampling={config.sampling}, n_mask_samples={config.n_mask_samples}")

    run_broadcast_test(model, config, device)
    run_sharded_eval_test(model, config, device)

    print0("\n" + "=" * 80)
    print0("ALL TESTS COMPLETE")
    print0("=" * 80)


if __name__ == "__main__":
    main()

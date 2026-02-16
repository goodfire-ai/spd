"""PGDReconLoss eval replicating the OLD semantics from training.

Differences vs current code:
  1. No broadcast_tensor on PGD source init — each rank gets independent random init
  2. ReduceOp.SUM (not AVG) for gradient all-reduce
  3. No bf16 autocast (didn't exist in the snapshot that trained s-892f140b)
"""

import json
import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.distributed.distributed_c10d as c10d
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import LMTaskConfig, PGDConfig, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import RoutingMasks, make_mask_infos
from spd.pretrain.run_info import PretrainRunInfo
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import (
    DistributedState,
    all_reduce,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import (
    calc_sum_recon_loss_lm,
    extract_batch_data,
    set_seed,
)

# ── Eval knobs ──────────────────────────────────────────────────────
RUN_ID = "s-892f140b"
N_BATCHES = 15
N_STEPS_LIST = [5, 20]
STEP_SIZE = 0.1
# ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd")


# ── Old PGD implementation (pre-fix DDP semantics) ──────────────────


def _old_forward_with_adv_sources(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    adv_sources: dict[str, Float[Tensor, "..."]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    batch_dims: tuple[int, ...],
) -> tuple[Float[Tensor, ""], int]:
    expanded_adv_sources = {k: v.expand(*batch_dims, -1) for k, v in adv_sources.items()}
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = expanded_adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas
            }
            adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}

    component_masks: dict[str, Float[Tensor, ...]] = {}
    for module_name in ci:
        adv_source = adv_sources_components[module_name]
        scaled_noise = (1 - ci[module_name]) * adv_source
        component_masks[module_name] = ci[module_name] + scaled_noise

    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )
    out = model(batch, mask_infos=mask_infos)
    sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = (
        target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
    )
    return sum_loss, n_examples


def old_pgd_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Old PGD: independent random init per rank, ReduceOp.SUM for grads."""
    batch_dims = next(iter(ci.values())).shape[:-1]

    router = AllLayersRouter()
    routing_masks = router.get_masks(module_names=model.target_module_paths, mask_shape=batch_dims)

    adv_sources: dict[str, Float[Tensor, ...]] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if weight_deltas is None else module_c + 1
        match pgd_config.mask_scope:
            case "unique_per_datapoint":
                shape = torch.Size([*batch_dims, mask_c])
            case "shared_across_batch":
                singleton_batch_dims = [1 for _ in batch_dims]
                shape = torch.Size([*singleton_batch_dims, mask_c])
        # OLD: no broadcast — each rank gets its own random init
        adv_sources[module_name] = torch.rand(shape, device=batch.device).requires_grad_(True)

    fwd_pass = partial(
        _old_forward_with_adv_sources,
        model=model,
        batch=batch,
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        target_out=target_out,
        output_loss_type=output_loss_type,
        batch_dims=batch_dims,
    )

    for _ in range(pgd_config.n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        with torch.enable_grad():
            sum_loss, n_examples = fwd_pass()
            loss = sum_loss / n_examples
        grads = torch.autograd.grad(loss, list(adv_sources.values()))
        # OLD: ReduceOp.SUM (not AVG)
        adv_sources_grads = {
            k: all_reduce(g, op=ReduceOp.SUM)
            for k, g in zip(adv_sources.keys(), grads, strict=True)
        }
        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(pgd_config.step_size * adv_sources_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    return fwd_pass()


# ── Main ────────────────────────────────────────────────────────────


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
        logger.info(f"Run: {RUN_ID} (OLD DDP semantics)")
        logger.info(f"PGD n_steps: {N_STEPS_LIST}, step_size: {STEP_SIZE}")
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

        with torch.no_grad():
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

                pgd_config = PGDReconLossConfig(
                    init="random",
                    step_size=STEP_SIZE,
                    n_steps=n_steps,
                    mask_scope="shared_across_batch",
                )
                sum_loss, n_examples = old_pgd_recon_loss(
                    model=component_model,
                    batch=batch,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    target_out=target_output.output,
                    output_loss_type=config.output_loss_type,
                    pgd_config=pgd_config,
                )
                sum_loss = all_reduce(sum_loss, op=ReduceOp.SUM)
                n_examples_total = all_reduce(
                    torch.tensor(n_examples, device=device), op=ReduceOp.SUM
                )
                loss = (sum_loss / n_examples_total).item()
                batch_losses.append(loss)

                if is_main_process():
                    logger.info(f"  n_steps={n_steps:>2d} | batch {batch_idx}: {loss:.6f}")

        per_batch[n_steps] = batch_losses
        if is_main_process():
            mean = sum(batch_losses) / len(batch_losses)
            logger.info(
                f"  n_steps={n_steps:>2d} | "
                f"mean: {mean:.6f}  batches: {[f'{v:.4f}' for v in batch_losses]}"
            )

        torch.cuda.empty_cache()

    if is_main_process():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"=== Summary ({RUN_ID}, OLD DDP, mean over {N_BATCHES} batches) ===")
        header = "".join(f"{'n=' + str(n):>12s}" for n in N_STEPS_LIST)
        logger.info(header)
        means = {n: sum(per_batch[n]) / len(per_batch[n]) for n in N_STEPS_LIST}
        row = "".join(f"{means[n]:>12.6f}" for n in N_STEPS_LIST)
        logger.info(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"pgd_recon_eval_old_ddp_{RUN_ID}_{timestamp}.json"
        results_json = {
            "run_id": RUN_ID,
            "n_batches": N_BATCHES,
            "step_size": STEP_SIZE,
            "n_steps_list": N_STEPS_LIST,
            "note": "OLD DDP semantics: no broadcast, ReduceOp.SUM",
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

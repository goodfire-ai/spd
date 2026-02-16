"""Compare PGD recon loss: shared sources (fixed) vs independent sources (broken).

Simulates 8-rank DDP on a single GPU. The "fixed" variant initializes sources once
and shares them across all ranks (as broadcast_tensor would). The "broken"
variant gives each rank independent random sources but averages their gradients.
"""

import time

import torch

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.metrics.pgd_utils import _forward_with_adv_sources
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.routing import AllLayersRouter
from spd.utils.general_utils import extract_batch_data


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pgd_recon_shared_sources(
    model: ComponentModel,
    sub_batches: list[torch.Tensor],
    sub_cis: list[dict[str, torch.Tensor]],
    weight_deltas: dict[str, torch.Tensor] | None,
    sub_target_outs: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    source_seed: int,
) -> tuple[torch.Tensor, int]:
    """Fixed: all ranks share the same source init (broadcast behavior)."""
    n_ranks = len(sub_batches)
    sub_batch_dims = sub_batches[0].shape

    routing_masks = AllLayersRouter().get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    gen = torch.Generator(device=sub_batches[0].device)
    gen.manual_seed(source_seed)

    adv_sources: dict[str, torch.Tensor] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if weight_deltas is None else module_c + 1
        singleton_batch_dims = [1 for _ in sub_batch_dims]
        shape = torch.Size([*singleton_batch_dims, mask_c])
        adv_sources[module_name] = torch.rand(
            shape, device=sub_batches[0].device, generator=gen
        ).requires_grad_(True)

    for _ in range(n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        avg_grads = {k: torch.zeros_like(v) for k, v in adv_sources.items()}

        for rank in range(n_ranks):
            with torch.enable_grad():
                sum_loss_r, n_examples_r = _forward_with_adv_sources(
                    model=model,
                    batch=sub_batches[rank],
                    adv_sources=adv_sources,
                    ci=sub_cis[rank],
                    weight_deltas=weight_deltas,
                    routing_masks=routing_masks,
                    target_out=sub_target_outs[rank],
                    output_loss_type="kl",
                    batch_dims=sub_batch_dims,
                )
                loss_r = sum_loss_r / n_examples_r

            grads = torch.autograd.grad(loss_r, list(adv_sources.values()))
            for k, g in zip(adv_sources.keys(), grads, strict=True):
                avg_grads[k] += g.detach() / n_ranks

        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(step_size * avg_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    total_sum_loss = torch.tensor(0.0, device=sub_batches[0].device)
    total_n_examples = 0
    for rank in range(n_ranks):
        sum_loss_r, n_examples_r = _forward_with_adv_sources(
            model=model,
            batch=sub_batches[rank],
            adv_sources=adv_sources,
            ci=sub_cis[rank],
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=sub_target_outs[rank],
            output_loss_type="kl",
            batch_dims=sub_batch_dims,
        )
        total_sum_loss = total_sum_loss + sum_loss_r.detach()
        total_n_examples += n_examples_r

    return total_sum_loss, total_n_examples


def pgd_recon_independent_sources(
    model: ComponentModel,
    sub_batches: list[torch.Tensor],
    sub_cis: list[dict[str, torch.Tensor]],
    weight_deltas: dict[str, torch.Tensor] | None,
    sub_target_outs: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    source_seed: int,
) -> tuple[torch.Tensor, int]:
    """Broken: each rank gets independent random source init, but gradients are averaged."""
    n_ranks = len(sub_batches)
    sub_batch_dims = sub_batches[0].shape

    routing_masks = AllLayersRouter().get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    # Each rank gets its own independent sources (the bug)
    per_rank_adv_sources: list[dict[str, torch.Tensor]] = []
    for rank in range(n_ranks):
        gen = torch.Generator(device=sub_batches[0].device)
        gen.manual_seed(source_seed + rank * 1000)
        rank_sources: dict[str, torch.Tensor] = {}
        for module_name in model.target_module_paths:
            module_c = model.module_to_c[module_name]
            mask_c = module_c if weight_deltas is None else module_c + 1
            singleton_batch_dims = [1 for _ in sub_batch_dims]
            shape = torch.Size([*singleton_batch_dims, mask_c])
            rank_sources[module_name] = torch.rand(
                shape, device=sub_batches[0].device, generator=gen
            ).requires_grad_(True)
        per_rank_adv_sources.append(rank_sources)

    for _ in range(n_steps):
        # Each rank computes its own gradient from its own sources
        per_rank_grads: list[dict[str, torch.Tensor]] = []
        for rank in range(n_ranks):
            assert all(adv.grad is None for adv in per_rank_adv_sources[rank].values())
            with torch.enable_grad():
                sum_loss_r, n_examples_r = _forward_with_adv_sources(
                    model=model,
                    batch=sub_batches[rank],
                    adv_sources=per_rank_adv_sources[rank],
                    ci=sub_cis[rank],
                    weight_deltas=weight_deltas,
                    routing_masks=routing_masks,
                    target_out=sub_target_outs[rank],
                    output_loss_type="kl",
                    batch_dims=sub_batch_dims,
                )
                loss_r = sum_loss_r / n_examples_r

            grads = torch.autograd.grad(loss_r, list(per_rank_adv_sources[rank].values()))
            per_rank_grads.append(dict(zip(per_rank_adv_sources[rank].keys(), grads, strict=True)))

        # Average gradients across ranks (the all_reduce(AVG))
        avg_grads = {k: torch.zeros_like(v) for k, v in per_rank_adv_sources[0].items()}
        for rank in range(n_ranks):
            for k in avg_grads:
                avg_grads[k] += per_rank_grads[rank][k].detach() / n_ranks

        # All ranks apply the same averaged gradient to their own (different) sources
        with torch.no_grad():
            for rank in range(n_ranks):
                for k in per_rank_adv_sources[rank]:
                    per_rank_adv_sources[rank][k].add_(step_size * avg_grads[k].sign())
                    per_rank_adv_sources[rank][k].clamp_(0.0, 1.0)

    # Final eval: each rank uses its own sources
    total_sum_loss = torch.tensor(0.0, device=sub_batches[0].device)
    total_n_examples = 0
    for rank in range(n_ranks):
        sum_loss_r, n_examples_r = _forward_with_adv_sources(
            model=model,
            batch=sub_batches[rank],
            adv_sources=per_rank_adv_sources[rank],
            ci=sub_cis[rank],
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=sub_target_outs[rank],
            output_loss_type="kl",
            batch_dims=sub_batch_dims,
        )
        total_sum_loss = total_sum_loss + sum_loss_r.detach()
        total_n_examples += n_examples_r

    return total_sum_loss, total_n_examples


WANDB_PATH = "wandb:goodfire/spd/runs/s-275c8f21"
N_EVAL_BATCHES = 1
N_RANKS = 8
EVAL_BATCH_SIZE = 128
RANK_BATCH_SIZE = EVAL_BATCH_SIZE // N_RANKS
N_STEPS_LIST = [5, 10, 20]


def main():
    device = "cuda:0"

    log(f"Loading run info from {WANDB_PATH}...")
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    config = run_info.config
    log(
        f"Config loaded. output_loss_type={config.output_loss_type}, "
        f"use_delta_component={config.use_delta_component}"
    )

    log("Loading model...")
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()
    log(
        f"Model loaded. {len(model.target_module_paths)} target modules, "
        f"module_to_c keys: {list(model.module_to_c.keys())[:3]}..."
    )

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
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

    log(
        f"Creating data loader (batch_size={RANK_BATCH_SIZE}, dataset={task_config.dataset_name})..."
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=RANK_BATCH_SIZE,
        buffer_size=task_config.buffer_size,
        global_seed=42,
    )
    log("Data loader ready.")

    variants = {
        "FIXED (shared sources)": pgd_recon_shared_sources,
        "BROKEN (independent sources)": pgd_recon_independent_sources,
    }

    all_results: dict[str, dict[int, float]] = {}

    for variant_name, pgd_fn in variants.items():
        log(f"\n{'=' * 70}")
        log(f"Variant: {variant_name}")
        log(f"{'=' * 70}")

        variant_results: dict[int, float] = {}

        for n_steps in N_STEPS_LIST:
            log(f"Starting n_steps={n_steps}...")
            eval_iterator = loop_dataloader(eval_loader)
            total_sum_loss = torch.tensor(0.0, device=device)
            total_n_examples = 0

            for eval_batch_idx in range(N_EVAL_BATCHES):
                batch_start = time.time()

                sub_batches = []
                sub_cis = []
                sub_target_outs = []

                for _rank_idx in range(N_RANKS):
                    raw = next(eval_iterator)
                    sub_batch = extract_batch_data(raw).to(device)
                    with torch.no_grad():
                        target_output = model(sub_batch, cache_type="input")
                        ci = model.calc_causal_importances(
                            pre_weight_acts=target_output.cache,
                            detach_inputs=False,
                            sampling=config.sampling,
                        )
                    sub_batches.append(sub_batch)
                    sub_cis.append(ci.lower_leaky)
                    sub_target_outs.append(target_output.output.detach())
                    del target_output, ci

                weight_deltas = model.calc_weight_deltas()
                source_seed = n_steps * 10000 + eval_batch_idx

                pgd_start = time.time()
                sum_loss, n_examples = pgd_fn(
                    model=model,
                    sub_batches=sub_batches,
                    sub_cis=sub_cis,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    sub_target_outs=sub_target_outs,
                    n_steps=n_steps,
                    step_size=0.1,
                    source_seed=source_seed,
                )
                pgd_elapsed = time.time() - pgd_start

                total_sum_loss += sum_loss
                total_n_examples += n_examples

                batch_avg = (total_sum_loss / total_n_examples).item()
                batch_elapsed = time.time() - batch_start
                log(
                    f"  n_steps={n_steps:>3d} | batch {eval_batch_idx + 1:>2d}/{N_EVAL_BATCHES}"
                    f" | running avg: {batch_avg:.4f}"
                    f" | pgd: {pgd_elapsed:.1f}s | total: {batch_elapsed:.1f}s"
                )

                del sub_batches, sub_cis, sub_target_outs, weight_deltas
                torch.cuda.empty_cache()

            final_loss = (total_sum_loss / total_n_examples).item()
            variant_results[n_steps] = final_loss
            log(f">>> n_steps={n_steps:>3d} | FINAL: {final_loss:.6f}")

        all_results[variant_name] = variant_results

    log(f"\n\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")
    header = f"{'n_steps':>8s}"
    for variant_name in all_results:
        header += f"  | {variant_name:>30s}"
    log(header)
    log("-" * len(header))
    for n_steps in N_STEPS_LIST:
        row = f"{n_steps:>8d}"
        for variant_name in all_results:
            row += f"  | {all_results[variant_name][n_steps]:>30.6f}"
        log(row)


if __name__ == "__main__":
    main()

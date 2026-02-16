"""Test why standalone PGDReconLoss differs from DDP eval.

Hypothesis: in DDP, each rank inits its own random adv_sources independently. The all_reduce(AVG)
averages gradients computed at DIFFERENT source values, which is not a valid gradient for any rank.
This makes PGD much less effective → lower loss in DDP.

We test: "shared sources" (standalone-style) vs "independent sources per rank" (DDP-style).
"""

import contextlib

import torch

from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.metrics.pgd_utils import _forward_with_adv_sources, _get_pgd_init_tensor
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.routing import AllLayersRouter
from spd.utils.general_utils import extract_batch_data

WANDB_PATH = "wandb:goodfire/spd/runs/s-275c8f21"
N_EVAL_BATCHES = 10
N_RANKS = 8
EVAL_BATCH_SIZE = 128
RANK_BATCH_SIZE = EVAL_BATCH_SIZE // N_RANKS
N_STEPS = 20


def pgd_shared_sources(
    model: ComponentModel,
    sub_batches: list[torch.Tensor],
    sub_cis: list[dict[str, torch.Tensor]],
    weight_deltas: dict[str, torch.Tensor] | None,
    sub_target_outs: list[torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """One set of sources, gradients averaged across ranks. (Standalone script behavior.)"""
    n_ranks = len(sub_batches)
    sub_batch_dims = sub_batches[0].shape
    routing_masks = AllLayersRouter().get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    adv_sources: dict[str, torch.Tensor] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if weight_deltas is None else module_c + 1
        singleton_batch_dims = [1 for _ in sub_batch_dims]
        shape = torch.Size([*singleton_batch_dims, mask_c])
        adv_sources[module_name] = _get_pgd_init_tensor(
            "random", shape, sub_batches[0].device
        ).requires_grad_(True)

    for _ in range(N_STEPS):
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
                adv_sources[k].add_(0.1 * avg_grads[k].sign())
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
        total_sum_loss += sum_loss_r.detach()
        total_n_examples += n_examples_r
    return total_sum_loss, total_n_examples


def pgd_independent_sources(
    model: ComponentModel,
    sub_batches: list[torch.Tensor],
    sub_cis: list[dict[str, torch.Tensor]],
    weight_deltas: dict[str, torch.Tensor] | None,
    sub_target_outs: list[torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Each rank has its own random sources, gradients averaged. (DDP behavior.)"""
    n_ranks = len(sub_batches)
    sub_batch_dims = sub_batches[0].shape
    routing_masks = AllLayersRouter().get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    # Each rank gets independent random sources (simulating DDP)
    per_rank_sources: list[dict[str, torch.Tensor]] = []
    for _ in range(n_ranks):
        rank_sources: dict[str, torch.Tensor] = {}
        for module_name in model.target_module_paths:
            module_c = model.module_to_c[module_name]
            mask_c = module_c if weight_deltas is None else module_c + 1
            singleton_batch_dims = [1 for _ in sub_batch_dims]
            shape = torch.Size([*singleton_batch_dims, mask_c])
            rank_sources[module_name] = _get_pgd_init_tensor(
                "random", shape, sub_batches[0].device
            ).requires_grad_(True)
        per_rank_sources.append(rank_sources)

    for _ in range(N_STEPS):
        # Collect grads from each rank (at its own source values)
        all_grads: list[dict[str, torch.Tensor]] = []
        for rank in range(n_ranks):
            with torch.enable_grad():
                sum_loss_r, n_examples_r = _forward_with_adv_sources(
                    model=model,
                    batch=sub_batches[rank],
                    adv_sources=per_rank_sources[rank],
                    ci=sub_cis[rank],
                    weight_deltas=weight_deltas,
                    routing_masks=routing_masks,
                    target_out=sub_target_outs[rank],
                    output_loss_type="kl",
                    batch_dims=sub_batch_dims,
                )
                loss_r = sum_loss_r / n_examples_r
            grads = torch.autograd.grad(loss_r, list(per_rank_sources[rank].values()))
            all_grads.append(
                {k: g.detach() for k, g in zip(per_rank_sources[rank].keys(), grads, strict=True)}
            )

        # Average gradients across ranks (simulating all_reduce AVG)
        module_names = list(per_rank_sources[0].keys())
        avg_grads = {k: torch.zeros_like(per_rank_sources[0][k]) for k in module_names}
        for rank_grads in all_grads:
            for k in module_names:
                avg_grads[k] += rank_grads[k] / n_ranks

        # All ranks apply the same sign update (but to different source values)
        with torch.no_grad():
            for rank in range(n_ranks):
                for k in module_names:
                    per_rank_sources[rank][k].add_(0.1 * avg_grads[k].sign())
                    per_rank_sources[rank][k].clamp_(0.0, 1.0)

    # Final loss: each rank evaluates with its own sources
    total_sum_loss = torch.tensor(0.0, device=sub_batches[0].device)
    total_n_examples = 0
    for rank in range(n_ranks):
        sum_loss_r, n_examples_r = _forward_with_adv_sources(
            model=model,
            batch=sub_batches[rank],
            adv_sources=per_rank_sources[rank],
            ci=sub_cis[rank],
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=sub_target_outs[rank],
            output_loss_type="kl",
            batch_dims=sub_batch_dims,
        )
        total_sum_loss += sum_loss_r.detach()
        total_n_examples += n_examples_r
    return total_sum_loss, total_n_examples


def run_eval(mode: str) -> float:
    assert mode in ("shared", "independent", "bf16_shared")
    device = "cuda:0"
    use_bf16 = mode == "bf16_shared"
    pgd_fn = pgd_shared_sources if mode != "independent" else pgd_independent_sources
    print(f"\n{'=' * 60}")
    print(f"Running: {mode}")
    print(f"{'=' * 60}")

    model = ComponentModel.from_pretrained(WANDB_PATH)
    config = SPDRunInfo.from_path(WANDB_PATH).config
    model.to(device)
    model.eval()

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
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=RANK_BATCH_SIZE,
        buffer_size=task_config.buffer_size,
        global_seed=42,
    )
    eval_iterator = loop_dataloader(eval_loader)

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    )

    total_sum_loss = torch.tensor(0.0, device=device)
    total_n_examples = 0

    with autocast_ctx:
        for batch_idx in range(N_EVAL_BATCHES):
            sub_batches = []
            sub_cis = []
            sub_target_outs = []

            for _ in range(N_RANKS):
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

            sum_loss, n_examples = pgd_fn(
                model=model,
                sub_batches=sub_batches,
                sub_cis=sub_cis,
                weight_deltas=weight_deltas if config.use_delta_component else None,
                sub_target_outs=sub_target_outs,
            )
            total_sum_loss += sum_loss
            total_n_examples += n_examples

            running_avg = (total_sum_loss / total_n_examples).item()
            print(
                f"  [{mode}] batch {batch_idx + 1}/{N_EVAL_BATCHES} | running avg: {running_avg:.4f}"
            )

            del sub_batches, sub_cis, sub_target_outs, weight_deltas
            torch.cuda.empty_cache()

    final_loss = (total_sum_loss / total_n_examples).item()
    print(f">>> [{mode}] FINAL: {final_loss:.6f}")

    del model
    torch.cuda.empty_cache()
    return final_loss


def main():
    results = {}
    for mode in ("shared", "independent", "bf16_shared"):
        results[mode] = run_eval(mode)

    print(f"\n{'=' * 60}")
    print("SUMMARY (n_steps=20)")
    print(f"{'=' * 60}")
    for mode, loss in results.items():
        print(f"  {mode:>20s}: {loss:.4f}")
    print("\n  WandB regular eval: ~1-5 (for reference)")


if __name__ == "__main__":
    main()

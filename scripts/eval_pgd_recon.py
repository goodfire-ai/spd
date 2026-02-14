"""Evaluate PGDReconLoss at different n_steps, simulating DDP gradient averaging.

Replicates the training-time eval behavior where 8 ranks each compute PGD gradients
on a sub-batch of 8 examples, then all_reduce(AVG) the gradients before the sign step.
"""


import torch

from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.metrics.pgd_utils import (
    _forward_with_adv_sources,
    _get_pgd_init_tensor,
)
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.routing import AllLayersRouter
from spd.utils.general_utils import extract_batch_data


def pgd_recon_loss_simulated_ddp(
    model: ComponentModel,
    batch: torch.Tensor,
    ci: dict[str, torch.Tensor],
    weight_deltas: dict[str, torch.Tensor] | None,
    target_out: torch.Tensor,
    n_steps: int,
    step_size: float,
    n_ranks: int,
) -> tuple[torch.Tensor, int]:
    """PGD with shared_across_batch, simulating DDP gradient averaging over n_ranks sub-batches."""
    rank_batch_size = batch.shape[0] // n_ranks
    assert batch.shape[0] == rank_batch_size * n_ranks

    # Split data into per-rank sub-batches
    batch_chunks = batch.chunk(n_ranks)
    target_out_chunks = target_out.chunk(n_ranks)
    ci_chunks = [{k: v.chunk(n_ranks) for k, v in ci.items()} for _ in range(1)]
    ci_per_rank = [
        {k: v.chunk(n_ranks)[r] for k, v in ci.items()} for r in range(n_ranks)
    ]

    router = AllLayersRouter()
    # batch_dims for shared_across_batch: use per-rank sub-batch dims
    sub_batch_dims = batch_chunks[0].shape  # (8, seq_len)

    routing_masks = router.get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    # Initialize shared adversarial sources (singleton batch dims)
    adv_sources: dict[str, torch.Tensor] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = module_c if weight_deltas is None else module_c + 1
        singleton_batch_dims = [1 for _ in sub_batch_dims]
        shape = torch.Size([*singleton_batch_dims, mask_c])
        adv_sources[module_name] = _get_pgd_init_tensor(
            "random", shape, batch.device
        ).requires_grad_(True)

    for _ in range(n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        # Accumulate gradients across simulated ranks
        avg_grads = {k: torch.zeros_like(v) for k, v in adv_sources.items()}

        for rank in range(n_ranks):
            with torch.enable_grad():
                sum_loss_r, n_examples_r = _forward_with_adv_sources(
                    model=model,
                    batch=batch_chunks[rank],
                    adv_sources=adv_sources,
                    ci=ci_per_rank[rank],
                    weight_deltas=weight_deltas,
                    routing_masks=routing_masks,
                    target_out=target_out_chunks[rank],
                    output_loss_type="kl",
                    batch_dims=sub_batch_dims,
                )
                loss_r = sum_loss_r / n_examples_r

            grads = torch.autograd.grad(loss_r, list(adv_sources.values()), retain_graph=False)
            for k, g in zip(adv_sources.keys(), grads, strict=True):
                avg_grads[k] += g.detach() / n_ranks

        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(step_size * avg_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    # Final forward pass: sum loss and examples across all ranks
    total_sum_loss = torch.tensor(0.0, device=batch.device)
    total_n_examples = 0
    for rank in range(n_ranks):
        sum_loss_r, n_examples_r = _forward_with_adv_sources(
            model=model,
            batch=batch_chunks[rank],
            adv_sources=adv_sources,
            ci=ci_per_rank[rank],
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=target_out_chunks[rank],
            output_loss_type="kl",
            batch_dims=sub_batch_dims,
        )
        total_sum_loss = total_sum_loss + sum_loss_r.detach()
        total_n_examples += n_examples_r

    return total_sum_loss, total_n_examples


def main():
    wandb_path = "wandb:goodfire/spd/runs/s-b9582efc"
    device = "cuda:0"
    n_eval_batches = 5
    n_ranks = 8
    global_batch_size = 64
    rank_batch_size = global_batch_size // n_ranks  # 8

    print("Loading run info...")
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config
    print(f"Output loss type: {config.output_loss_type}")
    print(f"Use delta component: {config.use_delta_component}")

    print("Loading model...")
    model = ComponentModel.from_run_info(run_info)
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

    # Load with per-rank batch size to avoid OOM
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=rank_batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=42,
    )

    n_steps_list = [1, 5, 20, 100]

    for n_steps in n_steps_list:
        eval_iterator = loop_dataloader(eval_loader)
        total_sum_loss = torch.tensor(0.0, device=device)
        total_n_examples = 0

        for eval_batch_idx in range(n_eval_batches):
            # Gather n_ranks sub-batches to form one global batch
            sub_batches = []
            for _ in range(n_ranks):
                raw = next(eval_iterator)
                sub_batches.append(extract_batch_data(raw).to(device))
            global_batch = torch.cat(sub_batches, dim=0)

            # Compute target output and CI on the full batch
            with torch.no_grad():
                target_output = model(global_batch, cache_type="input")
                ci = model.calc_causal_importances(
                    pre_weight_acts=target_output.cache,
                    detach_inputs=False,
                    sampling=config.sampling,
                )
            weight_deltas = model.calc_weight_deltas()

            sum_loss, n_examples = pgd_recon_loss_simulated_ddp(
                model=model,
                batch=global_batch,
                ci=ci.lower_leaky,
                weight_deltas=weight_deltas if config.use_delta_component else None,
                target_out=target_output.output,
                n_steps=n_steps,
                step_size=0.1,
                n_ranks=n_ranks,
            )
            total_sum_loss += sum_loss
            total_n_examples += n_examples
            torch.cuda.empty_cache()

        final_loss = (total_sum_loss / total_n_examples).item()
        print(f"n_steps={n_steps:>3d} | PGDReconLoss (simulated {n_ranks}-rank DDP): {final_loss:.6f}")


if __name__ == "__main__":
    main()

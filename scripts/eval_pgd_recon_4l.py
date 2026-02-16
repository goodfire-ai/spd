"""Evaluate PGDReconLoss at different n_steps for finetune vs original 4L model.

Simulates 8-rank DDP gradient averaging on a single GPU.
Computes CI/target per sub-batch (rank_batch_size=16) to reduce peak memory.
Uses paired seeding: all models see the same data and source init per eval batch.
"""

import torch

from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.metrics.pgd_utils import _forward_with_adv_sources
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.routing import AllLayersRouter
from spd.utils.general_utils import extract_batch_data


def pgd_recon_loss_simulated_ddp(
    model: ComponentModel,
    sub_batches: list[torch.Tensor],
    sub_cis: list[dict[str, torch.Tensor]],
    weight_deltas: dict[str, torch.Tensor] | None,
    sub_target_outs: list[torch.Tensor],
    n_steps: int,
    step_size: float,
    source_seed: int,
) -> tuple[torch.Tensor, int]:
    """PGD with shared_across_batch, simulating DDP gradient averaging."""
    n_ranks = len(sub_batches)
    sub_batch_dims = sub_batches[0].shape

    routing_masks = AllLayersRouter().get_masks(
        module_names=model.target_module_paths, mask_shape=sub_batch_dims
    )

    # Seed source init for reproducibility across models
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


MODELS = {
    "original (s-275c8f21)": "wandb:goodfire/spd/runs/s-275c8f21",
    "ft IM=0.0005 (s-6e4ac7ae)": "wandb:goodfire/spd/runs/s-6e4ac7ae",
    "ft IM=0.0008 (s-ae03d45a)": "wandb:goodfire/spd/runs/s-ae03d45a",
    "ft IM=0.001 (s-c37766d7)": "wandb:goodfire/spd/runs/s-c37766d7",
}

N_EVAL_BATCHES = 10
N_RANKS = 8
EVAL_BATCH_SIZE = 128
RANK_BATCH_SIZE = EVAL_BATCH_SIZE // N_RANKS  # 16
N_STEPS_LIST = [5, 20]


def main():
    device = "cuda:0"

    all_results: dict[str, dict[int, float]] = {}

    for model_name, wandb_path in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        print("Loading model...")
        model = ComponentModel.from_pretrained(wandb_path)
        config = SPDRunInfo.from_path(wandb_path).config
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

        model_results: dict[int, float] = {}

        for n_steps in N_STEPS_LIST:
            eval_iterator = loop_dataloader(eval_loader)
            total_sum_loss = torch.tensor(0.0, device=device)
            total_n_examples = 0

            for eval_batch_idx in range(N_EVAL_BATCHES):
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

                # Seed per (n_steps, batch_idx) so all models get same init
                source_seed = n_steps * 10000 + eval_batch_idx

                sum_loss, n_examples = pgd_recon_loss_simulated_ddp(
                    model=model,
                    sub_batches=sub_batches,
                    sub_cis=sub_cis,
                    weight_deltas=weight_deltas if config.use_delta_component else None,
                    sub_target_outs=sub_target_outs,
                    n_steps=n_steps,
                    step_size=0.1,
                    source_seed=source_seed,
                )
                total_sum_loss += sum_loss
                total_n_examples += n_examples

                batch_avg = (total_sum_loss / total_n_examples).item()
                print(
                    f"  n_steps={n_steps:>3d} | batch {eval_batch_idx + 1:>2d}/{N_EVAL_BATCHES}"
                    f" | running avg: {batch_avg:.4f}"
                )

                del sub_batches, sub_cis, sub_target_outs, weight_deltas
                torch.cuda.empty_cache()

            final_loss = (total_sum_loss / total_n_examples).item()
            model_results[n_steps] = final_loss
            print(f">>> n_steps={n_steps:>3d} | FINAL: {final_loss:.6f}")

        all_results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'n_steps':>8s}"
    for model_name in all_results:
        header += f"  | {model_name:>27s}"
    print(header)
    print("-" * len(header))
    for n_steps in N_STEPS_LIST:
        row = f"{n_steps:>8d}"
        for model_name in all_results:
            row += f"  | {all_results[model_name][n_steps]:>27.4f}"
        print(row)


if __name__ == "__main__":
    main()

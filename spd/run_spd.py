"""Run SPD on a model."""

import gc
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import wandb
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.data import loop_dataloader
from spd.eval import evaluate
from spd.log import logger
from spd.losses import calculate_losses
from spd.models.component_model import ComponentModel
from spd.utils.alive_components_tracker import AliveComponentsTracker
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import (
    avg_eval_metrics_across_ranks,
    avg_metrics_across_ranks,
    get_world_size,
    is_distributed,
    is_main_process,
    sync_across_processes,
)
from spd.utils.general_utils import (
    extract_batch_data,
    get_linear_annealed_p,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)
from spd.utils.run_utils import save_file


def local_log(data: Mapping[str, float | Image.Image], step: int, out_dir: Path) -> None:
    metrics_file = out_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics_without_images = {}
    for k, v in data.items():
        if isinstance(v, Image.Image):
            v.save(fig_dir / f"{k.replace('/', '_')}_{step}.png")
        else:
            metrics_without_images[k] = v

    with open(metrics_file, "a") as f:
        f.write(json.dumps({"step": step, **metrics_without_images}) + "\n")


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    out_dir: Path | None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    target_model.requires_grad_(False)
    model = ComponentModel(
        target_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        gate_type=config.gate_type,
        gate_hidden_dims=config.gate_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    model.to(device)

    # Wrap model with DDP if distributed
    world_size = get_world_size()
    wrapped_model: nn.Module = model
    if world_size > 1:
        if device.startswith("cuda"):
            # Parse device string to get device id for GPU
            device_id = int(device.split(":")[1]) if ":" in device else 0
            wrapped_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device_id],
                output_device=device_id,
            )
        else:
            # For CPU, don't pass device_ids or output_device
            wrapped_model = torch.nn.parallel.DistributedDataParallel(model)
        # Access the underlying module for component operations
        component_model = wrapped_model.module  # type: ignore[attr-defined]
    else:
        component_model = model

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            tgt = component_model.components_or_modules[tgt_name].components
            src = component_model.components_or_modules[src_name].components
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in component_model.components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(component_model.gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = sum(
        component.original.weight.numel()
        for component in component_model.components_or_modules.values()
    )

    # Track which components are alive based on firing frequency
    alive_tracker = AliveComponentsTracker(
        module_names=component_model.target_module_paths,
        C=config.C,
        n_examples_until_dead=config.n_examples_until_dead,
        device=device,
        ci_alive_threshold=config.ci_alive_threshold,
    )

    for step in tqdm(range(config.steps + 1), ncols=0):
        optimizer.zero_grad()

        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        microbatch_log_data: defaultdict[str, float] = defaultdict(float)
        current_p = config.pnorm  # Initialize with default value
        for _ in range(config.gradient_accumulation_steps):
            batch = extract_batch_data(next(train_iterator)).to(device)

            target_out, pre_weight_acts = wrapped_model(
                batch,
                mode="pre_forward_cache",
                module_names=component_model.target_module_paths,
            )
            # NOTE: pre_weight_acts are now part of the DDP computation graph, so when they pass
            # through the parameters in calc_causal_importances below, the DDP hook will get called
            # and gradients will be properly synced across ranks on the next backward pass.
            causal_importances, causal_importances_upper_leaky = (
                component_model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                )
            )

            alive_tracker.watch_batch(causal_importances)

            # Calculate current p value with annealing
            current_p = get_linear_annealed_p(
                step=step,
                steps=config.steps,
                initial_p=config.pnorm,
                p_anneal_start_frac=config.p_anneal_start_frac,
                p_anneal_final_p=config.p_anneal_final_p,
                p_anneal_end_frac=config.p_anneal_end_frac,
            )

            microbatch_total_loss, microbatch_loss_terms = calculate_losses(
                model=component_model,
                batch=batch,
                config=config,
                causal_importances=causal_importances,
                causal_importances_upper_leaky=causal_importances_upper_leaky,
                target_out=target_out,
                device=device,
                n_params=n_params,
                current_p=current_p,
            )
            microbatch_total_loss.div_(config.gradient_accumulation_steps).backward()

            for loss_name, loss_value in microbatch_loss_terms.items():
                microbatch_log_data[f"train/loss/{loss_name}"] += (
                    loss_value / config.gradient_accumulation_steps
                )

            for layer_name, layer_ci in causal_importances.items():
                l0_val = calc_ci_l_zero(layer_ci, config.ci_alive_threshold)
                microbatch_log_data[f"train/{layer_name}/l0"] += (
                    l0_val / config.gradient_accumulation_steps
                )

        # --- Train Logging --- #
        if step % config.train_log_freq == 0:
            if is_distributed():
                avg_metrics = avg_metrics_across_ranks(microbatch_log_data, device=device)
                microbatch_log_data = cast(defaultdict[str, float], avg_metrics)

            # Already reduced alive counts across ranks, so no need to reduce again
            for layer_name, n_alive_count in alive_tracker.n_alive().items():
                n_alive_key = f"train/{layer_name}/n_alive_{alive_tracker.ci_alive_threshold}"
                microbatch_log_data[n_alive_key] = n_alive_count

            grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
            for param in component_params + gate_params:
                if param.grad is not None:
                    grad_norm += param.grad.data.flatten().pow(2).sum()
            microbatch_log_data["train/misc/grad_norm"] = grad_norm.sqrt().item()
            microbatch_log_data["train/misc/lr"] = step_lr
            microbatch_log_data["train/misc/current_p"] = current_p

            if is_main_process():
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                for name, value in microbatch_log_data.items():
                    tqdm.write(f"{name}: {value:.15f}")
                if out_dir is not None:
                    local_log(microbatch_log_data, step, out_dir)
                if config.wandb_project:
                    wandb.log(microbatch_log_data, step=step)

        # --- Evaluation --- #
        if step % config.eval_freq == 0:
            with torch.inference_mode():
                run_slow: bool = (
                    config.slow_eval_on_first_step
                    if step == 0
                    else step % config.slow_eval_freq == 0
                )

                metrics = evaluate(
                    model=component_model,  # No backward passes so DDP wrapped_model not needed
                    eval_iterator=eval_iterator,
                    device=device,
                    config=config,
                    run_slow=run_slow,
                    n_steps=n_eval_steps,
                )

                if is_distributed():
                    metrics = avg_eval_metrics_across_ranks(metrics, device=device)

                if is_main_process():
                    for k, v in metrics.items():
                        tqdm.write(f"eval/{k}: {v}")
                    if out_dir is not None:
                        local_log(metrics, step, out_dir)
                    if config.wandb_project:
                        wandb_logs: dict[str, int | float | str | wandb.Image] = {
                            f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                            for k, v in metrics.items()
                        }
                        wandb.log(wandb_logs, step=step)

                del metrics
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        if (
            (
                (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
                or step == config.steps
            )
            and out_dir is not None
            and is_main_process()
        ):
            # Save the state dict of the underlying module (not DDP wrapper)
            save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            sync_across_processes()
            optimizer.step()

    if is_main_process():
        logger.info("Finished training loop.")

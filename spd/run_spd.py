"""Run SPD on a model."""

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

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
from spd.eval import avg_eval_metrics_across_ranks, evaluate
from spd.identity_insertion import insert_identity_operations_
from spd.log import logger
from spd.losses import compute_total_loss
from spd.metrics import faithfulness_loss
from spd.metrics.alive_components import AliveComponentsTracker
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import (
    avg_metrics_across_ranks,
    get_world_size,
    is_distributed,
    is_main_process,
    sync_across_processes,
)
from spd.utils.general_utils import (
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)
from spd.utils.module_utils import replace_std_values_in_layernorm
from spd.utils.run_utils import save_file


def run_faithfulness_warmup(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: Config,
) -> None:
    """Run faithfulness warmup phase to improve initialization.

    Args:
        component_model: The component model to warm up
        component_params: List of component parameters to optimize
        config: Configuration object containing warmup settings
    """

    logger.info("Starting faithfulness warmup phase...")

    assert component_params, "component_params is empty"

    faithfulness_warmup_optimizer = optim.AdamW(
        component_params,
        lr=config.faithfulness_warmup_lr,
        weight_decay=config.faithfulness_warmup_weight_decay,
    )

    for faithfulness_warmup_step in range(config.faithfulness_warmup_steps):
        faithfulness_warmup_optimizer.zero_grad()
        weight_deltas = component_model.calc_weight_deltas()
        loss = faithfulness_loss(weight_deltas)
        loss.backward()
        faithfulness_warmup_optimizer.step()

        if (
            faithfulness_warmup_step % 100 == 0
            or faithfulness_warmup_step == config.faithfulness_warmup_steps - 1
        ):
            logger.info(
                f"Faithfulness warmup step {faithfulness_warmup_step + 1} / {config.faithfulness_warmup_steps}; Faithfulness loss: {loss.item():.9f}"
            )
    del faithfulness_warmup_optimizer
    torch.cuda.empty_cache()
    gc.collect()


def local_log(data: dict[str, Any], step: int, out_dir: Path) -> None:
    metrics_file = out_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics_without_images = {}
    for k, v in data.items():
        if isinstance(v, Image.Image):
            filename = f"{k.replace('/', '_')}_{step}.png"
            v.save(fig_dir / filename)
            tqdm.write(f"Saved figure {k} to {fig_dir / filename}")
        elif isinstance(v, wandb.plot.CustomChart):
            json_path = fig_dir / f"{k.replace('/', '_')}_{step}.json"
            payload = {"columns": list(v.table.columns), "data": list(v.table.data), "step": step}
            with open(json_path, "w") as f:
                json.dump(payload, f, default=str)
            tqdm.write(f"Saved custom chart data {k} to {json_path}")
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
    ln_stds: dict[str, float] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    target_model.requires_grad_(False)

    model = ComponentModel(
        target_model=target_model,
        target_module_patterns=config.all_module_patterns,
        C=config.C,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    if ln_stds is not None:
        # model has ablated layernorms, patch in the fixed std values
        replace_std_values_in_layernorm(model, ln_stds)
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
            tgt = component_model.components[tgt_name]
            src = component_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    ci_fn_params: list[torch.nn.Parameter] = []
    for name, component in component_model.components.items():
        component_params.extend(list(component.parameters()))
        ci_fn_params.extend(list(component_model.ci_fns[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + ci_fn_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    # Track which components are alive based on firing frequency
    alive_tracker = AliveComponentsTracker(
        module_paths=model.module_paths,
        C=config.C,
        device=device,
        n_examples_until_dead=config.n_examples_until_dead,
        ci_alive_threshold=config.ci_alive_threshold,
        global_n_examples_per_batch=extract_batch_data(next(train_iterator)).shape[:-1].numel(),
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

        for _ in range(config.gradient_accumulation_steps):
            weight_deltas = component_model.calc_weight_deltas()
            batch = extract_batch_data(next(train_iterator)).to(device)

            # NOTE: we need to call the wrapped_model at least once each step in order to setup
            # the DDP gradient syncing for all parameters in the component model. Gradients will
            # sync regardless of whether the parameters are used in this call to wrapped_model.
            target_model_output: OutputWithCache = wrapped_model(batch, cache_type="input")

            causal_importances, causal_importances_upper_leaky = (
                component_model.calc_causal_importances(
                    pre_weight_acts=target_model_output.cache,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                    sampling=config.sampling,
                )
            )

            alive_tracker.update(ci=causal_importances)

            microbatch_total_loss, microbatch_loss_terms = compute_total_loss(
                loss_metric_configs=config.loss_metric_configs,
                model=component_model,
                batch=batch,
                ci=causal_importances,
                ci_upper_leaky=causal_importances_upper_leaky,
                target_out=target_model_output.output,
                weight_deltas=weight_deltas,
                pre_weight_acts=target_model_output.cache,
                current_frac_of_training=step / config.steps,
                sampling=config.sampling,
                use_delta_component=config.use_delta_component,
                n_mask_samples=config.n_mask_samples,
                output_loss_type=config.output_loss_type,
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

            alive_counts = alive_tracker.compute()
            for metric_name, n_alive_count in alive_counts.items():
                n_alive_key = f"train/{metric_name}_{alive_tracker.ci_alive_threshold}"
                microbatch_log_data[n_alive_key] = n_alive_count

            grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
            for param in component_params + ci_fn_params:
                if param.grad is not None:
                    grad_norm += param.grad.data.flatten().pow(2).sum()
            microbatch_log_data["train/misc/grad_norm"] = grad_norm.sqrt().item()
            microbatch_log_data["train/misc/lr"] = step_lr

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
            with torch.no_grad():
                slow_step: bool = (
                    config.slow_eval_on_first_step
                    if step == 0
                    else step % config.slow_eval_freq == 0
                )

                metrics = evaluate(
                    metric_configs=config.eval_metric_configs + config.loss_metric_configs,
                    model=component_model,  # No backward passes so DDP wrapped_model not needed
                    eval_iterator=eval_iterator,
                    device=device,
                    run_config=config,
                    slow_step=slow_step,
                    n_eval_steps=n_eval_steps,
                    current_frac_of_training=step / config.steps,
                )

                if is_distributed():
                    metrics = avg_eval_metrics_across_ranks(metrics, device=device)

                if is_main_process():
                    for k, v in metrics.items():
                        tqdm.write(f"eval/{k}: {v}")
                    if out_dir is not None:
                        local_log(metrics, step, out_dir)
                    if config.wandb_project:
                        wandb_logs = {
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

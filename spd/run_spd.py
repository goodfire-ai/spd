"""Run SPD on a model."""

import gc
import os
from collections import defaultdict
from collections.abc import Iterator
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.base_config import BaseConfig
from spd.configs import (
    Config,
    LossMetricConfigType,
    MetricConfigType,
    PGDMultiBatchConfig,
    PGDMultiBatchReconLossConfig,
    PGDMultiBatchReconSubsetLossConfig,
)
from spd.data import loop_dataloader
from spd.eval import evaluate, evaluate_multibatch_pgd
from spd.identity_insertion import insert_identity_operations_
from spd.log import logger
from spd.losses import compute_total_loss
from spd.metrics import faithfulness_loss
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.settings import SPD_OUT_DIR
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import (
    avg_metrics_across_ranks,
    broadcast_model_params,
    get_distributed_state,
    is_main_process,
    seed_per_rank,
    sync_across_processes,
)
from spd.utils.general_utils import (
    bf16_autocast,
    dict_safe_update_,
    extract_batch_data,
    get_scheduled_value,
    save_pre_run_info,
)
from spd.utils.logging_utils import get_grad_norms_dict, local_log
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import generate_run_id, save_file
from spd.utils.wandb_utils import init_wandb, try_wandb


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
    # TODO: we should reverse the order of these two calls
    torch.cuda.empty_cache()
    gc.collect()


def get_unique_metric_configs(
    loss_configs: list[LossMetricConfigType], eval_configs: list[MetricConfigType]
) -> list[MetricConfigType]:
    """If a metric appears in both loss and eval configs, only include the eval version."""
    eval_config_names = [type(cfg).__name__ for cfg in eval_configs]
    eval_metric_configs = eval_configs[:]
    for cfg in loss_configs:
        if type(cfg).__name__ not in eval_config_names:
            eval_metric_configs.append(cfg)
        else:
            logger.warning(
                f"{type(cfg).__name__} is in both loss and eval configs, only including eval config"
            )
    return eval_metric_configs


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

    def create_pgd_data_iter() -> (
        Iterator[Int[Tensor, "..."]] | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]
    ):
        assert hasattr(train_loader, "generator") and train_loader.generator is not None
        train_loader.generator.manual_seed(config.seed)
        return iter(train_loader)

    if is_main_process():
        logger.info(f"Train+eval logs saved to directory: {out_dir}")

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    target_model.requires_grad_(False)

    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )

    model.to(device)
    broadcast_model_params(model)

    # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
    seed_per_rank(config.seed)

    # Wrap model with DDP if distributed
    dist_state = get_distributed_state()
    wrapped_model: nn.Module = model
    if dist_state is not None:
        if dist_state.backend == "nccl":
            device_id = dist_state.local_rank
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
    assert isinstance(component_model, ComponentModel), "component_model is not a ComponentModel"

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
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())
        ci_fn_params.extend(component_model.ci_fns[name].parameters())

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimized_params = component_params + ci_fn_params
    optimizer = optim.AdamW(optimized_params, lr=config.lr_schedule.start_val, weight_decay=0)

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    eval_metric_configs = get_unique_metric_configs(
        loss_configs=config.loss_metric_configs, eval_configs=config.eval_metric_configs
    )

    multibatch_pgd_eval_configs: list[
        PGDMultiBatchReconLossConfig | PGDMultiBatchReconSubsetLossConfig
    ] = [cfg for cfg in eval_metric_configs if isinstance(cfg, PGDMultiBatchConfig)]

    eval_metric_configs = [
        cfg for cfg in eval_metric_configs if cfg not in multibatch_pgd_eval_configs
    ]
    batch_dims: tuple[int, ...] | None = None

    sample_batch = extract_batch_data(next(train_iterator))
    batch_dims = (
        sample_batch.shape[:-1]
        if config.output_loss_type == "mse"  # if mse then input is a vector
        else sample_batch.shape  # else it's a batch of token ids
    )

    for step in tqdm(range(config.steps + 1), ncols=0, disable=not is_main_process()):
        optimizer.zero_grad()

        step_lr = get_scheduled_value(
            step=step, total_steps=config.steps, config=config.lr_schedule
        )
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        weight_deltas = component_model.calc_weight_deltas()

        step_log_data: defaultdict[str, float] = defaultdict(float)

        batch = extract_batch_data(next(train_iterator)).to(device)

        with bf16_autocast(enabled=config.autocast_bf16):
            # NOTE: we need to call the wrapped_model at least once each step in order
            # to setup the DDP gradient syncing for all parameters in the component model.
            # Gradients will sync regardless of whether the parameters are used in this
            # call to wrapped_model.
            target_model_output: OutputWithCache = wrapped_model(batch, cache_type="input")

            ci = component_model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )

            total_loss, loss_terms = compute_total_loss(
                loss_metric_configs=config.loss_metric_configs,
                model=component_model,
                batch=batch,
                ci=ci,
                target_out=target_model_output.output,
                weight_deltas=weight_deltas,
                pre_weight_acts=target_model_output.cache,
                current_frac_of_training=step / config.steps,
                sampling=config.sampling,
                use_delta_component=config.use_delta_component,
                n_mask_samples=config.n_mask_samples,
                output_loss_type=config.output_loss_type,
            )

        total_loss.backward()

        for loss_name, loss_value in loss_terms.items():
            step_log_data[f"train/{loss_name}"] += loss_value

        for layer_name, layer_ci in ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, config.ci_alive_threshold)
            step_log_data[f"train/l0/{layer_name}"] += l0_val

        # --- Train Logging --- #
        if step % config.train_log_freq == 0:
            avg_metrics = avg_metrics_across_ranks(step_log_data, device=device)
            step_log_data = cast(defaultdict[str, float], avg_metrics)

            grad_norms = get_grad_norms_dict(component_model, device)
            dict_safe_update_(
                step_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )

            step_log_data["train/schedules/lr"] = step_lr

            if is_main_process():
                assert out_dir is not None
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                for name, value in step_log_data.items():
                    tqdm.write(f"{name}: {value:.15f}")
                local_log(step_log_data, step, out_dir)
                if config.wandb_project:
                    try_wandb(wandb.log, step_log_data, step=step)

        # --- Evaluation --- #
        if step % config.eval_freq == 0:
            with torch.no_grad():
                slow_step: bool = (
                    config.slow_eval_on_first_step
                    if step == 0
                    else step % config.slow_eval_freq == 0
                )

                assert batch_dims is not None, "batch_dims is not set"
                multibatch_pgd_metrics = evaluate_multibatch_pgd(
                    multibatch_pgd_eval_configs=multibatch_pgd_eval_configs,
                    model=component_model,
                    create_data_iter=create_pgd_data_iter,
                    config=config,
                    batch_dims=batch_dims,
                    device=device,
                )

                metrics = evaluate(
                    eval_metric_configs=eval_metric_configs,
                    model=component_model,  # No backward passes so DDP wrapped_model not needed
                    eval_iterator=eval_iterator,
                    device=device,
                    run_config=config,
                    slow_step=slow_step,
                    n_eval_steps=n_eval_steps,
                    current_frac_of_training=step / config.steps,
                )

                dict_safe_update_(metrics, multibatch_pgd_metrics)

                if is_main_process():
                    assert out_dir is not None
                    for k, v in metrics.items():
                        tqdm.write(f"eval/{k}: {v}")
                    local_log(metrics, step, out_dir)
                    if config.wandb_project:
                        wandb_logs = {
                            f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                            for k, v in metrics.items()
                        }
                        try_wandb(wandb.log, wandb_logs, step=step)

                del metrics
                # TODO: we should reverse the order of these two calls
                torch.cuda.empty_cache()
                gc.collect()

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and is_main_process():
            assert out_dir is not None
            # Save the state dict of the underlying module (not DDP wrapper)
            save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                try_wandb(
                    wandb.save,
                    str(out_dir / f"model_{step}.pth"),
                    base_path=str(out_dir),
                    policy="now",
                )

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            sync_across_processes()
            if config.grad_clip_norm_components is not None:
                clip_grad_norm_(component_params, config.grad_clip_norm_components)
            if config.grad_clip_norm_ci_fns is not None:
                clip_grad_norm_(ci_fn_params, config.grad_clip_norm_ci_fns)
            optimizer.step()

    if is_main_process():
        logger.info("Finished training loop.")


def run_experiment(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    experiment_tag: str,
    run_id: str | None = None,
    launch_id: str | None = None,
    evals_id: str | None = None,
    sweep_params: dict[str, Any] | None = None,
    target_model_train_config: BaseConfig | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run a full SPD experiment: setup, optimize, cleanup.

    All ranks call this function. Only the main process does wandb/logging setup.
    """
    if is_main_process():
        run_id = run_id or generate_run_id("spd")
        out_dir = SPD_OUT_DIR / "spd" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {out_dir}")

        tags = [str(i) for i in [experiment_tag, evals_id, launch_id] if i is not None]
        slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        if slurm_array_job_id is not None:
            tags.append(f"slurm-array-job-id_{slurm_array_job_id}")

        if config.wandb_project:
            init_wandb(config, config.wandb_project, run_id, config.wandb_run_name, tags)

        logger.info(config)

        save_pre_run_info(
            save_to_wandb=config.wandb_project is not None,
            out_dir=out_dir,
            spd_config=config,
            sweep_params=sweep_params,
            target_model=target_model if target_model_train_config is not None else None,
            train_config=target_model_train_config,
            task_name=getattr(config.task_config, "task_name", None),
        )
    else:
        out_dir = None

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    if is_main_process() and config.wandb_project:
        wandb.finish()

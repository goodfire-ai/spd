"""Run SPD on a model."""

import gc
from collections import defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
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
from torch.nn.utils import clip_grad_norm_
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

from spd.configs import (
    Config,
    LossMetricConfigType,
    MetricConfigType,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDMultiBatchConfig,
    PGDMultiBatchReconLossConfig,
    PGDMultiBatchReconSubsetLossConfig,
)
from spd.data import loop_dataloader
from spd.eval import evaluate, evaluate_multibatch_pgd
from spd.identity_insertion import insert_identity_operations_
from spd.log import logger
from spd.losses import compute_losses
from spd.metrics import faithfulness_loss
from spd.metrics.alive_components import AliveComponentsTracker
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.persistent_pgd import PersistentPGDState
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    sync_across_processes,
)
from spd.utils.general_utils import (
    dict_safe_update_,
    extract_batch_data,
    get_scheduled_value,
)
from spd.utils.logging_utils import get_grad_norms_dict, local_log
from spd.utils.module_utils import expand_module_patterns, replace_std_values_in_layernorm
from spd.utils.run_utils import save_file
from spd.utils.wandb_utils import try_wandb


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


def get_training_ctx(config: Config) -> AbstractContextManager[profile | None]:
    if not config.profiling:
        return nullcontext()

    profiler_schedule = schedule(
        wait=config.profiling.wait_steps,
        warmup=config.profiling.warmup_steps,
        active=config.profiling.active_steps,
        repeat=1,
    )
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(config.profiling.trace_dir),
        record_shapes=True,
        with_stack=True,
    )


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
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    if ln_stds is not None:
        # model has ablated layernorms, patch in the fixed std values
        replace_std_values_in_layernorm(model, ln_stds)
    model.to(device)

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
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())

    ci_fn_params = list(component_model.ci_fn.parameters())

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimized_params = component_params + ci_fn_params
    optimizer = optim.AdamW(optimized_params, lr=config.lr_schedule.start_val, weight_decay=0)

    logger.info(f"LR scheduler: {config.lr_schedule.fn_type}")

    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(component_model, component_params, config)

    loss_configs = config.loss_metric_configs
    ppgd_loss_configs = [
        cfg
        for cfg in loss_configs
        if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
    ]

    eval_metric_configs = get_unique_metric_configs(
        loss_configs=loss_configs, eval_configs=config.eval_metric_configs
    )

    multibatch_pgd_eval_configs: list[
        PGDMultiBatchReconLossConfig | PGDMultiBatchReconSubsetLossConfig
    ] = [cfg for cfg in eval_metric_configs if isinstance(cfg, PGDMultiBatchConfig)]

    eval_metric_configs = [
        cfg for cfg in eval_metric_configs if cfg not in multibatch_pgd_eval_configs
    ]
    batch_dims: tuple[int, ...] | None = None

    # Track which components are alive based on firing frequency
    sample_batch = extract_batch_data(next(train_iterator))
    batch_dims = (
        sample_batch.shape[:-1]
        if config.output_loss_type == "mse"  # if mse then input is a vector
        else sample_batch.shape  # else it's a batch of token ids
    )
    alive_tracker = AliveComponentsTracker(
        module_to_c=model.module_to_c,
        device=device,
        n_examples_until_dead=config.n_examples_until_dead,
        ci_alive_threshold=config.ci_alive_threshold,
        global_n_examples_per_batch=batch_dims.numel(),
    )

    # Initialize PersistentPGD states if needed
    ppgd_states: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
    ] = {
        ppgd_cfg: PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=batch_dims,
            device=device,
            use_delta_component=config.use_delta_component,
            cfg=ppgd_cfg,
        )
        for ppgd_cfg in ppgd_loss_configs
    }

    with get_training_ctx(config) as maybe_profiler:
        for step in range(config.steps + 1):
            optimizer.zero_grad()

            step_lr = get_scheduled_value(
                step=step, total_steps=config.steps, config=config.lr_schedule
            )
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            weight_deltas = component_model.calc_weight_deltas()

            batch_log_data: defaultdict[str, float] = defaultdict(float)

            ppgd_batch_grads = {
                ppgd_cfg: ppgd_states[ppgd_cfg].empty_grads() for ppgd_cfg in ppgd_loss_configs
            }

            for _ in range(config.gradient_accumulation_steps):
                mb = extract_batch_data(next(train_iterator)).to(device)

                # NOTE: we need to call the wrapped_model at least once each step in order to setup
                # the DDP gradient syncing for all parameters in the component model. Gradients will
                # sync regardless of whether the parameters are used in this call to wrapped_model.
                with record_function("forward_pass"):
                    target_model_output: OutputWithCache = wrapped_model(mb, cache_type="input")

                with record_function("calc_causal_importances"):
                    ci = component_model.calc_causal_importances(
                        pre_weight_acts=target_model_output.cache,
                        detach_inputs=False,
                        sampling=config.sampling,
                    )

                alive_tracker.update(ci=ci.lower_leaky)
                # for state in ppgd_states.values():
                #     for mask in state.masks.values():
                #         assert mask.grad is None
                #         mask.requires_grad_(True)

                with record_function("compute_total_loss"):
                    mb_losses = compute_losses(
                        loss_metric_configs=loss_configs,
                        model=component_model,
                        batch=mb,
                        ci=ci,
                        target_out=target_model_output.output,
                        weight_deltas=weight_deltas,
                        pre_weight_acts=target_model_output.cache,
                        current_frac_of_training=step / config.steps,
                        sampling=config.sampling,
                        use_delta_component=config.use_delta_component,
                        n_mask_samples=config.n_mask_samples,
                        ppgd_maskss={cfg: state.masks for cfg, state in ppgd_states.items()},
                        output_loss_type=config.output_loss_type,
                    )

                for ppgd_cfg in ppgd_loss_configs:
                    # TODO abstract this out into a function on state object
                    # TODO figure out whether we actually need this
                    mb_loss = mb_losses[ppgd_cfg]
                    state = ppgd_states[ppgd_cfg]
                    batch_grads = ppgd_batch_grads[ppgd_cfg]
                    mb_grads = state.get_grads(mb_loss)

                    for module_name, grad in mb_grads.items():
                        batch_grads[module_name] += grad / config.gradient_accumulation_steps

                    # # TODO figure out whether we actually need this
                    # for mask in state.masks.values():
                    #     assert mask.grad is None
                    #     mask.requires_grad_(False)

                mb_total_loss = torch.tensor(0.0, device=device)
                for loss_cfg, loss_tensor in mb_losses.items():
                    assert loss_cfg.coeff is not None, "Loss config must have a coeff"
                    mb_total_loss += loss_cfg.coeff * loss_tensor
                    batch_log_data[f"train/loss/{loss_cfg.classname}"] += (
                        loss_tensor.item() / config.gradient_accumulation_steps
                    )

                with record_function("backward"):
                    mb_total_loss.div_(config.gradient_accumulation_steps).backward()

                for layer_name, layer_ci in ci.lower_leaky.items():
                    l0_val = calc_ci_l_zero(layer_ci, config.ci_alive_threshold)
                    batch_log_data[f"train/l0/{layer_name}"] += (
                        l0_val / config.gradient_accumulation_steps
                    )

            # do the PPGD step using the accumulated gradients
            for ppgd_cfg in ppgd_loss_configs:
                state = ppgd_states[ppgd_cfg]
                batch_grads = ppgd_batch_grads[ppgd_cfg]
                mean_abs_steps = state.step(batch_grads, step)
                for module_name, mean_abs_step in mean_abs_steps.items():
                    batch_log_data[f"train/ppgd/mean_abs_step/{module_name}"] = mean_abs_step

            # --- Train Logging --- #
            if step % config.train_log_freq == 0:
                avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
                batch_log_data = cast(defaultdict[str, float], avg_metrics)

                alive_counts = alive_tracker.compute()
                for target_module_path, n_alive_count in alive_counts.items():
                    n_alive_key = (
                        f"train/n_alive/t{alive_tracker.ci_alive_threshold}_{target_module_path}"
                    )
                    batch_log_data[n_alive_key] = n_alive_count

                grad_norms = get_grad_norms_dict(component_model, device)
                dict_safe_update_(
                    batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
                )

                batch_log_data["train/schedules/lr"] = step_lr

                if is_main_process():
                    assert out_dir is not None
                    logger.info(f"--- Step {step} ---")
                    logger.info(f"LR: {step_lr:.6f}")
                    for name, value in batch_log_data.items():
                        logger.info(f"{name}: {value:.15f}")
                    local_log(batch_log_data, step, out_dir)
                    if config.wandb_project:
                        try_wandb(wandb.log, batch_log_data, step=step)

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
                        # ppgd_maskss={cfg: state.masks for cfg, state in ppgd_states.items()},
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
                            logger.info(f"eval/{k}: {v}")
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

                if maybe_profiler is not None:
                    maybe_profiler.step()

    if is_main_process():
        logger.info("Finished training loop.")

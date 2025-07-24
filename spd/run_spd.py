"""Run SPD on a model."""

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.figures import create_figures
from spd.log import logger
from spd.losses import calculate_losses
from spd.metrics import create_metrics
from spd.models.component_model import ComponentModel
from spd.utils.alive_components_tracker import AliveComponentsTracker
from spd.utils.general_utils import (
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)
from spd.utils.run_utils import save_file


def loop_dl[T](dl: DataLoader[T]):
    dl_iter = iter(dl)
    while True:
        try:
            yield next(dl_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            dl_iter = iter(dl)
            yield next(dl_iter)


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

    train_iterator = loop_dl(train_loader)
    eval_iterator = loop_dl(eval_loader)

    logger.info(f"Output directory: {out_dir}")
    metrics_file = out_dir / "metrics.jsonl" if out_dir is not None else None

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

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            tgt = model.components_or_modules[tgt_name].components
            src = model.components_or_modules[src_name].components
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in model.components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(model.gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = sum(
        component.original.weight.numel() for component in model.components_or_modules.values()
    )

    # Track which components are alive based on firing frequency
    alive_tracker = AliveComponentsTracker(
        module_names=model.target_module_paths,
        C=config.C,
        n_examples_until_dead=config.n_examples_until_dead,
        device=torch.device(device),
        ci_alive_threshold=config.ci_alive_threshold,
    )

    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        optimizer.zero_grad()

        mb_log_data = defaultdict[str, float](float)

        for _ in range(config.gradient_accumulation_steps):
            batch = extract_batch_data(next(train_iterator)).to(device)

            target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                batch, module_names=model.target_module_paths
            )

            causal_importances, causal_importances_upper_leaky = model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sigmoid_type=config.sigmoid_type,
                detach_inputs=False,
            )

            alive_tracker.watch_batch(causal_importances)

            micro_total_loss, micro_loss_terms = calculate_losses(
                model=model,
                batch=batch,
                config=config,
                causal_importances=causal_importances,
                causal_importances_upper_leaky=causal_importances_upper_leaky,
                target_out=target_out,
                device=device,
                n_params=n_params,
            )

            micro_total_loss.div_(config.gradient_accumulation_steps).backward()

            for loss_name, loss_value in micro_loss_terms.items():
                mb_log_data[f"train/loss/{loss_name}"] += (
                    loss_value / config.gradient_accumulation_steps
                )

            for layer_name, ci in causal_importances.items():
                l0 = (ci > config.ci_alive_threshold).float().sum(-1).mean().item()
                mb_log_data[f"train/{layer_name}/l0"] += l0 / config.gradient_accumulation_steps

        if step % config.train_log_freq == 0:
            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR: {step_lr:.6f}")
            for name, value in mb_log_data.items():
                tqdm.write(f"{name}: {value:.7f}")

            if config.wandb_project:

                for layer_name, n_alive_count in alive_tracker.n_alive().items():
                    mb_log_data[
                        f"train/{layer_name}/n_alive_{alive_tracker.ci_alive_threshold}"
                    ] = n_alive_count

                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in component_params + gate_params:
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()

                mb_log_data["train/misc/grad_norm"] = grad_norm.sqrt().item()

                mb_log_data["train/misc/lr"] = step_lr

                wandb.log(mb_log_data, step=step)

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.eval_freq == 0:
                metrics = create_metrics(
                    model=model,
                    eval_iterator=eval_iterator,
                    n_eval_steps=n_eval_steps,
                    device=device,
                    config=config,
                )

                if metrics_file is not None:
                    # Filter out non-JSON-serializable objects (like wandb.Table) for file logging
                    file_metrics = {
                        k: v for k, v in metrics.items() if not isinstance(v, wandb.Table)
                    }
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps(file_metrics) + "\n")

                if config.wandb_project:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")

                fig_dict = create_figures(
                    model=model,
                    eval_iterator=eval_iterator,
                    device=device,
                    config=config,
                    n_eval_steps=n_eval_steps,
                )

                if config.wandb_project:
                    wandb.log(
                        {f"figures/{k}": wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        fig_dir = out_dir / "figures"
                        for k, v in fig_dict.items():
                            save_file(v, fig_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {fig_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            save_file(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            optimizer.step()

    logger.info("Finished training loop.")

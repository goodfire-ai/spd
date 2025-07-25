"""Run SPD on a model."""

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.eval import EvalMetricValue, eval
from spd.log import logger
from spd.losses import calculate_losses
from spd.models.component_model import ComponentModel
from spd.models.components import VectorGateMLPs
from spd.utils.alive_components_tracker import AliveComponentsTracker
from spd.utils.component_utils import ci_l_zero
from spd.utils.general_utils import (
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)
from spd.utils.run_utils import save_file


def local_log(data: Mapping[str, EvalMetricValue], step: int, out_dir: Path) -> None:
    metrics_file = out_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    for k, v in data.items():
        metrics_dict = {}

        if isinstance(v, Image.Image):
            v.save(fig_dir / f"{k.replace('/', '_')}_{step}.png")
        else:
            metrics_dict[k] = v

        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")


def get_target_module_mean_input_norms(
    model: ComponentModel,
    target_module_patterns: list[str],
    train_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    n_batches: int,
) -> dict[str, float]:
    input_norms = defaultdict[str, list[float]](list)
    for _ in tqdm(total=n_batches, desc="Computing target module mean input norms"):
        batch = extract_batch_data(next(train_iterator)).to(device)
        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=target_module_patterns
        )
        for name, act in pre_weight_acts.items():
            norm = act.norm(dim=-1).mean().item()
            input_norms[name].append(norm)
    return {name: sum(norms) / len(norms) for name, norms in input_norms.items()}


def loop_dataloader[T](dl: DataLoader[T]):
    dl_iter = iter(dl)
    while True:
        try:
            yield next(dl_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            dl_iter = iter(dl)
            yield next(dl_iter)


def get_component_model(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    tied_weights: list[tuple[str, str]] | None = None,
) -> ComponentModel:
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

    if config.init_gates_from_mean_input_norms:
        target_module_mean_input_norms = get_target_module_mean_input_norms(
            model=model,
            target_module_patterns=config.target_module_patterns,
            train_iterator=train_iterator,
            device=device,
            n_batches=10,
        )

        for module_name, gates in model.gates.items():
            assert isinstance(gates, VectorGateMLPs), (
                "norm-based gate initialization is only supported for vector_mlp gates"
            )
            gates.init_weights_from_mean_input_norm_(target_module_mean_input_norms[module_name])

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            tgt = model.components_or_modules[tgt_name].components
            src = model.components_or_modules[src_name].components
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    return model


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

    logger.info(f"Output directory: {out_dir}")

    model = get_component_model(
        target_model=target_model,
        config=config,
        device=device,
        train_iterator=train_iterator,
        tied_weights=tied_weights,
    )

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

        microbatch_log_data = defaultdict[str, float](float)
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

            microbatch_total_loss, microbatch_loss_terms = calculate_losses(
                model=model,
                batch=batch,
                config=config,
                causal_importances=causal_importances,
                causal_importances_upper_leaky=causal_importances_upper_leaky,
                target_out=target_out,
                device=device,
                n_params=n_params,
            )

            microbatch_total_loss.div_(config.gradient_accumulation_steps).backward()

            for loss_name, loss_value in microbatch_loss_terms.items():
                microbatch_log_data[f"train/loss/{loss_name}"] += (
                    loss_value / config.gradient_accumulation_steps
                )

            for layer_name, layer_ci in causal_importances.items():
                l0_val = ci_l_zero(layer_ci, config.ci_alive_threshold)
                microbatch_log_data[f"train/{layer_name}/l0"] += (
                    l0_val / config.gradient_accumulation_steps
                )

        # --- Train Logging --- #
        if step % config.train_log_freq == 0:
            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR: {step_lr:.6f}")
            for name, value in microbatch_log_data.items():
                tqdm.write(f"{name}: {value:.7f}")

            for layer_name, n_alive_count in alive_tracker.n_alive().items():
                n_alive_key = f"train/{layer_name}/n_alive_{alive_tracker.ci_alive_threshold}"
                microbatch_log_data[n_alive_key] = n_alive_count

            grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
            for param in component_params + gate_params:
                if param.grad is not None:
                    grad_norm += param.grad.data.flatten().pow(2).sum()
            microbatch_log_data["train/misc/grad_norm"] = grad_norm.sqrt().item()

            microbatch_log_data["train/misc/lr"] = step_lr

            if out_dir is not None:
                local_log(microbatch_log_data, step, out_dir)
            if config.wandb_project:
                wandb.log(microbatch_log_data, step=step)

        # --- Evaluation --- #
        if step % config.eval_freq == 0:
            with torch.inference_mode():
                run_slow = step % config.slow_eval_freq == 0

                metrics = eval(
                    model=model,
                    eval_iterator=eval_iterator,
                    device=device,
                    config=config,
                    run_slow=run_slow,
                    n_steps=n_eval_steps,
                )

                if out_dir is not None:
                    local_log(metrics, step, out_dir)
                if config.wandb_project:
                    wandb_logs: dict[str, int | float | str | wandb.Image] = {
                        f"eval/{k}": wandb.Image(v) if isinstance(v, Image.Image) else v
                        for k, v in metrics.items()
                    }
                    wandb.log(wandb_logs, step=step)

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

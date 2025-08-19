"""Run SPD on a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Union

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.data_utils import SparseFeatureDataset
from spd.log import logger
from spd.losses import calc_ce_losses, calc_kl_divergence_lm, calculate_losses
from spd.models.component_model import ComponentModel, init_As_and_Bs_
from spd.models.component_utils import (
    calc_causal_importances,
    calc_ci_l_zero,
    component_activation_statistics,
)
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import (
    create_embed_ci_sample_table,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
)
from spd.utils import extract_batch_data, get_lr_schedule_fn, get_lr_with_warmup

wandb.require("core")


def get_common_run_name_suffix(config: Config) -> str:
    """Common code for setting WandB run names."""
    run_suffix = f"C{config.C}"
    run_suffix += f"_imp{config.importance_minimality_coeff:.0e}"
    if config.n_ci_mlp_neurons > 0:
        run_suffix += f"_mlp{config.n_ci_mlp_neurons}"
    if config.stochastic_recon_coeff:
        run_suffix += f"_srec{config.stochastic_recon_coeff:.0e}"
    if config.stochastic_recon_layerwise_coeff:
        run_suffix += f"_srclay{config.stochastic_recon_layerwise_coeff:.0e}"
    if config.lr_exponential_halflife is not None:
        run_suffix += f"_hlf{config.lr_exponential_halflife:.0e}"
    run_suffix += f"_bs{config.batch_size}"
    run_suffix += f"_pnorm{config.pnorm}"
    return run_suffix


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: Union[
        DataLoader[Int[Tensor, "..."]], 
        DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]
    ],
    eval_loader: Union[
        DataLoader[Int[Tensor, "..."]], 
        DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]
    ],
    n_eval_steps: int,
    out_dir: Union[Path, None],
    plot_results_fn: Union[Callable[..., dict[str, plt.Figure]], None] = None,
    tied_weights: Union[list[tuple[str, str]], None] = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model parameters frozen.")

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    model.to(device)
    init_As_and_Bs_(model=model, components=components)

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            components[tgt_name].B.data = components[src_name].A.data.T
            components[tgt_name].A.data = components[src_name].B.data.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = sum(model.model.get_parameter(n + ".weight").numel() for n in components)

    log_data = {}
    data_iter = iter(train_loader)

    alive_components: dict[str, Bool[Tensor, " C"]] = {
        layer_name: torch.zeros(config.C, device=device).bool() for layer_name in components
    }

    # Iterate one extra step for final logging/plotting/saving
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
        log_data["lr"] = step_lr

        optimizer.zero_grad()

        try:
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(train_loader)
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        batch = batch.to(device)

        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: components[module_name].A for module_name in components}

        causal_importances, causal_importances_upper_leaky = calc_causal_importances(
            pre_weight_acts=pre_weight_acts, As=As, gates=gates, detach_inputs=False
        )

        for layer_name, ci in causal_importances.items():
            alive_components[layer_name] = alive_components[layer_name] | (ci > 0.1).any(dim=(0, 1))

        total_loss, loss_terms = calculate_losses(
            model=model,
            batch=batch,
            config=config,
            components=components,
            causal_importances=causal_importances,
            causal_importances_upper_leaky=causal_importances_upper_leaky,
            target_out=target_out,
            device=device,
            n_params=n_params,
        )

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.print_freq == 0:
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
                for name, value in loss_terms.items():
                    tqdm.write(f"{name}: {value:.7f}")

                if step > 0:
                    for layer_name, layer_alive_components in alive_components.items():
                        log_data[f"{layer_name}/n_alive_01"] = layer_alive_components.sum().item()
                        alive_components[layer_name] = torch.zeros(config.C, device=device).bool()

                # Calculate component logits and KL losses
                masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=causal_importances
                )
                unmasked_component_logits = model.forward_with_components(
                    batch, components=components, masks=None
                )

                target_logits = model(batch)

                log_data["misc/unmasked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=unmasked_component_logits, target=target_logits
                ).item()
                log_data["misc/masked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                ).item()

                if config.log_ce_losses:
                    ce_losses = calc_ce_losses(
                        model=model,
                        batch=batch,
                        components=components,
                        masks=causal_importances,
                        unmasked_component_logits=unmasked_component_logits,
                        masked_component_logits=masked_component_logits,
                        target_logits=target_logits,
                    )
                    log_data.update(ce_losses)

                embed_ci_table = create_embed_ci_sample_table(causal_importances)
                if embed_ci_table is not None:
                    log_data["misc/embed_ci_sample"] = embed_ci_table

                if config.wandb_project:
                    ci_l_zero = calc_ci_l_zero(causal_importances=causal_importances)
                    for layer_name, layer_ci_l_zero in ci_l_zero.items():
                        log_data[f"{layer_name}/ci_l0"] = layer_ci_l_zero
                    wandb.log(log_data, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")
                fig_dict = {}
                if plot_results_fn is not None:
                    fig_dict = plot_results_fn(
                        model=model,
                        components=components,
                        gates=gates,
                        batch_shape=batch.shape,
                        device=device,
                    )

                ci_histogram_figs = plot_ci_histograms(causal_importances=causal_importances)
                fig_dict.update(ci_histogram_figs)

                mean_component_activation_counts = component_activation_statistics(
                    model=model, dataloader=eval_loader, n_steps=n_eval_steps, device=device
                )[1]
                assert mean_component_activation_counts is not None
                fig_dict["mean_component_activation_counts"] = (
                    plot_mean_component_activation_counts(
                        mean_component_activation_counts=mean_component_activation_counts,
                    )
                )

                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        for k, v in fig_dict.items():
                            v.savefig(out_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")
                wandb.save(
                    str(out_dir / f"optimizer_{step}.pth"), base_path=str(out_dir), policy="now"
                )

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()  # type: ignore
                grad_norm_val = grad_norm.sqrt().item()
                wandb.log({"grad_norm": grad_norm_val}, step=step)

            optimizer.step()

    logger.info("Finished training loop.")

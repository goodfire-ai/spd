from functools import partial
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader

from spd.configs import (
    Config,
    PGDConfig,
    PGDGlobalConfig,
    PGDGlobalConfigType,
    PGDGlobalReconLossConfig,
    PGDGlobalReconSubsetLossConfig,
    PGDInitStrategy,
    SamplingType,
)
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.utils.component_utils import RoutingType, sample_uniform_k_subset_routing_masks
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm, extract_batch_data


def _compute_pgd_objective_loss(
    adv_sources: Float[Tensor, "n_layers *batch_dims C2"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: dict[str, Float[Tensor, "*batch_dims"]] | Literal["all"],
    target_out: Float[Tensor, "... vocab"],
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
) -> Float[Tensor, ""]:
    """Compute reconstruction loss for given adversarial sources.

    Args:
        adv_sources: Adversarial source tensor with shape [n_layers, *batch_dims, C2]
        batch: Input batch
        ci: Causal importance scores per layer
        weight_deltas: Optional weight delta tensors per layer
        routing_masks: Routing masks (either "all" or dict of masks)
        target_out: Target model output for comparison
        model: The ComponentModel to evaluate
        output_loss_type: Loss type ("mse" or "kl")

    Returns:
        Total reconstruction loss (summed over all examples)
    """
    adv_sources_components: Float[Tensor, "n_layers *batch_dims C"]
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], adv_sources[i, ..., -1]) for i, k in enumerate(weight_deltas)
            }
            adv_sources_components = adv_sources[..., :-1]

    mask_infos = make_mask_infos(
        component_masks=_interpolate_component_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )
    out = model(batch, mask_infos=mask_infos)
    sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    return sum_loss


def _get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
    adv_sources_components: Float[Tensor, "n_layers *batch_dims C"],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    """Set the mask value to ci + (1 - ci) * adv_sources_components.

    NOTE: This is not ideal. Suppose ci is 0.2 and adv_sources_components is 0.8. Then the mask
    value will be 0.2 + (1 - 0.2) * 0.8 = 0.84. It would make more sense to set the mask value to
    0.8 directly, since this is the value output by PGD. If we wanted to instead use the more
    natural setup of mask = max(ci, adv_sources_components), we would need to work out how to
    maintain the gradient flow. We feel that this likely isn't a big problem as it is right now
    since the change would just give PGD more optimization power, and we already get a very bad
    loss value for it.
    """
    assert torch.all(adv_sources_components <= 1.0) and torch.all(adv_sources_components >= 0.0)
    assert adv_sources_components.shape[0] == len(ci)
    assert all(ci[k].shape[-1] == adv_sources_components.shape[-1] for k in ci)
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for i, module_name in enumerate(ci):
        scaled_noise_to_add = (1 - ci[module_name]) * adv_sources_components[i]
        component_masks[module_name] = ci[module_name] + scaled_noise_to_add
    return component_masks


def pgd_masked_recon_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Central implementation of PGD masked reconstruction loss.

    Optimizes adversarial stochastic masks and optionally weight deltas for the given objective function.
    """
    *batch_dims, C = next(iter(ci.values())).shape
    n_layers = len(ci)
    # C2 represents the total number of components including the optional weight delta
    C2 = C if weight_deltas is None else C + 1

    match routing:
        case "all":
            routing_masks = "all"
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=tuple(batch_dims),
                module_names=list(ci.keys()),
                device=batch.device,
            )

    # We create a single adv_sources tensor and index into it for each layer
    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            adv_source_shape = torch.Size([n_layers, *batch_dims, C2])
        case "shared_across_batch":
            adv_source_shape = torch.Size([n_layers] + [1 for _ in batch_dims] + [C2])

    adv_sources: Float[Tensor, "n_layers *batch_dims C2"] | Float[Tensor, "n_layers *1 C2"] = (
        _get_pgd_init_tensor(pgd_config.init, adv_source_shape, batch.device).requires_grad_(True)
    )

    objective_fn = partial(
        _compute_pgd_objective_loss,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        target_out=target_out,
        model=model,
        output_loss_type=output_loss_type,
    )

    # PGD ascent
    for _ in range(pgd_config.n_steps):
        assert adv_sources.grad is None
        with torch.enable_grad():
            obj = objective_fn(adv_sources=adv_sources.expand(n_layers, *batch_dims, C2))

        adv_sources_grads = torch.autograd.grad(obj, adv_sources)
        assert len(adv_sources_grads) == 1
        reduced_adv_sources_grads = all_reduce(adv_sources_grads[0], op=ReduceOp.SUM)
        with torch.no_grad():
            adv_sources.add_(pgd_config.step_size * reduced_adv_sources_grads.sign())
            adv_sources.clamp_(0.0, 1.0)

    final_loss = objective_fn(adv_sources=adv_sources.expand(n_layers, *batch_dims, C2))

    n_examples = (
        target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
    )

    return final_loss, n_examples


def _global_pgd_step(
    adv_sources: Float[Tensor, "n_layers *batch_dim_or_ones C2"],
    pgd_config: PGDGlobalConfig,
    model: ComponentModel,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: torch.device | str,
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    sampling: SamplingType,
    batch_dims: tuple[int, ...],
) -> tuple[Float[Tensor, ""], int]:
    """Perform a single PGD step with gradient accumulation over multiple batches.

    Returns:
        - The total loss for the PGD step (only used for the final step)
        - The number of examples used in the PGD step
    Also updates adv_sources in place.
    """
    assert adv_sources.grad is None
    pgd_step_accum_loss = torch.tensor(0.0, device=device)
    pgd_step_accum_grads = torch.zeros_like(adv_sources)

    dataloader_iter = iter(dataloader)
    n_examples = 0

    # Accumulate gradients over n_batches batches
    for batch_idx in range(pgd_config.gradient_accumulation_steps):
        try:
            batch_item = next(dataloader_iter)
        except StopIteration:
            logger.warning(f"Dataloader exhausted after {batch_idx} batches, ending PGD step.")
            break
        batch = extract_batch_data(batch_item).to(device)

        # TODO: Verify that we want no_grad. Using this in training?
        with torch.no_grad():
            batch_target_model_output: OutputWithCache = model(batch, cache_type="input")
            batch_ci = model.calc_causal_importances(
                pre_weight_acts=batch_target_model_output.cache,
                detach_inputs=True,
                sampling=sampling,
            ).lower_leaky

        match routing:
            case "all":
                batch_routing_masks = "all"
            case "uniform_k-stochastic":
                batch_routing_masks = sample_uniform_k_subset_routing_masks(
                    mask_shape=tuple(batch_dims),
                    module_names=list(batch_ci.keys()),
                    device=batch.device,
                )

        with torch.enable_grad():
            total_loss = _compute_pgd_objective_loss(
                adv_sources=adv_sources.expand(-1, *batch_dims, -1),
                batch=batch,
                ci=batch_ci,
                weight_deltas=weight_deltas,
                routing_masks=batch_routing_masks,
                target_out=batch_target_model_output.output,
                model=model,
                output_loss_type=output_loss_type,
            )
        batch_grads = torch.autograd.grad(total_loss, adv_sources)

        assert len(batch_grads) == 1
        pgd_step_accum_grads += batch_grads[0].detach()
        pgd_step_accum_loss += total_loss.detach()
        n_examples += (
            batch_target_model_output.output.shape.numel()
            if output_loss_type == "mse"
            else batch_target_model_output.output.shape[:-1].numel()
        )

        del batch_target_model_output, batch_ci

    reduced_accumulated_grads = all_reduce(pgd_step_accum_grads, op=ReduceOp.SUM)
    with torch.no_grad():
        adv_sources.add_(pgd_config.step_size * reduced_accumulated_grads.sign())
        adv_sources.clamp_(0.0, 1.0)

    return pgd_step_accum_loss, n_examples


def calc_pgd_global_masked_recon_loss(
    pgd_config: PGDGlobalConfig,
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    sampling: SamplingType,
    use_delta_component: bool,
    batch_dims: tuple[int, ...],
    device: str,
) -> float:
    """PGD masked reconstruction loss with gradient accumulation over multiple batches.

    This function optimizes adversarial masks by accumulating gradients over pgd_config.n_batches
    batches before each PGD update step.

    Args:
        pgd_config: PGD global configuration
        model: The ComponentModel to evaluate
        dataloader: DataLoader or iterator yielding batches
        output_loss_type: Loss type for reconstruction ("mse" or "kl")
        routing: Routing strategy ("all" or "uniform_k-stochastic")
        sampling: Sampling mode for causal importance calculation
        use_delta_component: Whether to include weight delta component
        batch_dims: Dimensions of batch (e.g., (batch_size,) or (batch_size, seq_len))
    Returns:
        Final reconstruction loss after PGD optimization
    """

    # C2 represents the total number of components including the optional weight delta
    C2 = model.C if not use_delta_component else model.C + 1
    n_layers = len(model.target_module_paths)
    adv_source_shape = torch.Size([n_layers] + [1 for _ in batch_dims] + [C2])
    weight_deltas = model.calc_weight_deltas() if use_delta_component else None

    adv_sources: Float[Tensor, "n_layers *batch_dim_or_ones C2"] = _get_pgd_init_tensor(
        init=pgd_config.init, shape=adv_source_shape, device=device
    ).requires_grad_(True)

    step = partial(
        _global_pgd_step,
        adv_sources=adv_sources,
        pgd_config=pgd_config,
        model=model,
        weight_deltas=weight_deltas,
        dataloader=dataloader,
        device=device,
        output_loss_type=output_loss_type,
        routing=routing,
        sampling=sampling,
        batch_dims=batch_dims,
    )

    for _ in range(pgd_config.n_steps):
        step()

    pgd_step_loss, n_examples = step()
    final_loss_summed = all_reduce(pgd_step_loss, op=ReduceOp.SUM)
    final_n_examples = all_reduce(torch.tensor(n_examples, device=device), op=ReduceOp.SUM)
    final_loss = final_loss_summed / final_n_examples
    return final_loss.item()


def calc_global_pgd_metrics(
    global_pgd_eval_configs: list[PGDGlobalConfigType],
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    config: Config,
    batch_dims: tuple[int, ...],
    device: str,
) -> dict[str, float]:
    """Calculate global PGD metrics."""
    metrics: dict[str, float] = {}
    for pgd_global_config in global_pgd_eval_configs:
        match pgd_global_config:
            case PGDGlobalReconLossConfig():
                routing = "all"
            case PGDGlobalReconSubsetLossConfig():
                routing = "uniform_k-stochastic"

        assert pgd_global_config.classname not in metrics, (
            f"Metric {pgd_global_config.classname} already exists"
        )

        metrics[pgd_global_config.classname] = calc_pgd_global_masked_recon_loss(
            pgd_config=pgd_global_config,
            model=model,
            dataloader=dataloader,
            output_loss_type=config.output_loss_type,
            routing=routing,
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            batch_dims=batch_dims,
            device=device,
        )
    return metrics

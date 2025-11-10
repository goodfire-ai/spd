from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, PGDInitStrategy, PGDMultiBatchConfig, SamplingType
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import RoutingMasks, make_mask_infos
from spd.utils.component_utils import RoutingType, sample_uniform_k_subset_routing_masks
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm, extract_batch_data


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
    C = model.C
    batch_dims = next(iter(ci.values())).shape[:-1]
    n_layers = len(ci)
    # C2 represents the total number of components including the optional weight delta
    C2 = C if weight_deltas is None else C + 1

    match routing:
        case "all":
            routing_masks = "all"
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=batch_dims,
                module_names=model.target_module_paths,
                device=batch.device,
            )

    # We create a single adv_sources tensor and index into it for each layer
    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            adv_source_shape = torch.Size([n_layers, *batch_dims, C2])
        case "shared_across_batch":
            singleton_batch_dims = [1 for _ in batch_dims]
            adv_source_shape = torch.Size([n_layers, *singleton_batch_dims, C2])

    adv_sources: Float[Tensor, "n_layers *batch_dims C2"] | Float[Tensor, "n_layers *1 C2"] = (
        _get_pgd_init_tensor(pgd_config.init, adv_source_shape, batch.device).requires_grad_(True)
    )

    fwd_bwd_fn = partial(
        _pgd_fwd_bwd,
        adv_sources=adv_sources,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        target_out=target_out,
        model=model,
        output_loss_type=output_loss_type,
        batch_dims=batch_dims,
    )

    for _ in range(pgd_config.n_steps):
        assert adv_sources.grad is None
        _, _, adv_sources_grads = fwd_bwd_fn()
        with torch.no_grad():
            adv_sources.add_(pgd_config.step_size * adv_sources_grads.sign())
            adv_sources.clamp_(0.0, 1.0)

    sum_loss, total_n_examples, _ = fwd_bwd_fn()
    return sum_loss, total_n_examples


CreateDataIter = Callable[
    [],
    Iterator[Int[Tensor, "..."]] | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
]


def calc_multibatch_pgd_masked_recon_loss(
    pgd_config: PGDMultiBatchConfig,
    model: ComponentModel,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    create_data_iter: CreateDataIter,
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    sampling: SamplingType,
    use_delta_component: bool,
    batch_dims: tuple[int, ...],
    device: str,
) -> Float[Tensor, ""]:
    """PGD masked reconstruction loss with gradient accumulation over multiple batches.

    This function optimizes adversarial masks by accumulating gradients over pgd_config.n_batches
    batches before each PGD update step.

    Args:
        pgd_config: Multibatch PGD configuration
        model: The ComponentModel to evaluate
        create_data_iter: Function to create an iterator over batches. This function should return
            an iterator which behaves identically each time. Specifically in terms of data ordering
            and shuffling.
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
    singleton_batch_dims = [1 for _ in batch_dims]
    adv_source_shape = torch.Size([n_layers] + singleton_batch_dims + [C2])

    adv_sources: Float[Tensor, "n_layers *ones C2"] = _get_pgd_init_tensor(
        init=pgd_config.init, shape=adv_source_shape, device=device
    ).requires_grad_(True)

    def get_routing_masks() -> RoutingMasks:
        match routing:
            case "all":
                return "all"
            case "uniform_k-stochastic":
                return sample_uniform_k_subset_routing_masks(
                    mask_shape=batch_dims,
                    module_names=model.target_module_paths,
                    device=device,
                )

    fwd_bwd_fn = partial(
        _multibatch_pgd_fwd_bwd,
        adv_sources=adv_sources,
        pgd_config=pgd_config,
        model=model,
        weight_deltas=weight_deltas,
        device=device,
        output_loss_type=output_loss_type,
        sampling=sampling,
        get_routing_masks=get_routing_masks,
        batch_dims=torch.Size(batch_dims),
    )

    for _ in range(pgd_config.n_steps):
        assert adv_sources.grad is None
        _, _, adv_sources_grads = fwd_bwd_fn(data_iter=create_data_iter())
        with torch.no_grad():
            adv_sources.add_(pgd_config.step_size * adv_sources_grads.sign())
            adv_sources.clamp_(0.0, 1.0)

    final_loss, final_n_examples, _ = fwd_bwd_fn(data_iter=create_data_iter())
    return final_loss / final_n_examples


def _pgd_fwd_bwd(
    adv_sources: Float[Tensor, "n_layers *batch_dim_or_ones C2"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    target_out: Float[Tensor, "... vocab"],
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch_dims: tuple[int, ...],
) -> tuple[Float[Tensor, ""], int, Float[Tensor, "n_layers *batch_dims C2"]]:
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
        batch_dims: Dimensions of batch (e.g., (batch_size,) or (batch_size, seq_len))

    Returns:
        Total reconstruction loss (summed over all examples)
    """

    C2 = model.C if weight_deltas is None else model.C + 1
    n_layers = len(model.target_module_paths)

    with torch.enable_grad():
        expanded_adv_sources = adv_sources.expand(n_layers, *batch_dims, C2)
        adv_sources_components: Float[Tensor, "n_layers *batch_dims C"]
        match weight_deltas:
            case None:
                weight_deltas_and_masks = None
                adv_sources_components = expanded_adv_sources
            case dict():
                weight_deltas_and_masks = {
                    k: (weight_deltas[k], expanded_adv_sources[i, ..., -1])
                    for i, k in enumerate(weight_deltas)
                }
                adv_sources_components = expanded_adv_sources[..., :-1]

        mask_infos = make_mask_infos(
            component_masks=_interpolate_component_mask(ci, adv_sources_components),
            weight_deltas_and_masks=weight_deltas_and_masks,
            routing_masks=routing_masks,
        )
        out = model(batch, mask_infos=mask_infos)
        sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)

    # important: take gradient wrt the unexpanded adv_sources, not the expanded ones
    (adv_sources_grads,) = torch.autograd.grad(sum_loss, adv_sources)
    adv_sources_grads = all_reduce(adv_sources_grads, op=ReduceOp.SUM)

    sum_loss = all_reduce(sum_loss, op=ReduceOp.SUM)
    n_examples = (
        target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
    )
    n_examples = int(
        all_reduce(torch.tensor(n_examples, device=batch.device), op=ReduceOp.SUM).item()
    )

    return sum_loss, n_examples, adv_sources_grads


def _multibatch_pgd_fwd_bwd(
    adv_sources: Float[Tensor, "n_layers *ones C2"],
    pgd_config: PGDMultiBatchConfig,
    model: ComponentModel,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    data_iter: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: torch.device | str,
    output_loss_type: Literal["mse", "kl"],
    get_routing_masks: Callable[[], RoutingMasks],
    sampling: SamplingType,
    batch_dims: tuple[int, ...],
) -> tuple[Float[Tensor, ""], int, Float[Tensor, "n_layers *batch_dim_or_ones C2"]]:
    """Perform a forward and backward pass over multiple batches with gradient accumulation.

    Returns:
        - The total loss for the PGD step (only used for the final step)
        - The number of examples used in the PGD step
        - The gradients of the adv_sources
    """
    pgd_step_accum_sum_loss = torch.tensor(0.0, device=device)
    pgd_step_accum_n_examples = 0
    pgd_step_accum_grads = torch.zeros_like(adv_sources)

    for microbatch_idx in range(pgd_config.gradient_accumulation_steps):
        try:
            microbatch_item = next(data_iter)
        except StopIteration:
            logger.warning(f"Dataloader exhausted after {microbatch_idx} batches, ending PGD step.")
            break
        microbatch = extract_batch_data(microbatch_item).to(device)

        # NOTE: technically this is duplicated work across PGD steps, but that's the price we pay to
        # enable accumulating gradients over more microbatches than we'd be able to fit CI values in
        # memory for. In other words, you can't fit 100,000 microbatches worth of CI values in memory.
        target_model_output: OutputWithCache = model(microbatch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_model_output.cache,
            sampling=sampling,
        ).lower_leaky

        # It's important that we call this every microbatch to ensure stochastic routing masks are
        # sampled independently for each example.
        routing_masks = get_routing_masks()

        batch_sum_loss, batch_n_examples, batch_grads = _pgd_fwd_bwd(
            adv_sources=adv_sources,
            batch=microbatch,
            ci=ci,
            weight_deltas=weight_deltas,
            routing_masks=routing_masks,
            target_out=target_model_output.output,
            model=model,
            output_loss_type=output_loss_type,
            batch_dims=batch_dims,
        )

        pgd_step_accum_grads += batch_grads.detach()
        pgd_step_accum_sum_loss += batch_sum_loss
        pgd_step_accum_n_examples += batch_n_examples

        del target_model_output, ci

    return pgd_step_accum_sum_loss, pgd_step_accum_n_examples, pgd_step_accum_grads


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

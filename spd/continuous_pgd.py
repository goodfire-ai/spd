"""Continuous PGD: Persistent adversarial masks that evolve across training steps.

Instead of reinitializing PGD masks each training step and running N optimization steps,
ContinuousPGD maintains persistent masks that receive one gradient update per training step.
Over many steps, these masks converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.
"""

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.models.component_model import ComponentModel
from spd.models.components import RoutingMasks, make_mask_infos
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


class ContinuousPGDState:
    """Persistent state for continuous PGD optimization.

    Holds adversarial masks that persist across training steps. Each position in the batch
    gets its own mask that evolves over time.

    Shape: {module_name: (batch_size, C)} per module, broadcasting along sequence dimension.
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        batch_size: int,
        device: torch.device | str,
        use_delta_component: bool,
    ) -> None:
        self.module_to_c = module_to_c
        self.batch_size = batch_size
        self.device = device
        self.use_delta_component = use_delta_component

        # Initialize masks randomly in [0, 1]
        # Shape: (batch_size, mask_c) per module - one mask per sequence in batch
        self.masks: dict[str, Float[Tensor, "batch mask_c"]] = {}
        for module_name, module_c in module_to_c.items():
            mask_c = module_c + 1 if use_delta_component else module_c
            self.masks[module_name] = torch.rand(
                batch_size, mask_c, device=device, requires_grad=True
            )

    def step(
        self,
        grads: dict[str, Float[Tensor, "batch mask_c"]],
        step_size: float,
    ) -> None:
        """Perform one PGD update step using the provided gradients.

        Updates masks in-place: mask += step_size * sign(grad), then clamps to [0, 1].
        """
        with torch.no_grad():
            for module_name in self.masks:
                self.masks[module_name].add_(step_size * grads[module_name].sign())
                self.masks[module_name].clamp_(0.0, 1.0)

        # Re-enable gradients for next step
        for mask in self.masks.values():
            mask.requires_grad_(True)

    def get_expanded_masks(
        self, batch_dims: tuple[int, ...]
    ) -> dict[str, Float[Tensor, "*batch_dims mask_c"]]:
        """Expand masks to full batch dimensions, broadcasting along sequence.

        Args:
            batch_dims: Target batch dimensions, e.g. (batch_size, seq_len)

        Returns:
            Masks expanded to (*batch_dims, mask_c) via broadcasting
        """
        # batch_dims is (batch, seq) for LMs
        # Our masks are (batch, mask_c)
        # We need to expand to (batch, seq, mask_c) by broadcasting
        assert batch_dims[0] == self.batch_size, (
            f"Batch size mismatch: state has {self.batch_size}, got {batch_dims[0]}"
        )

        expanded: dict[str, Float[Tensor, "*batch_dims mask_c"]] = {}
        for module_name, mask in self.masks.items():
            # mask is (batch, mask_c), we want (batch, seq, mask_c)
            # Insert singleton dimensions for seq dims
            view_shape = [self.batch_size] + [1] * (len(batch_dims) - 1) + [mask.shape[-1]]
            expanded[module_name] = mask.view(*view_shape).expand(*batch_dims, -1)

        return expanded


def continuous_pgd_recon_loss(
    model: ComponentModel,
    batch: torch.Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    pgd_state: ContinuousPGDState,
    step_size: float,
) -> Float[Tensor, ""]:
    """Compute reconstruction loss with continuous PGD masks and update state.

    Unlike standard PGD which runs N steps per training step, this:
    1. Uses persistent masks from pgd_state
    2. Computes forward pass and loss
    3. Computes gradients w.r.t. masks
    4. Updates masks with one PGD step
    5. Returns the loss

    The masks persist across training steps, accumulating adversarial pressure over time.

    Args:
        model: ComponentModel to evaluate
        batch: Input batch
        ci: Causal importance values per module
        weight_deltas: Optional weight deltas for delta component
        target_out: Target model output
        output_loss_type: "mse" or "kl"
        pgd_state: Persistent PGD state holding masks
        step_size: PGD step size for mask updates

    Returns:
        Reconstruction loss (scalar tensor)
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
    router = AllLayersRouter()
    routing_masks: RoutingMasks = router.get_masks(
        module_names=model.target_module_paths, mask_shape=batch_dims
    )

    # Get expanded masks from state
    expanded_adv_sources = pgd_state.get_expanded_masks(batch_dims)

    # Split into component masks and weight delta masks
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]]
    weight_deltas_and_masks: (
        dict[str, tuple[Float[Tensor, "d_out d_in"], Float[Tensor, ...]]] | None
    )
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = expanded_adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas
            }
            adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}

    # Interpolate CI with adversarial masks: mask = ci + (1 - ci) * adv
    component_masks = _interpolate_component_mask(ci, adv_sources_components)

    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )

    # Forward pass
    with torch.enable_grad():
        out = model(batch, mask_infos=mask_infos)
        sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
        n_examples = (
            target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
        )
        loss = sum_loss / n_examples

    # Compute gradients w.r.t. the unexpanded masks in state
    grads = torch.autograd.grad(loss, list(pgd_state.masks.values()))
    grads_dict = {
        k: all_reduce(g, op=ReduceOp.SUM)
        for k, g in zip(pgd_state.masks.keys(), grads, strict=True)
    }

    # Update state with one PGD step
    pgd_state.step(grads_dict, step_size)

    return loss


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    """Interpolate CI with adversarial masks: final = ci + (1 - ci) * adv."""
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for module_name in ci:
        adv_source = adv_sources_components[module_name]
        scaled_noise = (1 - ci[module_name]) * adv_source
        component_masks[module_name] = ci[module_name] + scaled_noise
    return component_masks

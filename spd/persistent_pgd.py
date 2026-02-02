"""Persistent PGD: Persistent adversarial masks that evolve across training steps.

Instead of reinitializing PGD masks each training step and running N optimization steps,
PersistentPGD maintains persistent masks that receive one gradient update per training step.
Over many steps, these masks converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDOptimizerConfig,
)
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, RoutingMasks, make_mask_infos
from spd.routing import AllLayersRouter, Router, get_subset_router
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm

PPGD_INIT_SEED = 42


class PersistentPGDState:
    """Persistent state for persistent PGD optimization.

    Holds a single adversarial mask per module that persists across training steps.
    The mask is shared across all batch elements and ranks.

    Shape: {module_name: (C,)} per module, broadcast to batch dims during forward.
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        device: torch.device | str,
        use_delta_component: bool,
        optimizer_cfg: PGDOptimizerConfig,
    ) -> None:
        self.module_to_c = module_to_c
        self.device = device
        self.optimizer_cfg = optimizer_cfg

        self._adam_step = 0
        self._adam_m: dict[str, Float[Tensor, " mask_c"]] = {}
        self._adam_v: dict[str, Float[Tensor, " mask_c"]] = {}

        # Initialize masks randomly in [0, 1] with fixed seed for consistency across ranks
        # Shape: (mask_c,) per module - single mask shared across all batch elements
        rng = torch.Generator(device=device).manual_seed(PPGD_INIT_SEED)
        self.masks: dict[str, Float[Tensor, " mask_c"]] = {}
        for module_name, module_c in module_to_c.items():
            mask_c = module_c + 1 if use_delta_component else module_c
            self.masks[module_name] = torch.rand(mask_c, device=device, generator=rng)
            if optimizer_cfg.type == "adam":
                self._adam_m[module_name] = torch.zeros_like(self.masks[module_name])
                self._adam_v[module_name] = torch.zeros_like(self.masks[module_name])

    def step(self, grads: dict[str, Float[Tensor, " mask_c"]]) -> None:
        """Perform one PGD update step using the provided gradients.

        Updates masks in-place, then clamps to [0, 1].
        """
        with torch.no_grad():
            if self.optimizer_cfg.type == "sign":
                cfg = self.optimizer_cfg
                for module_name in self.masks:
                    self.masks[module_name].add_(cfg.step_size * grads[module_name].sign())
            elif self.optimizer_cfg.type == "adam":
                cfg = self.optimizer_cfg
                self._adam_step += 1
                bias_correction1 = 1 - cfg.beta1**self._adam_step
                bias_correction2 = 1 - cfg.beta2**self._adam_step
                for module_name, mask in self.masks.items():
                    grad = grads[module_name]
                    m = self._adam_m[module_name]
                    v = self._adam_v[module_name]
                    m.mul_(cfg.beta1).add_(grad, alpha=1 - cfg.beta1)
                    v.mul_(cfg.beta2).addcmul_(grad, grad, value=1 - cfg.beta2)
                    m_hat = m / bias_correction1
                    v_hat = v / bias_correction2
                    denom = v_hat.sqrt().add_(cfg.eps)
                    mask.addcdiv_(m_hat, denom, value=cfg.lr)
            else:
                raise ValueError(f"Unknown PersistentPGD optimizer: {self.optimizer_cfg.type}")

            for mask in self.masks.values():
                mask.clamp_(0.0, 1.0)

    def empty_grads(self) -> dict[str, Float[Tensor, " mask_c"]]:
        """Empty the gradients of the masks."""
        return {module_name: torch.zeros_like(mask) for module_name, mask in self.masks.items()}

    @contextmanager
    def requires_grad(self) -> Generator[None]:
        """Set the requires_grad flag for the masks."""
        for mask in self.masks.values():
            mask.requires_grad_(True)
        yield
        for mask in self.masks.values():
            mask.requires_grad_(False)


def get_mask_infos(
    model: ComponentModel,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    masks: dict[str, Float[Tensor, "*batch_dims mask_c"]],
    router: Router,
) -> dict[str, ComponentsMaskInfo]:
    """Get mask infos for persistent PGD."""

    batch_dims = next(iter(ci.values())).shape[:-1]
    routing_masks: RoutingMasks = router.get_masks(
        module_names=model.target_module_paths, mask_shape=batch_dims
    )

    # Get expanded masks from state
    expanded_adv_sources: dict[str, Float[Tensor, "*batch_dims mask_c"]] = {}
    for module_name, mask in masks.items():
        # mask is (mask_c,), expand to (*batch_dims, mask_c)
        # Use contiguous() to create a new tensor (not a view) while preserving gradients.
        # This is needed because we update masks in-place later, which would invalidate views.
        expanded_adv_sources[module_name] = mask.expand(*batch_dims, -1).contiguous()

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

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def persistent_pgd_recon_loss(
    model: ComponentModel,
    batch: torch.Tensor,
    ppgd_cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
    ppgd_masks: dict[str, Float[Tensor, " mask_c"]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, " mask_c"]]]:
    """Compute reconstruction loss with persistent PGD masks.

    Unlike standard PGD which runs N steps per training step, this:
    1. Uses persistent masks from pgd_state
    2. Computes forward pass and loss
    3. Computes gradients w.r.t. masks
    4. Returns the loss, having updated the masks in-place

    Args:
        ppgd_cfg: PersistentPGD config
        model: ComponentModel to evaluate
        batch: Input batch
        ci: Causal importance values per module
        weight_deltas: Optional weight deltas for delta component
        target_out: Target model output
        output_loss_type: "mse" or "kl"
        pgd_state: Persistent PGD state holding masks

    Returns:
        PersistentPGDResult containing loss and deferred gradients
    """

    match ppgd_cfg:
        case PersistentPGDReconLossConfig():
            router = AllLayersRouter()
        case PersistentPGDReconSubsetLossConfig(routing=routing):
            router = get_subset_router(routing, batch.device)

    mask_infos = get_mask_infos(model, ci, weight_deltas, ppgd_masks, router)

    out = model(batch, mask_infos=mask_infos)

    sum_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
    n_examples = (
        target_out.shape.numel() if output_loss_type == "mse" else target_out.shape[:-1].numel()
    )
    loss = sum_loss / n_examples

    grads = torch.autograd.grad(loss, list(ppgd_masks.values()), retain_graph=True)
    grads_dict = {
        k: all_reduce(g, op=ReduceOp.SUM) for k, g in zip(ppgd_masks.keys(), grads, strict=True)
    }
    return loss, grads_dict


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

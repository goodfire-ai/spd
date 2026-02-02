"""Persistent PGD: Persistent adversarial masks that evolve across training steps.

Instead of reinitializing PGD masks each training step and running N optimization steps,
PersistentPGD maintains persistent masks that receive one gradient update per training step.
Over many steps, these masks converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.
"""

from typing import Literal

import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDOptimizerConfig
from spd.models.component_model import ComponentModel
from spd.models.components import RoutingMasks, make_mask_infos
from spd.routing import Router
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm
from spd.utils.wandb_utils import try_wandb

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
        self.use_delta_component = use_delta_component
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
            self.masks[module_name] = torch.rand(
                mask_c, device=device, requires_grad=True, generator=rng
            )
            if optimizer_cfg.type == "adam":
                self._adam_m[module_name] = torch.zeros_like(self.masks[module_name])
                self._adam_v[module_name] = torch.zeros_like(self.masks[module_name])

    def step(
        self,
        grads: dict[str, Float[Tensor, " mask_c"]],
    ) -> None:
        """Perform one PGD update step using the provided gradients.

        Updates masks in-place, then clamps to [0, 1].
        """
        with torch.no_grad():
            if self.optimizer_cfg.type == "sign":
                cfg = self.optimizer_cfg
                for module_name in self.masks:
                    self.masks[module_name].add_(cfg.step_size * grads[module_name].sign())
                    self.masks[module_name].clamp_(0.0, 1.0)
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
                    mask.clamp_(0.0, 1.0)
            else:
                raise ValueError(f"Unknown PersistentPGD optimizer: {self.optimizer_cfg.type}")

        # Re-enable gradients for next step
        for mask in self.masks.values():
            mask.requires_grad_(True)

    def get_expanded_masks(
        self, batch_dims: tuple[int, ...]
    ) -> dict[str, Float[Tensor, "*batch_dims mask_c"]]:
        """Expand masks to full batch dimensions.

        Args:
            batch_dims: Target batch dimensions, e.g. (batch_size, seq_len)

        Returns:
            Masks expanded to (*batch_dims, mask_c) via broadcasting
        """
        expanded: dict[str, Float[Tensor, "*batch_dims mask_c"]] = {}
        for module_name, mask in self.masks.items():
            # mask is (mask_c,), expand to (*batch_dims, mask_c)
            # Use contiguous() to create a new tensor (not a view) while preserving gradients.
            # This is needed because we update masks in-place later, which would invalidate views.
            expanded[module_name] = mask.expand(*batch_dims, -1).contiguous()

        return expanded


class PersistentPGDResult:
    """Result from persistent_pgd_recon_loss containing loss and deferred mask gradients."""

    def __init__(
        self,
        loss: Float[Tensor, ""],
        grads: dict[str, Float[Tensor, " mask_c"]],
        pgd_state: PersistentPGDState,
    ) -> None:
        self.loss = loss
        self.grads = grads
        self.pgd_state = pgd_state

    def apply_pgd_step(self) -> None:
        """Apply the PGD step to update masks. Call this AFTER .backward()."""
        try_wandb(
            wandb.log,
            {
                f"persistent_pgd_loss/mean_abs_grad/{module_name}": v.abs().mean()
                for module_name, v in self.grads.items()
            },
        )
        self.pgd_state.step(self.grads)


def persistent_pgd_recon_loss(
    model: ComponentModel,
    batch: torch.Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    pgd_state: PersistentPGDState,
    router: Router,
) -> PersistentPGDResult:
    """Compute reconstruction loss with persistent PGD masks.

    Unlike standard PGD which runs N steps per training step, this:
    1. Uses persistent masks from pgd_state
    2. Computes forward pass and loss
    3. Computes gradients w.r.t. masks
    4. Returns the loss and gradients (does NOT update masks yet)

    The caller must call result.apply_pgd_step() AFTER .backward() to update masks.
    This is necessary because in-place mask updates before .backward() would invalidate
    the computation graph.

    Args:
        model: ComponentModel to evaluate
        batch: Input batch
        ci: Causal importance values per module
        weight_deltas: Optional weight deltas for delta component
        target_out: Target model output
        output_loss_type: "mse" or "kl"
        pgd_state: Persistent PGD state holding masks
        router: Router for subset routing masks.

    Returns:
        PersistentPGDResult containing loss and deferred gradients
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
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
    # retain_graph=True because the loss will be used in .backward() later
    grads = torch.autograd.grad(loss, list(pgd_state.masks.values()), retain_graph=True)
    grads_dict = {
        k: all_reduce(g, op=ReduceOp.SUM)
        for k, g in zip(pgd_state.masks.keys(), grads, strict=True)
    }

    return PersistentPGDResult(loss=loss, grads=grads_dict, pgd_state=pgd_state)


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

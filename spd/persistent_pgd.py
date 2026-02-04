"""Persistent PGD: Persistent adversarial masks that evolve across training steps.

Instead of reinitializing PGD masks each training step and running N optimization steps,
PersistentPGD maintains persistent masks that receive one gradient update per training step.
Over many steps, these masks converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.
"""

from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    SubsetRoutingType,
)
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import ComponentsMaskInfo, RoutingMasks, make_mask_infos
from spd.routing import AllLayersRouter, Router, get_subset_router
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm

PPGDMasks = dict[str, Float[Tensor, " mask_c"]]


class PersistentPGDState:
    """Persistent state for persistent PGD optimization.

    Holds a single adversarial mask per module that persists across training steps.
    The mask is shared across all batch elements and ranks.

    Shape: {module_name: (C,)} per module, broadcast to batch dims during forward.
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        batch_dims: tuple[int, ...],
        device: torch.device | str,
        use_delta_component: bool,
        cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
    ) -> None:
        self.optimizer = cfg.optimizer
        self.scope = cfg.scope

        self._adam_step = 0
        self._adam_m: PPGDMasks = {}
        self._adam_v: PPGDMasks = {}

        # Initialize masks randomly in [0, 1] with fixed seed for consistency across ranks
        # Shape: (mask_c,) per module - single mask shared across all batch elements
        self.masks: PPGDMasks = {}

        assert len(batch_dims) == 2, "PersistentPGD only supports the (batch, seq_len) shape case"
        B, S = batch_dims
        mask_leading_dims = {
            "single_mask": [1, 1],
            "broadcast_across_batch": [1, S],
            "unique_per_batch_per_token": [B, S],
        }[self.scope]

        for module_name, module_c in module_to_c.items():
            mask_c = module_c + 1 if use_delta_component else module_c
            mask_shape = mask_leading_dims + [mask_c]
            self.masks[module_name] = torch.rand(mask_shape, requires_grad=True, device=device)
            if self.optimizer.type == "adam":
                self._adam_m[module_name] = torch.zeros_like(self.masks[module_name])
                self._adam_v[module_name] = torch.zeros_like(self.masks[module_name])

    def get_grads(self, loss: Float[Tensor, ""]) -> PPGDMasks:
        grads = torch.autograd.grad(loss, list(self.masks.values()), retain_graph=True)

        return {
            k: all_reduce(g, op=ReduceOp.SUM) for k, g in zip(self.masks.keys(), grads, strict=True)
        }

    def step(self, grads: PPGDMasks) -> dict[str, float]:
        """Perform one PGD update step using the provided gradients.

        Updates masks in-place, then clamps to [0, 1].

        Returns:
            Mean absolute step per module (before clamping).
        """
        mean_abs_step: dict[str, float] = {}
        with torch.no_grad():
            if self.optimizer.type == "sign":
                cfg = self.optimizer
                for module_name in self.masks:
                    step_tensor = cfg.step_size * grads[module_name].sign()
                    mean_abs_step[module_name] = step_tensor.abs().mean().item()
                    self.masks[module_name].add_(step_tensor)
            elif self.optimizer.type == "adam":
                cfg = self.optimizer
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
                    step_tensor = cfg.lr * m_hat / denom
                    mean_abs_step[module_name] = step_tensor.abs().mean().item()
                    mask.add_(step_tensor)
            else:
                raise ValueError(f"Unknown PersistentPGD optimizer: {self.optimizer.type}")

            for mask in self.masks.values():
                mask.clamp_(0.0, 1.0)

        return mean_abs_step

    def empty_grads(self) -> PPGDMasks:
        return {module_name: torch.zeros_like(mask) for module_name, mask in self.masks.items()}


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


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "... C"]],
    adv_sources_components: dict[str, Float[Tensor, "... C"]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Interpolate CI with adversarial masks: final = ci + (1 - ci) * adv."""
    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for module_name in ci:
        adv_source = adv_sources_components[module_name]
        scaled_noise = (1 - ci[module_name]) * adv_source
        component_masks[module_name] = ci[module_name] + scaled_noise
    return component_masks


def _persistent_pgd_recon_subset_loss_update(
    model: ComponentModel,
    ppgd_masks: PPGDMasks,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"

    mask_infos = get_mask_infos(model, ci, weight_deltas, ppgd_masks, router)
    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    n_examples = out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()

    return loss, n_examples


def persistent_pgd_recon_subset_loss(
    model: ComponentModel,
    ppgd_masks: PPGDMasks,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing: SubsetRoutingType,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _persistent_pgd_recon_subset_loss_update(
        model=model,
        ppgd_masks=ppgd_masks,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        router=get_subset_router(routing, batch.device),
    )
    return sum_loss / n_examples


def persistent_pgd_recon_loss(
    model: ComponentModel,
    ppgd_masks: PPGDMasks,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _persistent_pgd_recon_subset_loss_update(
        model=model,
        ppgd_masks=ppgd_masks,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        router=AllLayersRouter(),
    )
    return sum_loss / n_examples


class AbstractPersistentPGDReconLoss(Metric):
    """Recon loss when sampling with persistently, adversarially optimized masks and routing."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        router: Router,
        ppgd_masks: PPGDMasks,
    ) -> None:
        self.model = model
        self.use_delta_component: bool = use_delta_component
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.router = router
        self.ppgd_masks = ppgd_masks

        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _persistent_pgd_recon_subset_loss_update(
            model=self.model,
            ppgd_masks=self.ppgd_masks,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            router=self.router,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples


class PersistentPGDReconLoss(AbstractPersistentPGDReconLoss):
    """Recon loss when sampling with persistently, adversarially optimized masks and routing to all component layers."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        ppgd_masks: PPGDMasks,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            use_delta_component=use_delta_component,
            output_loss_type=output_loss_type,
            router=AllLayersRouter(),
            ppgd_masks=ppgd_masks,
        )


class PersistentPGDReconSubsetLoss(AbstractPersistentPGDReconLoss):
    """Recon loss when sampling with persistently, adversarially optimized masks and routing to subsets of component layers."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        ppgd_masks: PPGDMasks,
        routing: SubsetRoutingType,
    ) -> None:
        super().__init__(
            model,
            device,
            use_delta_component,
            output_loss_type,
            get_subset_router(routing, device),
            ppgd_masks=ppgd_masks,
        )

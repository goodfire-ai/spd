"""Persistent PGD: Persistent adversarial sources that evolve across training steps.

Instead of reinitializing PGD sources each training step and running N optimization steps,
PersistentPGD maintains persistent sources that receive one gradient update per training step.
Over many steps, these sources converge to strong adversarial configurations.

The key insight is that this amortizes PGD optimization across training steps - getting the
benefit of many PGD steps without the per-step computational cost.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import (
    AdamPGDConfig,
    BroadcastAcrossBatchScope,
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDOptimizerConfig,
    RepeatAcrossBatchScope,
    SignPGDConfig,
    SingleSourceScope,
    SubsetRoutingType,
)
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import ComponentsMaskInfo, RoutingMasks, make_mask_infos
from spd.routing import AllLayersRouter, Router, get_subset_router
from spd.utils.distributed_utils import all_reduce, call_on_rank0_then_broadcast
from spd.utils.general_utils import calc_sum_recon_loss_lm

PPGDSources = dict[str, Float[Tensor, " source_c"]]


class PPGDOptimizer(ABC):
    """Interface for persistent PGD optimizers."""

    @abstractmethod
    def init_state(self, sources: PPGDSources) -> None:
        """Initialize any optimizer-specific state for the given sources."""

    @abstractmethod
    def step(self, sources: PPGDSources, grads: PPGDSources) -> dict[str, float]:
        """Perform one update step on sources using gradients.

        Updates sources in-place. Returns mean absolute step per module.
        """


class SignPGDOptimizer(PPGDOptimizer):
    def __init__(self, cfg: SignPGDConfig) -> None:
        self._step_size = cfg.step_size

    @override
    def init_state(self, sources: PPGDSources) -> None:
        pass

    @override
    def step(self, sources: PPGDSources, grads: PPGDSources) -> dict[str, float]:
        mean_abs_step: dict[str, float] = {}
        for module_name in sources:
            step_tensor = self._step_size * grads[module_name].sign()
            mean_abs_step[module_name] = step_tensor.abs().mean().item()
            sources[module_name].add_(step_tensor)
        return mean_abs_step


class AdamPGDOptimizer(PPGDOptimizer):
    def __init__(self, cfg: AdamPGDConfig) -> None:
        self._lr = cfg.lr
        self._beta1 = cfg.beta1
        self._beta2 = cfg.beta2
        self._eps = cfg.eps
        self._step_count = 0
        self._m: PPGDSources = {}
        self._v: PPGDSources = {}

    @override
    def init_state(self, sources: PPGDSources) -> None:
        for module_name, source in sources.items():
            self._m[module_name] = torch.zeros_like(source)
            self._v[module_name] = torch.zeros_like(source)

    @override
    def step(self, sources: PPGDSources, grads: PPGDSources) -> dict[str, float]:
        self._step_count += 1
        bias_correction1 = 1 - self._beta1**self._step_count
        bias_correction2 = 1 - self._beta2**self._step_count
        mean_abs_step: dict[str, float] = {}
        for module_name, source in sources.items():
            grad = grads[module_name]
            m = self._m[module_name]
            v = self._v[module_name]
            m.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
            v.mul_(self._beta2).addcmul_(grad, grad, value=1 - self._beta2)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            denom = v_hat.sqrt().add_(self._eps)
            step_tensor = self._lr * m_hat / denom
            mean_abs_step[module_name] = step_tensor.abs().mean().item()
            source.add_(step_tensor)
        return mean_abs_step


def make_ppgd_optimizer(cfg: PGDOptimizerConfig) -> PPGDOptimizer:
    match cfg:
        case SignPGDConfig():
            return SignPGDOptimizer(cfg)
        case AdamPGDConfig():
            return AdamPGDOptimizer(cfg)


class PersistentPGDState:
    """Persistent state for persistent PGD optimization.

    Holds adversarial sources per module that persist across training steps.
    Source shape depends on scope: shared across batch (SingleSource, BroadcastAcrossBatch),
    repeated along batch dim (RepeatAcrossBatch), or per-batch-element-per-position with no
    cross-rank synchronization (PerBatchPerPosition).
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        seq_len: int,
        device: torch.device | str,
        use_delta_component: bool,
        cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
        batch_size: int | None = None,
    ) -> None:
        self.optimizer = make_ppgd_optimizer(cfg.optimizer)
        self._skip_all_reduce = isinstance(cfg.scope, PerBatchPerPositionScope)
        self._use_sigmoid_parameterization = cfg.use_sigmoid_parameterization

        self.sources: PPGDSources = {}

        match cfg.scope:
            case SingleSourceScope():
                source_leading_dims = [1, 1]
            case BroadcastAcrossBatchScope():
                source_leading_dims = [1, seq_len]
            case RepeatAcrossBatchScope(n_sources=n):
                source_leading_dims = [n, seq_len]
            case PerBatchPerPositionScope():
                assert batch_size is not None, "batch_size required for PerBatchPerPositionScope"
                source_leading_dims = [batch_size, seq_len]

        init_fn = torch.randn if self._use_sigmoid_parameterization else torch.rand
        for module_name, module_c in module_to_c.items():
            source_c = module_c + 1 if use_delta_component else module_c
            source_shape = source_leading_dims + [source_c]
            source_data = call_on_rank0_then_broadcast(init_fn, source_shape)
            self.sources[module_name] = source_data.to(device=device).requires_grad_(True)

        self.optimizer.init_state(self.sources)

    def get_grads(
        self, loss: Float[Tensor, ""], r1_coeff: float = 0.0
    ) -> tuple[PPGDSources, float]:
        use_r1 = r1_coeff > 0.0
        source_values = list(self.sources.values())
        grads = torch.autograd.grad(
            loss, source_values, retain_graph=True, create_graph=use_r1
        )

        if use_r1:
            r1_val = r1_coeff * torch.stack([g.pow(2).sum() for g in grads]).sum()
            r1_grads = torch.autograd.grad(r1_val, source_values, retain_graph=True)
            grads = tuple(
                (g - rg).detach() for g, rg in zip(grads, r1_grads, strict=True)
            )
            r1_log = r1_val.item()
        else:
            r1_log = 0.0

        if self._skip_all_reduce:
            return dict(zip(self.sources.keys(), grads, strict=True)), r1_log
        return {
            k: all_reduce(g, op=ReduceOp.SUM)
            for k, g in zip(self.sources.keys(), grads, strict=True)
        }, r1_log

    def step(self, grads: PPGDSources) -> dict[str, float]:
        """Perform one PGD update step using the provided gradients.

        Updates sources in-place, then clamps to [0, 1] (or leaves unbounded when using sigmoid
        parameterization, where sigmoid is applied when reading effective sources).

        Returns:
            Mean absolute step per module (before clamping).
        """
        with torch.no_grad():
            mean_abs_step = self.optimizer.step(self.sources, grads)

            if not self._use_sigmoid_parameterization:
                for source in self.sources.values():
                    source.clamp_(0.0, 1.0)

        return mean_abs_step

    def get_effective_sources(self) -> PPGDSources:
        """Return sources in [0, 1] range.

        If using sigmoid parameterization, applies sigmoid to unconstrained values. Otherwise
        returns raw sources (already clamped to [0, 1]).
        """
        if self._use_sigmoid_parameterization:
            return {k: torch.sigmoid(v) for k, v in self.sources.items()}
        return self.sources

    def empty_grads(self) -> PPGDSources:
        return {
            module_name: torch.zeros_like(source) for module_name, source in self.sources.items()
        }


def get_mask_infos(
    model: ComponentModel,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    ppgd_sources: dict[str, Float[Tensor, "*batch_dims source_c"]],
    router: Router,
) -> dict[str, ComponentsMaskInfo]:
    """Get mask infos for persistent PGD."""

    batch_dims = next(iter(ci.values())).shape[:-1]
    routing_masks: RoutingMasks = router.get_masks(
        module_names=model.target_module_paths, mask_shape=batch_dims
    )

    expanded_adv_sources: dict[str, Float[Tensor, "*batch_dims source_c"]] = {}
    for module_name, source in ppgd_sources.items():
        B = batch_dims[0]
        N = source.shape[0]
        if N == 1 or N == B:
            expanded_adv_sources[module_name] = source.expand(*batch_dims, -1)
        else:
            assert B % N == 0, f"source leading dim {N} must divide batch dim {B}"
            expanded_adv_sources[module_name] = source.repeat(B // N, 1, 1)

    # Split into component sources and weight delta sources
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
    """Interpolate CI with adversarial sources: final = ci + (1 - ci) * adv."""
    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for module_name in ci:
        adv_source = adv_sources_components[module_name]
        scaled_adv = (1 - ci[module_name]) * adv_source
        component_masks[module_name] = ci[module_name] + scaled_adv
    return component_masks


def _persistent_pgd_recon_subset_loss_update(
    model: ComponentModel,
    ppgd_sources: PPGDSources,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"

    mask_infos = get_mask_infos(model, ci, weight_deltas, ppgd_sources, router)
    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    n_examples = out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()

    return loss, n_examples


def persistent_pgd_recon_subset_loss(
    model: ComponentModel,
    ppgd_sources: PPGDSources,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing: SubsetRoutingType,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _persistent_pgd_recon_subset_loss_update(
        model=model,
        ppgd_sources=ppgd_sources,
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
    ppgd_sources: PPGDSources,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _persistent_pgd_recon_subset_loss_update(
        model=model,
        ppgd_sources=ppgd_sources,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        router=AllLayersRouter(),
    )
    return sum_loss / n_examples


class AbstractPersistentPGDReconLoss(Metric):
    """Recon loss when sampling with persistently, adversarially optimized sources and routing."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        router: Router,
        ppgd_sources: PPGDSources,
    ) -> None:
        self.model = model
        self.use_delta_component: bool = use_delta_component
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.router = router
        self.ppgd_sources = ppgd_sources

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
            ppgd_sources=self.ppgd_sources,
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
    """Recon loss when sampling with persistently, adversarially optimized sources and routing to all component layers."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        ppgd_sources: PPGDSources,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            use_delta_component=use_delta_component,
            output_loss_type=output_loss_type,
            router=AllLayersRouter(),
            ppgd_sources=ppgd_sources,
        )


class PersistentPGDReconSubsetLoss(AbstractPersistentPGDReconLoss):
    """Recon loss when sampling with persistently, adversarially optimized sources and routing to subsets of component layers."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
        ppgd_sources: PPGDSources,
        routing: SubsetRoutingType,
    ) -> None:
        super().__init__(
            model,
            device,
            use_delta_component,
            output_loss_type,
            get_subset_router(routing, device),
            ppgd_sources=ppgd_sources,
        )

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import Tensor
from tqdm.auto import tqdm

from spd.configs import ImportanceMinimalityLossConfig, PGDInitStrategy, SamplingType
from spd.metrics import importance_minimality_loss
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.routing import AllLayersRouter
from spd.spd_types import Probability
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_component_mask_info
from spd.utils.general_utils import bf16_autocast

MaskType = Literal["stochastic", "ci"]


class AdvPGDConfig(BaseModel):
    """PGD adversary config for robust CI optimization."""

    n_steps: int
    step_size: float
    init: PGDInitStrategy


class CELossConfig(BaseModel):
    """Cross-entropy loss: optimize for a specific token at a position."""

    type: Literal["ce"] = "ce"
    coeff: float
    position: int
    label_token: int


class KLLossConfig(BaseModel):
    """KL divergence loss: match target model distribution at a position."""

    type: Literal["kl"] = "kl"
    coeff: float
    position: int


LossConfig = CELossConfig | KLLossConfig


def _compute_recon_loss(
    logits: Tensor,
    loss_config: LossConfig,
    target_out: Tensor,
    device: str,
) -> Tensor:
    """Compute recon loss (CE or KL) from model output logits at the configured position."""
    match loss_config:
        case CELossConfig(position=pos, label_token=label_token):
            return F.cross_entropy(
                logits[0, pos, :].unsqueeze(0),
                torch.tensor([label_token], device=device),
            )
        case KLLossConfig(position=pos):
            target_probs = F.softmax(target_out[0, pos, :], dim=-1)
            pred_log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
            return F.kl_div(pred_log_probs, target_probs, reduction="sum")


def _interpolate_masks(
    ci: dict[str, Tensor],
    sources: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute PGD component masks: ci + (1 - ci) * source."""
    return {layer: ci[layer] + (1 - ci[layer]) * sources[layer] for layer in ci}


@dataclass
class AliveComponentInfo:
    """Info about which components are alive at each position for each layer."""

    alive_masks: dict[str, Bool[Tensor, "1 seq C"]]  # Per-layer masks of alive positions
    alive_counts: dict[str, list[int]]  # Number of alive components per position per layer


def compute_alive_info(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
) -> AliveComponentInfo:
    """Compute which (position, component) pairs are alive (CI > 0)."""
    alive_masks: dict[str, Bool[Tensor, "1 seq C"]] = {}
    alive_counts: dict[str, list[int]] = {}

    for layer_name, ci in ci_lower_leaky.items():
        mask = ci > 0.0
        alive_masks[layer_name] = mask
        # Count alive components per position: mask is [1, seq, C], sum over C
        counts_per_pos = mask[0].sum(dim=-1)  # [seq]
        alive_counts[layer_name] = counts_per_pos.tolist()

    return AliveComponentInfo(alive_masks=alive_masks, alive_counts=alive_counts)


class OptimizationMetrics(BaseModel):
    """Final loss metrics from CI optimization."""

    ci_masked_label_prob: float | None = None  # Probability of label under CI mask (CE loss only)
    stoch_masked_label_prob: float | None = (
        None  # Probability of label under stochastic mask (CE loss only)
    )
    adv_pgd_label_prob: float | None = None  # Probability of label under adversarial mask (CE only)
    l0_total: float  # Total L0 (active components)


@dataclass
class OptimizableCIParams:
    """Container for optimizable CI pre-sigmoid parameters."""

    # List of pre-sigmoid tensors for alive positions at each sequence position
    ci_pre_sigmoid: dict[str, list[Tensor]]  # layer_name -> list of [alive_at_pos] values
    alive_info: AliveComponentInfo

    def create_ci_outputs(self, model: ComponentModel, device: str) -> CIOutputs:
        """Expand sparse pre-sigmoid values to full CI tensors and create CIOutputs."""
        pre_sigmoid: dict[str, Tensor] = {}

        for layer_name, mask in self.alive_info.alive_masks.items():
            # Create full tensors (default to 0 for non-alive positions)
            full_pre_sigmoid = torch.zeros_like(mask, dtype=torch.float32, device=device)

            # Get pre-sigmoid list for this layer
            layer_pre_sigmoid_list = self.ci_pre_sigmoid[layer_name]

            # For each position, place the values
            seq_len = mask.shape[1]
            for pos in range(seq_len):
                pos_mask = mask[0, pos, :]  # [C]
                pos_pre_sigmoid = layer_pre_sigmoid_list[pos]  # [alive_at_pos]
                full_pre_sigmoid[0, pos, pos_mask] = pos_pre_sigmoid

            pre_sigmoid[layer_name] = full_pre_sigmoid

        return CIOutputs(
            lower_leaky={k: model.lower_leaky_fn(v) for k, v in pre_sigmoid.items()},
            upper_leaky={k: model.upper_leaky_fn(v) for k, v in pre_sigmoid.items()},
            pre_sigmoid=pre_sigmoid,
        )

    def get_parameters(self) -> list[Tensor]:
        """Get all optimizable parameters."""
        params: list[Tensor] = []
        for layer_pre_sigmoid_list in self.ci_pre_sigmoid.values():
            params.extend(layer_pre_sigmoid_list)
        return params


def create_optimizable_ci_params(
    alive_info: AliveComponentInfo,
    initial_pre_sigmoid: dict[str, Tensor],
) -> OptimizableCIParams:
    """Create optimizable CI parameters for alive positions.

    Creates parameters initialized from the initial pre-sigmoid values for each
    (position, component) pair where initial CI > threshold.
    """
    ci_pre_sigmoid: dict[str, list[Tensor]] = {}

    for layer_name, mask in alive_info.alive_masks.items():
        # Get initial pre-sigmoid values for this layer
        layer_initial = initial_pre_sigmoid[layer_name]  # [1, seq, C]

        # Create a tensor for each position
        layer_pre_sigmoid_list: list[Tensor] = []
        seq_len = mask.shape[1]
        for pos in range(seq_len):
            pos_mask = mask[0, pos, :]  # [C]
            # Extract initial values for alive positions at this position
            initial_values = layer_initial[0, pos, pos_mask].clone().detach()
            initial_values.requires_grad_(True)
            layer_pre_sigmoid_list.append(initial_values)
        ci_pre_sigmoid[layer_name] = layer_pre_sigmoid_list

    return OptimizableCIParams(
        ci_pre_sigmoid=ci_pre_sigmoid,
        alive_info=alive_info,
    )


def compute_l0_stats(
    ci_outputs: CIOutputs,
    ci_alive_threshold: float,
) -> dict[str, float]:
    """Compute L0 statistics for each layer."""
    stats: dict[str, float] = {}
    for layer_name, layer_ci in ci_outputs.lower_leaky.items():
        l0_val = calc_ci_l_zero(layer_ci, ci_alive_threshold)
        stats[f"l0/{layer_name}"] = l0_val
    stats["l0/total"] = sum(stats.values())
    return stats


def compute_specific_pos_ce_kl(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
    rounding_threshold: float,
    loss_seq_pos: int,
) -> dict[str, float]:
    """Compute CE and KL metrics for a specific sequence position.

    Args:
        model: The ComponentModel.
        batch: Input tokens of shape [1, seq_len].
        target_out: Target model output logits of shape [1, seq_len, vocab].
        ci: Causal importance values (lower_leaky) per layer.
        rounding_threshold: Threshold for rounding CI values to binary masks.
        loss_seq_pos: Sequence position to compute metrics for.

    Returns:
        Dict with kl and ce_difference metrics for ci_masked, unmasked, and rounded_masked.
    """
    assert batch.ndim == 2 and batch.shape[0] == 1, "Expected batch shape [1, seq_len]"

    # Get target logits at the specified position
    target_logits = target_out[0, loss_seq_pos, :]  # [vocab]

    def kl_vs_target(logits: Tensor) -> float:
        """KL divergence between predicted and target logits at target position."""
        pos_logits = logits[0, loss_seq_pos, :]  # [vocab]
        target_probs = F.softmax(target_logits, dim=-1)
        pred_log_probs = F.log_softmax(pos_logits, dim=-1)
        return F.kl_div(pred_log_probs, target_probs, reduction="sum").item()

    def ce_vs_target(logits: Tensor) -> float:
        """CE between predicted logits and target's argmax at target position."""
        pos_logits = logits[0, loss_seq_pos, :]  # [vocab]
        target_token = target_logits.argmax()
        return F.cross_entropy(pos_logits.unsqueeze(0), target_token.unsqueeze(0)).item()

    # Target model CE (baseline)
    target_ce = ce_vs_target(target_out)

    # CI masked
    ci_mask_infos = make_mask_infos(ci)
    with bf16_autocast():
        ci_masked_logits = model(batch, mask_infos=ci_mask_infos)
    ci_masked_kl = kl_vs_target(ci_masked_logits)
    ci_masked_ce = ce_vs_target(ci_masked_logits)

    # Unmasked (all components active)
    unmasked_infos = make_mask_infos({k: torch.ones_like(v) for k, v in ci.items()})
    with bf16_autocast():
        unmasked_logits = model(batch, mask_infos=unmasked_infos)
    unmasked_kl = kl_vs_target(unmasked_logits)
    unmasked_ce = ce_vs_target(unmasked_logits)

    # Rounded masked (binary masks based on threshold)
    rounded_mask_infos = make_mask_infos(
        {k: (v > rounding_threshold).float() for k, v in ci.items()}
    )
    with bf16_autocast():
        rounded_masked_logits = model(batch, mask_infos=rounded_mask_infos)
    rounded_masked_kl = kl_vs_target(rounded_masked_logits)
    rounded_masked_ce = ce_vs_target(rounded_masked_logits)

    return {
        "kl_ci_masked": ci_masked_kl,
        "kl_unmasked": unmasked_kl,
        "kl_rounded_masked": rounded_masked_kl,
        "ce_difference_ci_masked": ci_masked_ce - target_ce,
        "ce_difference_unmasked": unmasked_ce - target_ce,
        "ce_difference_rounded_masked": rounded_masked_ce - target_ce,
    }


@dataclass
class OptimCIConfig:
    """Configuration for optimizing CI values on a single prompt."""

    seed: int

    # Optimization hyperparameters
    lr: float
    steps: int
    weight_decay: float
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"]
    lr_exponential_halflife: float | None
    lr_warmup_pct: Probability

    log_freq: int

    # Loss config (exactly one of CE or KL)
    imp_min_config: ImportanceMinimalityLossConfig
    loss_config: LossConfig

    sampling: SamplingType

    ce_kl_rounding_threshold: float
    mask_type: MaskType
    adv_pgd: AdvPGDConfig | None


ProgressCallback = Callable[[int, int, str], None]  # (current, total, stage)


@dataclass
class OptimizeCIResult:
    """Result from CI optimization including params and final metrics."""

    params: OptimizableCIParams
    metrics: OptimizationMetrics
    adv_pgd_out_logits: Float[Tensor, "seq vocab"] | None = None


def _run_adv_pgd(
    model: ComponentModel,
    tokens: Tensor,
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    alive_masks: dict[str, Bool[Tensor, "1 seq C"]],
    adv_config: AdvPGDConfig,
    loss_config: LossConfig,
    target_out: Tensor,
    device: str,
) -> dict[str, Float[Tensor, "1 seq C"]]:
    """Run PGD to find adversarial sources maximizing reconstruction loss.

    Sources are optimized via signed gradient ascent. Only alive positions are optimized.
    Masks are computed as ci + (1 - ci) * source (same interpolation as training PGD).

    Returns detached adversarial source tensors.
    """
    ci_detached = {k: v.detach() for k, v in ci_lower_leaky.items()}

    adv_sources: dict[str, Tensor] = {}
    for layer_name, ci in ci_detached.items():
        match adv_config.init:
            case "random":
                source = torch.rand_like(ci)
            case "ones":
                source = torch.ones_like(ci)
            case "zeroes":
                source = torch.zeros_like(ci)
        source[~alive_masks[layer_name]] = 0.0
        source.requires_grad_(True)
        adv_sources[layer_name] = source

    source_list = list(adv_sources.values())

    for _ in range(adv_config.n_steps):
        mask_infos = make_mask_infos(_interpolate_masks(ci_detached, adv_sources))

        with bf16_autocast():
            out = model(tokens, mask_infos=mask_infos)

        loss = _compute_recon_loss(out, loss_config, target_out, device)
        grads = torch.autograd.grad(loss, source_list)
        with torch.no_grad():
            for (layer_name, source), grad in zip(adv_sources.items(), grads, strict=True):
                source.add_(adv_config.step_size * grad.sign())
                source.clamp_(0.0, 1.0)
                source[~alive_masks[layer_name]] = 0.0

    return {k: v.detach() for k, v in adv_sources.items()}


def optimize_ci_values(
    model: ComponentModel,
    tokens: Tensor,
    config: OptimCIConfig,
    device: str,
    on_progress: ProgressCallback | None,
) -> OptimizeCIResult:
    """Optimize CI values for a single prompt.

    Args:
        model: The ComponentModel (weights will be frozen).
        tokens: Tokenized prompt of shape [1, seq_len].
        config: Optimization configuration (includes loss configs).
        device: Device to run on.

    Returns:
        OptimizeCIResult containing params and final metrics.
    """
    imp_min_coeff = config.imp_min_config.coeff
    assert imp_min_coeff is not None, "Importance minimality loss coefficient must be set"

    model.requires_grad_(False)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)
    ci_params: OptimizableCIParams = create_optimizable_ci_params(
        alive_info=alive_info,
        initial_pre_sigmoid=initial_ci_outputs.pre_sigmoid,
    )

    weight_deltas = model.calc_weight_deltas()

    params = ci_params.get_parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    progress_interval = max(1, config.steps // 20)  # Report ~20 times during optimization
    for step in tqdm(range(config.steps), desc="Optimizing CI values"):
        if on_progress is not None and step % progress_interval == 0:
            on_progress(step, config.steps, "optimizing")

        optimizer.zero_grad()

        ci_outputs = ci_params.create_ci_outputs(model, device)

        # Recon forward pass (stochastic or CI masking)
        match config.mask_type:
            case "stochastic":
                recon_mask_infos = calc_stochastic_component_mask_info(
                    causal_importances=ci_outputs.lower_leaky,
                    component_mask_sampling=config.sampling,
                    weight_deltas=weight_deltas,
                    router=AllLayersRouter(),
                )
            case "ci":
                recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)

        with bf16_autocast():
            recon_out = model(tokens, mask_infos=recon_mask_infos)

        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=step / config.steps,
            pnorm=config.imp_min_config.pnorm,
            beta=config.imp_min_config.beta,
            eps=config.imp_min_config.eps,
            p_anneal_start_frac=config.imp_min_config.p_anneal_start_frac,
            p_anneal_final_p=config.imp_min_config.p_anneal_final_p,
            p_anneal_end_frac=config.imp_min_config.p_anneal_end_frac,
        )

        recon_loss = _compute_recon_loss(recon_out, config.loss_config, target_out, device)
        total_loss = config.loss_config.coeff * recon_loss + imp_min_coeff * imp_min_loss

        # PGD adversarial loss (runs in tandem with recon)
        if config.adv_pgd is not None:
            adv_sources = _run_adv_pgd(
                model=model,
                tokens=tokens,
                ci_lower_leaky=ci_outputs.lower_leaky,
                alive_masks=alive_info.alive_masks,
                adv_config=config.adv_pgd,
                loss_config=config.loss_config,
                target_out=target_out,
                device=device,
            )
            pgd_mask_infos = make_mask_infos(
                _interpolate_masks(ci_outputs.lower_leaky, adv_sources)
            )

            with bf16_autocast():
                pgd_out = model(tokens, mask_infos=pgd_mask_infos)

            pgd_loss = _compute_recon_loss(pgd_out, config.loss_config, target_out, device)
            total_loss = total_loss + config.loss_config.coeff * pgd_loss

        if step % config.log_freq == 0 or step == config.steps - 1:
            l0_stats = compute_l0_stats(ci_outputs, ci_alive_threshold=0.0)

            with torch.no_grad():
                ce_kl_stats = compute_specific_pos_ce_kl(
                    model=model,
                    batch=tokens,
                    target_out=target_out,
                    ci=ci_outputs.lower_leaky,
                    rounding_threshold=config.ce_kl_rounding_threshold,
                    loss_seq_pos=config.loss_config.position,
                )

            log_terms: dict[str, float] = {
                "imp_min_loss": imp_min_loss.item(),
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
            }

            if isinstance(config.loss_config, CELossConfig):
                pos = config.loss_config.position
                label_token = config.loss_config.label_token
                recon_label_prob = F.softmax(recon_out[0, pos, :], dim=-1)[label_token]
                log_terms["recon_masked_label_prob"] = recon_label_prob.item()

                with torch.no_grad():
                    mask_infos = make_mask_infos(ci_outputs.lower_leaky, routing_masks="all")
                    logits = model(tokens, mask_infos=mask_infos)
                    probs = F.softmax(logits[0, pos, :], dim=-1)
                    log_terms["ci_masked_label_prob"] = float(probs[label_token].item())

            tqdm.write(f"\n--- Step {step} ---")
            for name, value in log_terms.items():
                tqdm.write(f"  {name}: {value:.6f}")
            for name, value in l0_stats.items():
                tqdm.write(f"  {name}: {value:.2f}")
            for name, value in ce_kl_stats.items():
                tqdm.write(f"  {name}: {value:.6f}")

        total_loss.backward()
        optimizer.step()

    # Compute final metrics after optimization
    with torch.no_grad():
        final_ci_outputs = ci_params.create_ci_outputs(model, device)
        final_l0_stats = compute_l0_stats(final_ci_outputs, ci_alive_threshold=0.0)

        final_ci_masked_label_prob: float | None = None
        final_stoch_masked_label_prob: float | None = None

        if isinstance(config.loss_config, CELossConfig):
            pos = config.loss_config.position
            label_token = config.loss_config.label_token

            # CI-masked probability
            ci_mask_infos = make_mask_infos(final_ci_outputs.lower_leaky, routing_masks="all")
            ci_logits = model(tokens, mask_infos=ci_mask_infos)
            ci_probs = F.softmax(ci_logits[0, pos, :], dim=-1)
            final_ci_masked_label_prob = float(ci_probs[label_token].item())

            # Stochastic-masked probability (sample once for final metric)
            stoch_mask_infos = calc_stochastic_component_mask_info(
                causal_importances=final_ci_outputs.lower_leaky,
                component_mask_sampling=config.sampling,
                weight_deltas=weight_deltas,
                router=AllLayersRouter(),
            )
            stoch_logits = model(tokens, mask_infos=stoch_mask_infos)
            stoch_probs = F.softmax(stoch_logits[0, pos, :], dim=-1)
            final_stoch_masked_label_prob = float(stoch_probs[label_token].item())

    # Adversarial PGD final evaluation (needs gradients for PGD, so outside no_grad block)
    adv_pgd_out_logits: Float[Tensor, "seq vocab"] | None = None
    final_adv_pgd_label_prob: float | None = None

    if config.adv_pgd is not None:
        final_adv_sources = _run_adv_pgd(
            model=model,
            tokens=tokens,
            ci_lower_leaky=final_ci_outputs.lower_leaky,
            alive_masks=alive_info.alive_masks,
            adv_config=config.adv_pgd,
            loss_config=config.loss_config,
            target_out=target_out,
            device=device,
        )
        with torch.no_grad():
            adv_pgd_masks = make_mask_infos(
                _interpolate_masks(final_ci_outputs.lower_leaky, final_adv_sources)
            )
            with bf16_autocast():
                adv_logits = model(tokens, mask_infos=adv_pgd_masks)
            adv_pgd_out_logits = adv_logits[0].detach()  # [seq, vocab]

            if isinstance(config.loss_config, CELossConfig):
                pos = config.loss_config.position
                label_token = config.loss_config.label_token
                adv_probs = F.softmax(adv_logits[0, pos, :], dim=-1)
                final_adv_pgd_label_prob = float(adv_probs[label_token].item())

    metrics = OptimizationMetrics(
        ci_masked_label_prob=final_ci_masked_label_prob,
        stoch_masked_label_prob=final_stoch_masked_label_prob,
        adv_pgd_label_prob=final_adv_pgd_label_prob,
        l0_total=final_l0_stats["l0/total"],
    )

    return OptimizeCIResult(
        params=ci_params,
        metrics=metrics,
        adv_pgd_out_logits=adv_pgd_out_logits,
    )


def get_out_dir() -> Path:
    """Get the output directory for optimization results."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

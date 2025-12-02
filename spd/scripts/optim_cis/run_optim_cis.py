# %%
"""Optimize CI values for a single prompt while keeping component weights fixed."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import ImportanceMinimalityLossConfig
from spd.metrics import importance_minimality_loss
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.routing import AllLayersRouter
from spd.scripts.model_loading import load_model_from_wandb
from spd.scripts.optim_cis.config import OptimCIConfig
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_component_mask_info
from spd.utils.general_utils import set_seed


@dataclass
class AliveComponentInfo:
    """Info about which components are alive at each position for each layer."""

    alive_masks: dict[str, Bool[Tensor, "1 seq C"]]  # Per-layer masks of alive positions
    alive_counts: dict[str, list[int]]  # Number of alive components per position per layer


def compute_alive_info(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    ci_threshold: float,
) -> AliveComponentInfo:
    """Compute which (position, component) pairs are alive based on initial CI values."""
    alive_masks: dict[str, Bool[Tensor, "1 seq C"]] = {}
    alive_counts: dict[str, list[int]] = {}

    for layer_name, ci in ci_lower_leaky.items():
        mask = ci > ci_threshold
        alive_masks[layer_name] = mask
        # Count alive components per position: mask is [1, seq, C], sum over C
        counts_per_pos = mask[0].sum(dim=-1)  # [seq]
        alive_counts[layer_name] = counts_per_pos.tolist()

    return AliveComponentInfo(alive_masks=alive_masks, alive_counts=alive_counts)


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


def compute_final_token_ce_kl(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
    rounding_threshold: float,
) -> dict[str, float]:
    """Compute CE and KL metrics for the final token only.

    Args:
        model: The ComponentModel.
        batch: Input tokens of shape [1, seq_len].
        target_out: Target model output logits of shape [1, seq_len, vocab].
        ci: Causal importance values (lower_leaky) per layer.
        rounding_threshold: Threshold for rounding CI values to binary masks.

    Returns:
        Dict with kl and ce_difference metrics for ci_masked, unmasked, and rounded_masked.
    """
    assert batch.ndim == 2 and batch.shape[0] == 1, "Expected batch shape [1, seq_len]"

    # Get the label for CE (next token prediction at final position)
    # The label is the token at the final position for the second-to-last logit prediction
    # But since we're optimizing for CI on a single prompt, we use the final logit position
    final_target_logits = target_out[0, -1, :]  # [vocab]

    def kl_vs_target(logits: Tensor) -> float:
        """KL divergence between predicted and target logits at final position."""
        final_logits = logits[0, -1, :]  # [vocab]
        target_probs = F.softmax(final_target_logits, dim=-1)
        pred_log_probs = F.log_softmax(final_logits, dim=-1)
        return F.kl_div(pred_log_probs, target_probs, reduction="sum").item()

    def ce_vs_target(logits: Tensor) -> float:
        """CE between predicted logits and target's argmax at final position."""
        final_logits = logits[0, -1, :]  # [vocab]
        target_token = final_target_logits.argmax()
        return F.cross_entropy(final_logits.unsqueeze(0), target_token.unsqueeze(0)).item()

    # Target model CE (baseline)
    target_ce = ce_vs_target(target_out)

    # CI masked
    ci_mask_infos = make_mask_infos(ci)
    ci_masked_logits = model(batch, mask_infos=ci_mask_infos)
    ci_masked_kl = kl_vs_target(ci_masked_logits)
    ci_masked_ce = ce_vs_target(ci_masked_logits)

    # Unmasked (all components active)
    unmasked_infos = make_mask_infos({k: torch.ones_like(v) for k, v in ci.items()})
    unmasked_logits = model(batch, mask_infos=unmasked_infos)
    unmasked_kl = kl_vs_target(unmasked_logits)
    unmasked_ce = ce_vs_target(unmasked_logits)

    # Rounded masked (binary masks based on threshold)
    rounded_mask_infos = make_mask_infos(
        {k: (v > rounding_threshold).float() for k, v in ci.items()}
    )
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


def optimize_ci_values(
    model: ComponentModel,
    tokens: Tensor,
    label_token: int,
    config: OptimCIConfig,
    device: str,
) -> OptimizableCIParams:
    """Optimize CI values for a single prompt.

    Args:
        model: The ComponentModel (weights will be frozen).
        tokens: Tokenized prompt of shape [1, seq_len].
        label_token: The token to optimize CI values for.
        config: Optimization configuration.
        device: Device to run on.
    Returns:
        The OptimizableCIParams object.
    """
    imp_min_coeff = config.imp_min_config.coeff
    assert imp_min_coeff is not None, "Importance minimality loss coefficient must be set"
    ce_loss_coeff = config.ce_loss_coeff

    # Freeze all model parameters
    model.requires_grad_(False)

    # Get initial CI values from the model
    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    # Compute alive info and create optimizable parameters
    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky, config.ci_threshold)
    ci_params = create_optimizable_ci_params(
        alive_info=alive_info,
        initial_pre_sigmoid=initial_ci_outputs.pre_sigmoid,
    )

    # Log initial alive counts
    total_alive = sum(sum(counts) for counts in alive_info.alive_counts.values())
    print(f"\nAlive components (CI > {config.ci_threshold}):")
    for layer_name, counts in alive_info.alive_counts.items():
        layer_total = sum(counts)
        print(f"  {layer_name}: {layer_total} total across {len(counts)} positions")
    print(f"  Total: {total_alive}")

    weight_deltas = model.calc_weight_deltas()

    params = ci_params.get_parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    for step in tqdm(range(config.steps), desc="Optimizing CI values"):
        optimizer.zero_grad()

        # Create CI outputs from current parameters
        ci_outputs = ci_params.create_ci_outputs(model, device)

        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci_outputs.lower_leaky,
            component_mask_sampling=config.sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        out = model(tokens, mask_infos=mask_infos)

        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=step / config.steps,
            pnorm=config.imp_min_config.pnorm,
            eps=config.imp_min_config.eps,
            p_anneal_start_frac=config.imp_min_config.p_anneal_start_frac,
            p_anneal_final_p=config.imp_min_config.p_anneal_final_p,
            p_anneal_end_frac=config.imp_min_config.p_anneal_end_frac,
        )
        ce_loss = F.cross_entropy(
            out[0, -1, :].unsqueeze(0), torch.tensor([label_token], device=device)
        )
        total_loss = ce_loss_coeff * ce_loss + imp_min_coeff * imp_min_loss
        # Get the output probability for the label_token in the final seq position
        label_prob = F.softmax(out[0, -1, :], dim=-1)[label_token]

        if step % config.log_freq == 0 or step == config.steps - 1:
            l0_stats = compute_l0_stats(ci_outputs, config.ci_threshold)

            # Compute CE/KL metrics for final token only
            with torch.no_grad():
                ce_kl_stats = compute_final_token_ce_kl(
                    model=model,
                    batch=tokens,
                    target_out=target_out,
                    ci=ci_outputs.lower_leaky,
                    rounding_threshold=config.ce_kl_rounding_threshold,
                )

            # Also calculate the ci-masked label probability
            with torch.no_grad():
                mask_infos = make_mask_infos(ci_outputs.lower_leaky, routing_masks="all")
                out = model(tokens, mask_infos=mask_infos)
                ci_masked_label_prob = F.softmax(out[0, -1, :], dim=-1)[label_token]

            log_terms = {
                "imp_min_loss": imp_min_loss.item(),
                "ce_loss": ce_loss.item(),
                "total_loss": total_loss.item(),
                "stoch_masked_label_prob": label_prob.item(),
                "ci_masked_label_prob": ci_masked_label_prob.item(),
            }
            tqdm.write(f"\n--- Step {step} ---")
            for name, value in log_terms.items():
                tqdm.write(f"  {name}: {value:.6f}")
            for name, value in l0_stats.items():
                tqdm.write(f"  {name}: {value:.2f}")
            for name, value in ce_kl_stats.items():
                tqdm.write(f"  {name}: {value:.6f}")

        total_loss.backward()
        optimizer.step()

    # with torch.no_grad():
    #     final_ci_outputs = ci_params.create_ci_outputs(model, device)

    # optimized_ci: dict[str, list[list[float]]] = {}
    # for layer_name, ci_tensor in final_ci_outputs.lower_leaky.items():
    #     optimized_ci[layer_name] = ci_tensor[0].cpu().tolist()

    return ci_params


def get_out_dir() -> Path:
    """Get the output directory for optimization results."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# %%
# Example configuration
if __name__ == "__main__":
    # Configuration
    config = OptimCIConfig(
        seed=0,
        # wandb_path="wandb:goodfire/spd/runs/jyo9duz5",  # ss_gpt2_simple-1.25M (4L)
        wandb_path="wandb:goodfire/spd/runs/33n6xjjt",  # ss_gpt2_simple-1L
        prompt="They walked hand in",
        label="hand",
        lr=1e-2,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        steps=2000,
        log_freq=500,
        imp_min_config=ImportanceMinimalityLossConfig(coeff=1e-1, pnorm=0.3),
        ce_loss_coeff=1,
        ci_threshold=1e-6,
        sampling="continuous",
        n_mask_samples=1,
        output_loss_type="kl",
        ce_kl_rounding_threshold=0.5,
    )

    set_seed(config.seed)

    loaded = load_model_from_wandb(config.wandb_path)
    model, run_config, device = loaded.model, loaded.config, loaded.device

    tokenizer = AutoTokenizer.from_pretrained(run_config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"

    print(f"\nPrompt: {config.prompt!r}")
    tokens = tokenizer.encode(config.prompt, return_tensors="pt", add_special_tokens=False)
    assert isinstance(tokens, Tensor), "Expected Tensor"
    tokens = tokens.to(device)
    print(f"Tokens shape: {tokens.shape}")
    token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
    print(f"Token strings: {token_strings}")
    label_token_ids = tokenizer.encode(config.label, add_special_tokens=False)
    assert len(label_token_ids) == 1, f"Expected single token for label, got {len(label_token_ids)}"
    label_token = label_token_ids[0]
    print(f"Label token: {label_token}")

    # Run optimization
    ci_params = optimize_ci_values(
        model=model,
        tokens=tokens,
        label_token=label_token,
        config=config,
        device=device,
    )

    # Get final metrics
    ci_outputs = ci_params.create_ci_outputs(model, device)
    l0_stats = compute_l0_stats(ci_outputs, config.ci_threshold)

    with torch.no_grad():
        target_out = model(tokens)
        ce_kl_stats = compute_final_token_ce_kl(
            model=model,
            batch=tokens,
            target_out=target_out,
            ci=ci_outputs.lower_leaky,
            rounding_threshold=config.ce_kl_rounding_threshold,
        )
        # Use ci-masked model to get final label probability
        mask_infos = make_mask_infos(ci_outputs.lower_leaky, routing_masks="all")
        out = model(tokens, mask_infos=mask_infos)
        label_prob = F.softmax(out[0, -1, :], dim=-1)[label_token]

    final_metrics = {**l0_stats, **ce_kl_stats, "ci_masked_label_prob": label_prob.item()}
    print(f"\nFinal metrics after {config.steps} steps:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.6f}")

    # Save results
    out_dir = get_out_dir()
    output_path = out_dir / f"optimized_ci_{loaded.wandb_id}.json"

    output_data = {
        "config": config.model_dump(),
        "prompt": config.prompt,
        "token_strings": token_strings,
        "optimized_ci": {k: v[0].cpu().tolist() for k, v in ci_outputs.lower_leaky.items()},
        "wandb_id": loaded.wandb_id,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved optimized CI values to {output_path}")

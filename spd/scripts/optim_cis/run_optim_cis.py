# %%
"""Optimize CI values for a single prompt while keeping component weights fixed."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim as optim
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import (
    ImportanceMinimalityLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from spd.losses import compute_total_loss
from spd.metrics.ce_and_kl_losses import CEandKLLosses
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.scripts.model_loading import load_model_from_wandb
from spd.scripts.optim_cis.config import OptimCIConfig
from spd.utils.component_utils import calc_ci_l_zero
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


def optimize_ci_values(
    model: ComponentModel,
    tokens: Tensor,
    config: OptimCIConfig,
    device: str,
) -> tuple[dict[str, list[list[float]]], dict[str, float]]:
    """Optimize CI values for a single prompt.

    Args:
        model: The ComponentModel (weights will be frozen).
        tokens: Tokenized prompt of shape [1, seq_len].
        config: Optimization configuration.
        device: Device to run on.

    Returns:
        Tuple of:
        - Optimized CI values as dict of layer_name -> [seq][C] nested lists
        - Final metrics dict
    """
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

    # Get weight deltas for losses that need them
    weight_deltas = model.calc_weight_deltas()

    # Setup optimizer
    params = ci_params.get_parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    # Optimization loop
    final_metrics: dict[str, float] = {}

    for step in tqdm(range(config.steps), desc="Optimizing CI values"):
        optimizer.zero_grad()

        # Create CI outputs from current parameters
        ci_outputs = ci_params.create_ci_outputs(model, device)

        # Compute losses
        total_loss, loss_terms = compute_total_loss(
            loss_metric_configs=config.loss_metric_configs,
            model=model,
            batch=tokens,
            ci=ci_outputs,
            target_out=target_out,
            weight_deltas=weight_deltas,
            pre_weight_acts=output_with_cache.cache,
            current_frac_of_training=step / config.steps,
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            n_mask_samples=config.n_mask_samples,
            output_loss_type=config.output_loss_type,
        )

        total_loss.backward()
        optimizer.step()

        # Logging
        if step % config.log_freq == 0 or step == config.steps - 1:
            l0_stats = compute_l0_stats(ci_outputs, config.ci_threshold)

            # Compute CE/KL metrics
            ce_kl_metric = CEandKLLosses(
                model=model,
                device=device,
                sampling=config.sampling,
                rounding_threshold=config.ce_kl_rounding_threshold,
            )
            with torch.no_grad():
                ce_kl_metric.update(batch=tokens, target_out=target_out, ci=ci_outputs)
            ce_kl_stats = ce_kl_metric.compute()

            tqdm.write(f"\n--- Step {step} ---")
            for name, value in loss_terms.items():
                tqdm.write(f"  {name}: {value:.6f}")
            for name, value in l0_stats.items():
                tqdm.write(f"  {name}: {value:.2f}")
            for name, value in ce_kl_stats.items():
                tqdm.write(f"  {name}: {value:.6f}")

            if step == config.steps - 1:
                final_metrics = {**loss_terms, **l0_stats, **ce_kl_stats}

    # Extract final CI values
    with torch.no_grad():
        final_ci_outputs = ci_params.create_ci_outputs(model, device)

    # Convert to nested lists for JSON serialization
    optimized_ci: dict[str, list[list[float]]] = {}
    for layer_name, ci_tensor in final_ci_outputs.lower_leaky.items():
        # ci_tensor is [1, seq, C], convert to [seq][C]
        optimized_ci[layer_name] = ci_tensor[0].cpu().tolist()

    return optimized_ci, final_metrics


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
        lr=1e-3,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        steps=10000,
        log_freq=500,
        loss_metric_configs=[
            StochasticReconSubsetLossConfig(coeff=1.0, routing=UniformKSubsetRoutingConfig()),
            ImportanceMinimalityLossConfig(coeff=6e-2, pnorm=0.3),
        ],
        ci_threshold=1e-6,
        sampling="continuous",
        n_mask_samples=1,
        output_loss_type="kl",
        use_delta_component=True,
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

    # Run optimization
    optimized_ci, final_metrics = optimize_ci_values(
        model=model,
        tokens=tokens,
        config=config,
        device=device,
    )

    # Save results
    out_dir = get_out_dir()
    output_path = out_dir / f"optimized_ci_{loaded.wandb_id}.json"

    output_data = {
        "config": config.model_dump(),
        "prompt": config.prompt,
        "token_strings": token_strings,
        "optimized_ci": optimized_ci,
        "final_metrics": final_metrics,
        "wandb_id": loaded.wandb_id,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved optimized CI values to {output_path}")

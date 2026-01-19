"""Train a Sparse Autoencoder with stochastic CI-based sparsity instead of BatchTopK.

Instead of using TopK selection for sparsity, we use:
- A CI (Causal Importance) function to predict importance for each latent
- Bernoulli sampling based on CI values during training
- Importance minimality loss to encourage sparsity

This tests whether stochastic ablation (from SPD) can improve SAE training.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb

# Add batchtopk to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "batchtopk"))

from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer

from logs import init_wandb, log_model_performance, save_checkpoint
from spd.metrics.importance_minimality_loss import importance_minimality_loss
from spd.models.components import LinearCiFn, MLPCiFn, VectorSharedMLPCiFn
from spd.models.sigmoids import SIGMOID_TYPES


class SharedScalarMLPCiFn(nn.Module):
    """Maps each component's scalar activation to CI using a single shared MLP.

    Unlike MLPCiFn which has separate per-component MLPs, this uses one MLP
    shared across all components. Each latent activation is processed independently
    through the same scalar -> scalar network.
    """

    def __init__(self, hidden_dims: list[int], chunk_size: int = 65536):
        super().__init__()
        self.chunk_size = chunk_size
        layers: list[nn.Module] = []
        in_dim = 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map latent activations to CI values.

        Args:
            x: Latent activations (batch, C)

        Returns:
            CI pre-sigmoid values (batch, C)
        """
        batch, C = x.shape
        x_flat = x.reshape(batch * C, 1)

        # Process in chunks to avoid OOM with large batch * C
        if x_flat.shape[0] <= self.chunk_size:
            out_flat = self.mlp(x_flat)
        else:
            chunks = []
            for i in range(0, x_flat.shape[0], self.chunk_size):
                chunk = x_flat[i : i + self.chunk_size]
                chunks.append(self.mlp(chunk))
            out_flat = torch.cat(chunks, dim=0)

        return out_flat.reshape(batch, C)


class StochasticSAE(nn.Module):
    """Sparse Autoencoder with stochastic CI-based sparsity.

    Instead of TopK selection:
    - Encoder produces latent activations (no ReLU)
    - CI function predicts importance [0,1] for each latent
    - During training: Bernoulli sample mask based on CI
    - During eval: use soft CI values as mask
    - Decoder reconstructs from masked latents
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])

        # Standard SAE parameters
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"]))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"]))
        )
        # Initialize decoder as transpose of encoder, normalized
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        # CI function
        self.ci_fn_type = cfg["ci_fn_type"]
        if cfg["ci_fn_type"] == "linear":
            self.ci_fn = LinearCiFn(C=cfg["dict_size"])
        elif cfg["ci_fn_type"] == "mlp":
            self.ci_fn = MLPCiFn(C=cfg["dict_size"], hidden_dims=cfg["ci_fn_hidden_dims"])
        elif cfg["ci_fn_type"] == "shared_mlp":
            self.ci_fn = VectorSharedMLPCiFn(
                C=cfg["dict_size"],
                input_dim=cfg["act_size"],
                hidden_dims=cfg["ci_fn_hidden_dims"],
            )
        elif cfg["ci_fn_type"] == "shared_scalar_mlp":
            self.ci_fn = SharedScalarMLPCiFn(hidden_dims=cfg["ci_fn_hidden_dims"])
        else:
            raise ValueError(f"Unknown ci_fn_type: {cfg['ci_fn_type']}")

        # Sigmoid functions
        if cfg.get("use_normal_sigmoid", False):
            self.lower_leaky_fn = torch.sigmoid
            self.upper_leaky_fn = torch.sigmoid
        else:
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]

        # Track dead features
        self.num_batches_not_active = torch.zeros(cfg["dict_size"], device=cfg["device"])

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(
        self, x_reconstruct: torch.Tensor, x_mean: torch.Tensor | None, x_std: torch.Tensor | None
    ) -> torch.Tensor:
        if self.cfg.get("input_unit_norm", False) and x_mean is not None and x_std is not None:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Normalize decoder weights to unit norm and project out radial gradient component."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, mask: torch.Tensor) -> None:
        """Track features that haven't been active (mask > 0.5) recently."""
        active = (mask > 0.5).any(dim=0)
        self.num_batches_not_active = torch.where(
            active,
            torch.zeros_like(self.num_batches_not_active),
            self.num_batches_not_active + 1,
        )

    def forward(
        self, x: torch.Tensor, current_step: int = 0, total_steps: int = 1
    ) -> dict[str, torch.Tensor]:
        """Forward pass with stochastic CI-based masking.

        Args:
            x: Input activations (batch_size, act_size)
            current_step: Current training step (for p-norm annealing). Default 0.
            total_steps: Total training steps (for p-norm annealing). Default 1.

        Returns:
            Dictionary with loss terms and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)

        # Center and encode (no ReLU - CI provides sparsity)
        x_cent = x - self.b_dec
        latents = x_cent @ self.W_enc  # (batch, dict_size)

        # Compute CI values
        if self.ci_fn_type == "shared_mlp":
            ci_pre_sigmoid = self.ci_fn(x_cent)  # (batch, dict_size)
        else:
            ci_pre_sigmoid = self.ci_fn(latents)  # (batch, dict_size)

        # Apply sigmoids
        ci_lower = self.lower_leaky_fn(ci_pre_sigmoid)  # for masking
        ci_upper = self.upper_leaky_fn(ci_pre_sigmoid)  # for loss

        # Sample mask during training, use soft values during eval
        if self.training and self.cfg["sampling"] == "binomial":
            # SPD-style differentiable stochastic masking
            # mask = ci + (1 - ci) * stochastic_source, where stochastic_source is 0 or 1
            # This keeps gradients flowing through ci
            stochastic_source = torch.randint(0, 2, ci_lower.shape, device=ci_lower.device).float()
            mask = ci_lower + (1 - ci_lower) * stochastic_source
        elif self.training and self.cfg["sampling"] == "continuous":
            # Continuous version: stochastic_source is Uniform(0, 1)
            stochastic_source = torch.rand_like(ci_lower)
            mask = ci_lower + (1 - ci_lower) * stochastic_source
        else:
            mask = ci_lower

        # Apply mask and decode
        masked_latents = latents * mask
        x_reconstruct = masked_latents @ self.W_dec + self.b_dec

        # Update dead feature tracking
        self.update_inactive_features(mask)

        # Compute losses
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        # Importance minimality loss
        current_frac = current_step / max(total_steps, 1)
        im_loss = importance_minimality_loss(
            ci_upper_leaky={"encoder": ci_upper},
            current_frac_of_training=current_frac,
            eps=1e-8,
            pnorm=self.cfg["pnorm"],
            p_anneal_start_frac=self.cfg["p_anneal_start_frac"],
            p_anneal_final_p=self.cfg["p_anneal_final_p"],
            p_anneal_end_frac=self.cfg["p_anneal_end_frac"],
        )

        # Max ablation loss: use soft CI values as mask
        max_ablate_reconstruct = (latents * ci_lower) @ self.W_dec + self.b_dec
        l2_loss_max_ablate = (max_ablate_reconstruct.float() - x.float()).pow(2).mean()

        # No ablation loss: all features active (mask = 1)
        no_ablate_reconstruct = latents @ self.W_dec + self.b_dec
        l2_loss_no_ablate = (no_ablate_reconstruct.float() - x.float()).pow(2).mean()

        # Total loss
        loss = (
            l2_loss
            + self.cfg["importance_coeff"] * im_loss
            + self.cfg["max_ablate_coeff"] * l2_loss_max_ablate
            + self.cfg["no_ablate_coeff"] * l2_loss_no_ablate
        )

        # Metrics
        l0_norm = (mask > 0.5).float().sum(-1).mean()  # average active features per sample
        mean_ci = ci_upper.mean()
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        return {
            "sae_out": sae_out,
            "feature_acts": masked_latents,
            "loss": loss,
            "l2_loss": l2_loss,
            "im_loss": im_loss,
            "l0_norm": l0_norm,
            "mean_ci": mean_ci,
            "num_dead_features": num_dead_features,
            "l2_loss_max_ablate": l2_loss_max_ablate,
            "l2_loss_no_ablate": l2_loss_no_ablate,
            # For compatibility with batchtopk logging
            "l1_loss": im_loss * self.cfg["importance_coeff"],
            "l1_norm": mean_ci,
        }


def get_stochastic_sae_cfg() -> dict:
    """Get default config with stochastic SAE specific settings."""
    cfg = get_default_cfg()

    # Override SAE type
    cfg["sae_type"] = "stochastic_ci"

    # Stochastic CI specific settings
    cfg["ci_fn_type"] = "shared_mlp"  # or "mlp"
    cfg["ci_fn_hidden_dims"] = [32]
    cfg["importance_coeff"] = 1e-9  # weight for importance minimality loss
    cfg["max_ablate_coeff"] = 0.0  # weight for max ablation L2 loss (soft CI mask)
    cfg["no_ablate_coeff"] = 0.0  # weight for no ablation L2 loss (all features active)
    cfg["pnorm"] = 1.0  # L_p norm for sparsity
    cfg["p_anneal_start_frac"] = 0.0  # when to start annealing p
    cfg["p_anneal_final_p"] = 0.5  # final p value
    cfg["p_anneal_end_frac"] = 0.5  # when to finish annealing p
    cfg["sampling"] = "binomial"  # "binomial", "continuous", or "none" (deterministic)
    cfg["use_normal_sigmoid"] = False  # True for normal sigmoid, False for leaky hard

    # Remove unused BatchTopK settings (keep for compatibility but they're not used)
    cfg["l1_coeff"] = 0  # Not used - we use importance_coeff instead

    return cfg


@torch.no_grad()
def log_ablation_performance(
    wandb_run,
    step: int,
    model: HookedTransformer,
    activation_store: ActivationsStore,
    sae: StochasticSAE,
) -> None:
    """Log model performance metrics under different ablation settings.

    Computes CE loss when using SAE with:
    - Max ablation (soft CI mask)
    - No ablation (all features active)
    """
    from functools import partial

    cfg = sae.cfg
    batch_tokens = activation_store.get_batch_tokens()[: cfg["batch_size"] // cfg["seq_len"]]
    batch = activation_store.get_activations(batch_tokens).reshape(-1, cfg["act_size"])

    # Preprocess
    x = batch
    if cfg.get("input_unit_norm", False):
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x = x / (x_std + 1e-5)
    else:
        x_mean, x_std = None, None

    # Encode
    x_cent = x - sae.b_dec
    latents = x_cent @ sae.W_enc

    # Compute CI
    ci_pre_sigmoid = sae.ci_fn(x_cent) if sae.ci_fn_type == "shared_mlp" else sae.ci_fn(latents)
    ci_lower = sae.lower_leaky_fn(ci_pre_sigmoid)

    # Max ablation reconstruction (soft CI mask)
    max_ablate_latents = latents * ci_lower
    max_ablate_out = max_ablate_latents @ sae.W_dec + sae.b_dec
    if x_mean is not None and x_std is not None:
        max_ablate_out = max_ablate_out * x_std + x_mean
    max_ablate_out = max_ablate_out.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    # No ablation reconstruction (all features active)
    no_ablate_out = latents @ sae.W_dec + sae.b_dec
    if x_mean is not None and x_std is not None:
        no_ablate_out = no_ablate_out * x_std + x_mean
    no_ablate_out = no_ablate_out.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    # Hook for replacing activations
    def reconstr_hook(activation, hook, sae_out):
        return sae_out

    # Compute losses
    original_loss = model(batch_tokens, return_type="loss").item()

    max_ablate_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(cfg["hook_point"], partial(reconstr_hook, sae_out=max_ablate_out))],
        return_type="loss",
    ).item()

    no_ablate_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(cfg["hook_point"], partial(reconstr_hook, sae_out=no_ablate_out))],
        return_type="loss",
    ).item()

    # Log metrics
    log_dict = {
        "ablation_perf/original_ce": original_loss,
        "ablation_perf/max_ablate_ce": max_ablate_loss,
        "ablation_perf/no_ablate_ce": no_ablate_loss,
        "ablation_perf/max_ablate_ce_delta": max_ablate_loss - original_loss,
        "ablation_perf/no_ablate_ce_delta": no_ablate_loss - original_loss,
    }
    wandb_run.log(log_dict, step=step)


@torch.no_grad()
def log_ci_visualizations(
    wandb_run,
    step: int,
    activation_store: ActivationsStore,
    sae: StochasticSAE,
) -> None:
    """Log visualizations of the CI function behavior."""
    cfg = sae.cfg
    batch_tokens = activation_store.get_batch_tokens()[: cfg["batch_size"] // cfg["seq_len"]]
    batch = activation_store.get_activations(batch_tokens).reshape(-1, cfg["act_size"])

    # Preprocess
    x = batch
    if cfg.get("input_unit_norm", False):
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x = x / (x_std + 1e-5)

    # Encode and compute CI
    x_cent = x - sae.b_dec
    latents = x_cent @ sae.W_enc

    ci_pre_sigmoid = sae.ci_fn(x_cent) if sae.ci_fn_type == "shared_mlp" else sae.ci_fn(latents)

    ci_upper = sae.upper_leaky_fn(ci_pre_sigmoid)

    # Move to CPU for plotting
    latents_np = latents.cpu().numpy()
    ci_upper_np = ci_upper.cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # 1. CI distribution histogram
    ax = axes[0, 0]
    ax.hist(ci_upper_np.flatten(), bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("CI Value")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of CI Values")
    ax.axvline(x=0.5, color="r", linestyle="--", label="0.5 threshold")
    ax.legend()

    # 2. Per-feature mean CI (sorted, log scale)
    ax = axes[0, 1]
    mean_ci_per_feature = ci_upper_np.mean(axis=0)
    sorted_mean_ci = np.sort(mean_ci_per_feature)[::-1]
    ax.semilogy(sorted_mean_ci + 1e-10)
    ax.set_xlabel("Feature Index (sorted by CI)")
    ax.set_ylabel("Mean CI (log scale)")
    ax.set_title(
        f"Per-Feature Mean CI (sorted)\n{(mean_ci_per_feature > 0.5).sum()}/{len(mean_ci_per_feature)} features > 0.5"
    )
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    # 3. CI vs Activation magnitude scatter (subsample for speed)
    ax = axes[0, 2]
    n_points = min(10000, latents_np.size)
    flat_latents = latents_np.flatten()
    flat_ci = ci_upper_np.flatten()
    indices = np.random.choice(len(flat_latents), n_points, replace=False)
    ax.scatter(flat_latents[indices], flat_ci[indices], alpha=0.1, s=1)
    ax.set_xlabel("Latent Activation")
    ax.set_ylabel("CI Value")
    ax.set_title("CI vs Latent Activation")

    # 4. Learned CI function (for scalar MLPs)
    ax = axes[1, 0]
    if sae.ci_fn_type in ["linear", "mlp", "shared_scalar_mlp"]:
        # Sample activation range and plot CI function
        act_range = np.linspace(float(latents.min().cpu()), float(latents.max().cpu()), 200)
        act_tensor = torch.tensor(act_range, dtype=sae.W_enc.dtype, device=sae.W_enc.device)

        if sae.ci_fn_type == "shared_scalar_mlp":
            # Process through the shared scalar MLP
            ci_out = sae.ci_fn.mlp(act_tensor.unsqueeze(1)).squeeze(1)
        else:
            # For linear/mlp, need to handle per-component - just show a few
            act_expanded = act_tensor.unsqueeze(1).expand(-1, min(5, cfg["dict_size"]))
            ci_out = sae.ci_fn(act_expanded)[:, 0]  # Just show first component

        ci_out_sigmoid = sae.upper_leaky_fn(ci_out).cpu().numpy()
        ax.plot(act_range, ci_out_sigmoid)
        ax.set_xlabel("Latent Activation")
        ax.set_ylabel("CI (after sigmoid)")
        ax.set_title(f"Learned CI Function ({sae.ci_fn_type})")
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
    else:
        ax.text(
            0.5, 0.5, f"N/A for {sae.ci_fn_type}", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Learned CI Function")

    # 5. Average features above CI threshold per sample
    ax = axes[1, 1]
    thresholds = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    # For each sample, count features with CI > threshold, then average across samples
    avg_features_above = [(ci_upper_np > t).sum(axis=1).mean() for t in thresholds]
    ax.bar([str(t) for t in thresholds], avg_features_above, edgecolor="black")
    ax.set_xlabel("CI Threshold")
    ax.set_ylabel("Avg Features per Sample")
    ax.set_title("Avg Features Above CI Threshold (per sample)")
    for i, c in enumerate(avg_features_above):
        ax.text(i, c + 0.5, f"{c:.1f}", ha="center", va="bottom", fontsize=9)

    # 6. CI percentiles over features
    ax = axes[1, 2]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    ci_percentiles = np.percentile(mean_ci_per_feature, percentiles)
    ax.bar([str(p) for p in percentiles], ci_percentiles, edgecolor="black")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("CI Value")
    ax.set_title("CI Percentiles Across Features")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    # 7. Latent activation distribution histogram
    ax = axes[2, 0]
    ax.hist(flat_latents, bins=100, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Latent Activation Value")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Latent Activations\nmean={flat_latents.mean():.3f}, std={flat_latents.std():.3f}"
    )
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # 8. Per-feature mean activation (sorted)
    ax = axes[2, 1]
    mean_act_per_feature = np.abs(latents_np).mean(axis=0)
    sorted_indices = np.argsort(mean_act_per_feature)[::-1]
    sorted_mean_act = mean_act_per_feature[sorted_indices]
    ax.semilogy(sorted_mean_act + 1e-10)
    ax.set_xlabel("Feature Index (sorted by |activation|)")
    ax.set_ylabel("Mean |Activation| (log scale)")
    ax.set_title("Per-Feature Mean |Activation| (sorted)")

    # 9. Activation sparsity: fraction of near-zero activations per feature
    ax = axes[2, 2]
    # Consider activation "active" if |act| > threshold
    act_threshold = 0.1
    active_frac_per_feature = (np.abs(latents_np) > act_threshold).mean(axis=0)
    sorted_active_frac = np.sort(active_frac_per_feature)[::-1]
    ax.plot(sorted_active_frac)
    ax.set_xlabel("Feature Index (sorted)")
    ax.set_ylabel(f"Fraction |act| > {act_threshold}")
    ax.set_title(f"Feature Activation Frequency\n(threshold={act_threshold})")

    plt.tight_layout()

    # Log to wandb
    wandb_run.log({"ci_plots/overview": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Also log some scalar metrics about CI distribution
    wandb_run.log(
        {
            "ci_stats/mean": float(ci_upper_np.mean()),
            "ci_stats/std": float(ci_upper_np.std()),
            "ci_stats/median": float(np.median(ci_upper_np)),
            "ci_stats/min": float(ci_upper_np.min()),
            "ci_stats/max": float(ci_upper_np.max()),
            "ci_stats/pct_above_0.5": float((ci_upper_np > 0.5).mean() * 100),
            "ci_stats/features_above_0.5": int((mean_ci_per_feature > 0.5).sum()),
            "ci_stats/features_above_0.1": int((mean_ci_per_feature > 0.1).sum()),
            # Latent activation stats
            "latent_stats/mean": float(flat_latents.mean()),
            "latent_stats/std": float(flat_latents.std()),
            "latent_stats/abs_mean": float(np.abs(flat_latents).mean()),
            "latent_stats/min": float(flat_latents.min()),
            "latent_stats/max": float(flat_latents.max()),
            "latent_stats/pct_positive": float((flat_latents > 0).mean() * 100),
            "latent_stats/sparsity": float((np.abs(flat_latents) < 0.1).mean() * 100),
        },
        step=step,
    )


def train_stochastic_sae(
    sae: StochasticSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
    cfg: dict,
) -> None:
    """Train the stochastic SAE."""
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch, current_step=i, total_steps=num_batches)

        # Log to wandb
        log_dict = {
            "loss": sae_output["loss"].item(),
            "l2_loss": sae_output["l2_loss"].item(),
            "im_loss": sae_output["im_loss"].item(),
            "l0_norm": sae_output["l0_norm"].item(),
            "mean_ci": sae_output["mean_ci"].item(),
            "num_dead_features": sae_output["num_dead_features"].item(),
            "n_dead_in_batch": (sae_output["feature_acts"].sum(0) == 0).sum().item(),
            "l2_loss_max_ablate": sae_output["l2_loss_max_ablate"].item(),
            "l2_loss_no_ablate": sae_output["l2_loss_no_ablate"].item(),
        }
        wandb_run.log(log_dict, step=i)

        if i % cfg["perf_log_freq"] == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)
            log_ablation_performance(wandb_run, i, model, activation_store, sae)
            log_ci_visualizations(wandb_run, i, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm'].item():.1f}",
                "L2": f"{sae_output['l2_loss'].item():.4f}",
                "IM": f"{sae_output['im_loss'].item():.4f}",
                "CI": f"{sae_output['mean_ci'].item():.3f}",
            }
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, num_batches)


def main(
    # Training settings
    num_tokens: int = 100_000_000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    # Model settings
    model_name: str = "gpt2-small",
    layer: int = 8,
    site: str = "resid_pre",
    dict_size: int = 768 * 4,  # 4x expansion
    # CI settings
    ci_fn_type: str = "shared_scalar_mlp",
    importance_coeff: float = 2e-3,
    max_ablate_coeff: float = 1.0,
    no_ablate_coeff: float = 1.0,
    pnorm: float = 2.0,
    p_anneal_start_frac: float = 0.0,
    p_anneal_final_p: float = 0.3,
    p_anneal_end_frac: float = 0.5,
    sampling: str = "continuous",
    use_normal_sigmoid: bool = False,
    # Other
    seed: int = 42,
    device: str = "cuda:0",
    wandb_project: str = "stochastic_sae",
) -> None:
    """Run stochastic SAE training experiment.

    Args:
        num_tokens: Total tokens to train on
        batch_size: Batch size for training
        lr: Learning rate
        model_name: Transformer model name
        layer: Layer to extract activations from
        site: Hook site (resid_pre, resid_post, etc.)
        dict_size: SAE dictionary size (number of features)
        ci_fn_type: CI function type ("linear", "mlp", "shared_mlp", or "shared_scalar_mlp")
        importance_coeff: Weight for importance minimality loss
        max_ablate_coeff: Weight for max ablation L2 loss (soft CI mask)
        no_ablate_coeff: Weight for no ablation L2 loss (all features active)
        pnorm: Initial p-norm for sparsity
        p_anneal_start_frac: When to start annealing p (fraction of training)
        p_anneal_final_p: Final p value after annealing
        p_anneal_end_frac: When to finish annealing p
        sampling: "binomial", "continuous", or "none" (deterministic)
        use_normal_sigmoid: Use normal sigmoid instead of leaky hard
        seed: Random seed
        device: Device to train on
        wandb_project: W&B project name
    """
    # Build config
    cfg = get_stochastic_sae_cfg()
    cfg["seed"] = seed
    cfg["num_tokens"] = num_tokens
    cfg["batch_size"] = batch_size
    cfg["lr"] = lr
    cfg["model_name"] = model_name
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["dict_size"] = dict_size
    cfg["ci_fn_type"] = ci_fn_type
    cfg["importance_coeff"] = importance_coeff
    cfg["max_ablate_coeff"] = max_ablate_coeff
    cfg["no_ablate_coeff"] = no_ablate_coeff
    cfg["pnorm"] = pnorm
    cfg["p_anneal_start_frac"] = p_anneal_start_frac
    cfg["p_anneal_final_p"] = p_anneal_final_p
    cfg["p_anneal_end_frac"] = p_anneal_end_frac
    cfg["sampling"] = sampling
    cfg["use_normal_sigmoid"] = use_normal_sigmoid
    cfg["device"] = device
    cfg["wandb_project"] = wandb_project
    cfg = post_init_cfg(cfg)

    print(f"Training Stochastic CI SAE on {model_name} layer {layer}")
    print(f"  dict_size: {dict_size}")
    print(f"  ci_fn_type: {ci_fn_type}")
    print(f"  importance_coeff: {importance_coeff}")
    print(f"  pnorm: {pnorm} -> {p_anneal_final_p}")
    print(f"  sampling: {sampling}")
    print(f"  num_tokens: {num_tokens:,}")

    # Load model
    model = HookedTransformer.from_pretrained(model_name).to(device)
    cfg["act_size"] = model.cfg.d_model

    # Create activation store
    activation_store = ActivationsStore(model, cfg)

    # Create SAE
    sae = StochasticSAE(cfg)
    print(f"  SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Train
    train_stochastic_sae(sae, activation_store, model, cfg)

    print("Training complete!")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

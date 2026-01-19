"""Train a Sparse Autoencoder with AbsTopK selection and stochastic K sampling.

Key differences from standard TopKSAE:
1. No ReLU - use raw latent activations (can be negative)
2. AbsTopK - select TopK based on absolute values, not just values
3. Stochastic K - sample K uniformly from [k_min, k_max] on each forward pass
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


class AbsTopKSAE(nn.Module):
    """Sparse Autoencoder with AbsTopK selection and stochastic K.

    Key features:
    - No ReLU: latents can be positive or negative
    - AbsTopK: select top K features by absolute value
    - Stochastic K: sample K from [k_min, k_max] each forward pass
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

    def update_inactive_features(self, acts: torch.Tensor) -> None:
        """Track features that haven't been active recently."""
        active = (acts != 0).any(dim=0)
        self.num_batches_not_active = torch.where(
            active,
            torch.zeros_like(self.num_batches_not_active),
            self.num_batches_not_active + 1,
        )

    def sample_k(self, batch_size: int) -> torch.Tensor:
        """Sample K log-uniformly from [k_min, k_max] for each sample in batch."""
        import math

        k_min = self.cfg["k_min"]
        k_max = self.cfg["k_max"]
        log2_min = math.log2(max(k_min, 1))
        log2_max = math.log2(k_max)

        log2_k = torch.empty(batch_size, device=self.W_enc.device).uniform_(log2_min, log2_max)
        k_values = (2**log2_k).round().long().clamp(min=k_min, max=k_max)
        return k_values

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with AbsTopK selection and stochastic per-sample K."""
        x, x_mean, x_std = self.preprocess_input(x)
        batch_size = x.shape[0]

        # Center and encode (NO ReLU - latents can be negative)
        x_cent = x - self.b_dec
        latents = x_cent @ self.W_enc  # (batch, dict_size)

        # AbsTopK selection
        abs_latents = latents.abs()

        if self.training:
            # Per-sample K: use topk with max_k, then mask
            k_values = self.sample_k(batch_size)
            max_k = k_values.max().item()

            _, topk_indices = torch.topk(abs_latents, max_k, dim=-1)
            topk_signed = latents.gather(-1, topk_indices)

            # Mask: keep first k_values[i] entries for each sample
            positions = torch.arange(max_k, device=latents.device)
            keep_mask = positions < k_values.unsqueeze(1)
            topk_signed = topk_signed * keep_mask

            # Scatter back
            acts_topk = torch.zeros_like(latents)
            acts_topk.scatter_(-1, topk_indices, topk_signed)
            avg_k = k_values.float().mean().item()
        else:
            # Fixed K for evaluation
            k = self.cfg.get("eval_k", (self.cfg["k_min"] + self.cfg["k_max"]) // 2)
            _, topk_indices = torch.topk(abs_latents, k, dim=-1)
            acts_topk = torch.zeros_like(latents)
            acts_topk.scatter_(-1, topk_indices, latents.gather(-1, topk_indices))
            avg_k = k

        # Decode
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        # Losses
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm

        # Full reconstruction loss (all features active) - ensures all features get gradients
        x_reconstruct_full = latents @ self.W_dec + self.b_dec
        l2_loss_full_raw = (x_reconstruct_full.float() - x.float()).pow(2).mean()
        full_coeff = self.cfg.get("full_recon_coeff", 0.0)
        l2_loss_full = full_coeff * l2_loss_full_raw

        self.update_inactive_features(acts_topk)

        # Auxiliary loss for dead features
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, latents)

        # Total loss
        loss = l2_loss + l1_loss + aux_loss + l2_loss_full

        # Metrics
        l0_norm = (acts_topk != 0).float().sum(-1).mean()
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        return {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "loss": loss,
            "l2_loss": l2_loss,
            "l2_loss_full_raw": l2_loss_full_raw,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": acts_topk.float().abs().sum(-1).mean(),
            "aux_loss": aux_loss,
            "num_dead_features": num_dead_features,
            "k": torch.tensor(avg_k, dtype=torch.float32),
        }

    def get_auxiliary_loss(
        self, x: torch.Tensor, x_reconstruct: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        """Auxiliary loss to revive dead features."""
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0 and self.cfg.get("aux_penalty", 0) > 0:
            residual = x.float() - x_reconstruct.float()

            # Get top-k of dead features by absolute value
            dead_latents = latents[:, dead_features]
            k_aux = min(self.cfg.get("top_k_aux", 32), dead_features.sum().item())

            if k_aux > 0:
                abs_dead = dead_latents.abs()
                topk_aux = torch.topk(abs_dead, k_aux, dim=-1)
                acts_aux = torch.zeros_like(dead_latents).scatter(
                    -1, topk_aux.indices, dead_latents.gather(-1, topk_aux.indices)
                )
                x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
                l2_loss_aux = (
                    self.cfg["aux_penalty"]
                    * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
                )
                return l2_loss_aux

        return torch.tensor(0, dtype=x.dtype, device=x.device)


def get_abs_topk_cfg() -> dict:
    """Get default config for AbsTopK SAE."""
    cfg = get_default_cfg()

    cfg["sae_type"] = "abs_topk"

    # K range for stochastic sampling
    cfg["k_min"] = 1
    cfg["k_max"] = 64  # Will be set based on dict_size if not specified
    cfg["eval_k"] = 32  # K to use during evaluation

    # Standard SAE settings
    cfg["l1_coeff"] = 0.0  # Can add L1 regularization if desired
    cfg["aux_penalty"] = 1 / 32  # Auxiliary loss for dead features
    cfg["top_k_aux"] = 32
    cfg["full_recon_coeff"] = 0.1  # Coefficient for full reconstruction loss (all features)

    return cfg


@torch.no_grad()
def log_k_sweep_performance(
    wandb_run,
    step: int,
    model: HookedTransformer,
    activation_store: ActivationsStore,
    sae: AbsTopKSAE,
) -> None:
    """Log CE degradation for different K values."""
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

    # Encode (no ReLU)
    x_cent = x - sae.b_dec
    latents = x_cent @ sae.W_enc

    # Hook for replacing activations
    def reconstr_hook(activation, hook, sae_out):
        return sae_out

    # Get original loss
    original_loss = model(batch_tokens, return_type="loss").item()

    # Test different K values (powers of 2 plus eval_k)
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    k_values = [k for k in k_values if k <= cfg["dict_size"]]
    if cfg["eval_k"] not in k_values:
        k_values.append(cfg["eval_k"])
    k_values = sorted(set(k_values))

    ce_deltas = []
    l2_losses = []

    for k in k_values:
        # AbsTopK selection
        abs_latents = latents.abs()
        topk_values, topk_indices = torch.topk(abs_latents, k, dim=-1)
        acts_topk = torch.zeros_like(latents)
        acts_topk.scatter_(-1, topk_indices, latents.gather(-1, topk_indices))

        # Reconstruct
        x_reconstruct = acts_topk @ sae.W_dec + sae.b_dec

        # L2 loss
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean().item()
        l2_losses.append(l2_loss)

        # Postprocess for model
        if x_mean is not None and x_std is not None:
            x_reconstruct = x_reconstruct * x_std + x_mean
        x_reconstruct = x_reconstruct.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

        # CE loss
        reconstr_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(cfg["hook_point"], partial(reconstr_hook, sae_out=x_reconstruct))],
            return_type="loss",
        ).item()

        ce_delta = reconstr_loss - original_loss
        ce_deltas.append(ce_delta)

        # Log individual K values
        wandb_run.log(
            {
                f"k_sweep/ce_delta_k{k}": ce_delta,
                f"k_sweep/l2_loss_k{k}": l2_loss,
            },
            step=step,
        )

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CE degradation vs K
    ax = axes[0]
    ax.plot(k_values, ce_deltas, "o-", linewidth=2, markersize=8)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("K (number of active features)")
    ax.set_ylabel("CE Degradation (higher = worse)")
    ax.set_xscale("log", base=2)
    ax.set_title("CE Degradation vs K")
    ax.grid(True, alpha=0.3)
    # Mark eval_k
    eval_k_idx = k_values.index(cfg["eval_k"])
    ax.scatter(
        [cfg["eval_k"]],
        [ce_deltas[eval_k_idx]],
        color="red",
        s=100,
        zorder=5,
        label=f"eval_k={cfg['eval_k']}",
    )
    ax.legend()

    # L2 loss vs K
    ax = axes[1]
    ax.plot(k_values, l2_losses, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("K (number of active features)")
    ax.set_ylabel("L2 Reconstruction Loss")
    ax.set_xscale("log", base=2)
    ax.set_title("L2 Loss vs K")
    ax.grid(True, alpha=0.3)
    ax.scatter(
        [cfg["eval_k"]],
        [l2_losses[eval_k_idx]],
        color="red",
        s=100,
        zorder=5,
        label=f"eval_k={cfg['eval_k']}",
    )
    ax.legend()

    plt.tight_layout()
    wandb_run.log({"k_sweep/overview": wandb.Image(fig)}, step=step)
    plt.close(fig)


@torch.no_grad()
def log_latent_visualizations(
    wandb_run,
    step: int,
    activation_store: ActivationsStore,
    sae: AbsTopKSAE,
) -> None:
    """Log visualizations of latent activations."""
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

    # Encode (no ReLU)
    x_cent = x - sae.b_dec
    latents = x_cent @ sae.W_enc

    # Move to CPU for plotting
    latents_np = latents.cpu().numpy()
    flat_latents = latents_np.flatten()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Latent activation distribution histogram
    ax = axes[0, 0]
    ax.hist(flat_latents, bins=100, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Latent Activation Value")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Latent Activations\nmean={flat_latents.mean():.3f}, std={flat_latents.std():.3f}"
    )
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # 2. Positive vs negative activation distribution
    ax = axes[0, 1]
    pos_latents = flat_latents[flat_latents > 0]
    neg_latents = flat_latents[flat_latents < 0]
    ax.hist(
        pos_latents,
        bins=50,
        alpha=0.6,
        label=f"Positive ({len(pos_latents) / len(flat_latents) * 100:.1f}%)",
        color="blue",
    )
    ax.hist(
        neg_latents,
        bins=50,
        alpha=0.6,
        label=f"Negative ({len(neg_latents) / len(flat_latents) * 100:.1f}%)",
        color="red",
    )
    ax.set_xlabel("Latent Activation Value")
    ax.set_ylabel("Count")
    ax.set_title("Positive vs Negative Activations")
    ax.legend()

    # 3. Per-feature mean |activation| (sorted, log scale)
    ax = axes[0, 2]
    mean_abs_act_per_feature = np.abs(latents_np).mean(axis=0)
    sorted_mean_act = np.sort(mean_abs_act_per_feature)[::-1]
    ax.semilogy(sorted_mean_act + 1e-10)
    ax.set_xlabel("Feature Index (sorted by |activation|)")
    ax.set_ylabel("Mean |Activation| (log scale)")
    ax.set_title("Per-Feature Mean |Activation|")

    # 4. Feature activation frequency (what fraction of samples have |act| > threshold)
    ax = axes[1, 0]
    act_threshold = 0.1
    active_frac_per_feature = (np.abs(latents_np) > act_threshold).mean(axis=0)
    sorted_active_frac = np.sort(active_frac_per_feature)[::-1]
    ax.plot(sorted_active_frac)
    ax.set_xlabel("Feature Index (sorted)")
    ax.set_ylabel(f"Fraction |act| > {act_threshold}")
    ax.set_title(
        f"Feature Activation Frequency\n{(active_frac_per_feature > 0).sum()} features ever active"
    )

    # 5. Top-K selection frequency: how often each feature is in top-K
    ax = axes[1, 1]
    eval_k = cfg["eval_k"]
    abs_latents = np.abs(latents_np)
    # Get top-k indices for each sample
    topk_indices = np.argpartition(-abs_latents, eval_k, axis=-1)[:, :eval_k]
    # Count how often each feature appears in top-k
    feature_counts = np.zeros(cfg["dict_size"])
    np.add.at(feature_counts, topk_indices.flatten(), 1)
    feature_counts /= len(latents_np)  # Normalize by number of samples
    sorted_counts = np.sort(feature_counts)[::-1]
    ax.semilogy(sorted_counts + 1e-10)
    ax.set_xlabel("Feature Index (sorted)")
    ax.set_ylabel(f"Fraction in Top-{eval_k} (log scale)")
    ax.set_title(
        f"Feature Selection Frequency (K={eval_k})\n{(feature_counts > 0).sum()} features ever selected"
    )

    # 6. Per-feature sign bias: fraction of positive activations
    ax = axes[1, 2]
    pos_frac_per_feature = (latents_np > 0).mean(axis=0)
    ax.hist(pos_frac_per_feature, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Fraction Positive")
    ax.set_ylabel("Number of Features")
    ax.set_title("Per-Feature Sign Bias")
    ax.axvline(x=0.5, color="r", linestyle="--", alpha=0.5, label="50%")
    ax.legend()

    plt.tight_layout()
    wandb_run.log({"latent_plots/overview": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Log scalar stats
    wandb_run.log(
        {
            "latent_stats/mean": float(flat_latents.mean()),
            "latent_stats/std": float(flat_latents.std()),
            "latent_stats/abs_mean": float(np.abs(flat_latents).mean()),
            "latent_stats/min": float(flat_latents.min()),
            "latent_stats/max": float(flat_latents.max()),
            "latent_stats/pct_positive": float((flat_latents > 0).mean() * 100),
            "latent_stats/pct_negative": float((flat_latents < 0).mean() * 100),
            "latent_stats/sparsity": float((np.abs(flat_latents) < 0.1).mean() * 100),
            "latent_stats/features_ever_active": int((active_frac_per_feature > 0).sum()),
            "latent_stats/features_in_topk": int((feature_counts > 0).sum()),
        },
        step=step,
    )


def train_abs_topk_sae(
    sae: AbsTopKSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
    cfg: dict,
) -> None:
    """Train the AbsTopK SAE."""
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch)

        # Log to wandb
        log_dict = {
            "loss": sae_output["loss"].item(),
            "l2_loss": sae_output["l2_loss"].item(),
            "l2_loss_full": sae_output["l2_loss_full_raw"].item(),
            "l1_loss": sae_output["l1_loss"].item(),
            "l0_norm": sae_output["l0_norm"].item(),
            "l1_norm": sae_output["l1_norm"].item(),
            "aux_loss": sae_output["aux_loss"].item(),
            "num_dead_features": sae_output["num_dead_features"].item(),
            "k": sae_output["k"].item(),
            "n_dead_in_batch": (sae_output["feature_acts"].sum(0) == 0).sum().item(),
        }
        wandb_run.log(log_dict, step=i)

        if i % cfg["perf_log_freq"] == 0:
            sae.eval()
            log_model_performance(wandb_run, i, model, activation_store, sae)
            log_k_sweep_performance(wandb_run, i, model, activation_store, sae)
            log_latent_visualizations(wandb_run, i, activation_store, sae)
            sae.train()

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L2": f"{sae_output['l2_loss'].item():.4f}",
                "L0": f"{sae_output['l0_norm'].item():.1f}",
                "K": f"{sae_output['k'].item():.0f}",
                "Dead": f"{sae_output['num_dead_features'].item():.0f}",
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
    # AbsTopK settings
    k_min: int = 32,
    k_max: int = 32,
    eval_k: int = 32,
    l1_coeff: float = 0.0,
    aux_penalty: float = (1 / 32),
    full_recon_coeff: float = 5e-3,
    # Other
    seed: int = 42,
    device: str = "cuda:0",
    wandb_project: str = "abs_topk_sae",
) -> None:
    """Run AbsTopK SAE training experiment.

    Args:
        num_tokens: Total tokens to train on
        batch_size: Batch size for training
        lr: Learning rate
        model_name: Transformer model name
        layer: Layer to extract activations from
        site: Hook site (resid_pre, resid_post, etc.)
        dict_size: SAE dictionary size (number of features)
        k_min: Minimum K for stochastic sampling
        k_max: Maximum K for stochastic sampling
        eval_k: K to use during evaluation
        l1_coeff: L1 regularization coefficient
        aux_penalty: Auxiliary loss penalty for dead features
        full_recon_coeff: Coefficient for full reconstruction loss (all features active)
        seed: Random seed
        device: Device to train on
        wandb_project: W&B project name
    """
    # Build config
    cfg = get_abs_topk_cfg()
    cfg["seed"] = seed
    cfg["num_tokens"] = num_tokens
    cfg["batch_size"] = batch_size
    cfg["lr"] = lr
    cfg["model_name"] = model_name
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["dict_size"] = dict_size
    cfg["k_min"] = k_min
    cfg["k_max"] = k_max
    cfg["eval_k"] = eval_k
    cfg["l1_coeff"] = l1_coeff
    cfg["aux_penalty"] = aux_penalty
    cfg["full_recon_coeff"] = full_recon_coeff
    cfg["device"] = device
    cfg["wandb_project"] = wandb_project
    cfg = post_init_cfg(cfg)

    print(f"Training AbsTopK SAE on {model_name} layer {layer}")
    print(f"  dict_size: {dict_size}")
    print(f"  k_min: {k_min}, k_max: {k_max}, eval_k: {eval_k}")
    print(f"  l1_coeff: {l1_coeff}")
    print(f"  aux_penalty: {aux_penalty}")
    print(f"  num_tokens: {num_tokens:,}")

    # Load model
    model = HookedTransformer.from_pretrained(model_name).to(device)
    cfg["act_size"] = model.cfg.d_model

    # Create activation store
    activation_store = ActivationsStore(model, cfg)

    # Create SAE
    sae = AbsTopKSAE(cfg)
    print(f"  SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Train
    train_abs_topk_sae(sae, activation_store, model, cfg)

    print("Training complete!")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

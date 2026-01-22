"""
Anthropic's JumpReLU SAE implementation based on:
"Dictionary Learning Optimization Techniques" (January 2025)
https://transformer-circuits.pub/2025/january-update/index.html#DL

Key techniques:
1. JumpReLU with straight-through estimator (gradients flow to all params)
2. Tanh sparsity penalty instead of L0
3. Pre-act loss to reduce dead features
4. Specific initialization (especially b_enc based on target activation rate)
5. Lambda_S warmup over entire training
6. LR decay over last 20% of training
7. Post-training decoder normalization
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import tqdm
import wandb

# Add batchtopk to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "batchtopk"))

from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from sae import BatchTopKSAE
from transformer_lens import HookedTransformer

from logs import init_wandb, log_model_performance, save_checkpoint


class RectangleFunction(autograd.Function):
    """Rectangle function for straight-through estimator gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Gradient is 1 inside the rectangle, 0 outside
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    """
    JumpReLU activation with straight-through estimator.

    Forward: x * (x > exp(t))
    Backward for x: 1 if x > exp(t), else 0
    Backward for t: -exp(t)/epsilon * rect((x - exp(t))/epsilon)

    Unlike Rajamanoharan et al., we allow gradients to flow through the
    straight-through estimator to ALL model parameters, not just thresholds.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, epsilon):
        threshold = torch.exp(log_threshold)
        mask = (x > threshold).float()
        ctx.save_for_backward(x, log_threshold, mask)
        ctx.epsilon = epsilon
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, mask = ctx.saved_tensors
        epsilon = ctx.epsilon
        threshold = torch.exp(log_threshold)

        # Gradient for x: straight-through (1 if active, 0 otherwise)
        # But we allow gradient to flow through to all params
        x_grad = mask * grad_output

        # Gradient for threshold via straight-through estimator
        # -exp(t)/epsilon when |x - exp(t)| < epsilon/2
        normalized_diff = (x - threshold) / epsilon
        rect_mask = RectangleFunction.apply(normalized_diff)
        threshold_grad = -(threshold / epsilon) * rect_mask * grad_output
        # Sum over batch dimension to get per-feature gradient
        threshold_grad = threshold_grad.sum(dim=0)

        return x_grad, threshold_grad, None


class AnthropicJumpReLUSAE(nn.Module):
    """
    Anthropic's JumpReLU SAE with all optimization tricks.

    Hyperparameters from paper:
    - c = 4 (tanh scaling)
    - epsilon = 2 (straight-through bandwidth)
    - lambda_P = 3e-6 (pre-act loss coefficient)
    - lambda_S ~ 10-20 (sparsity coefficient, warmed up)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.get("seed", 42))

        n = cfg["act_size"]  # input dimension
        m = cfg["dict_size"]  # hidden dimension

        # Hyperparameters from paper
        self.c = cfg.get("tanh_c", 4.0)
        self.epsilon = cfg.get("epsilon", 2.0)
        self.lambda_P = cfg.get("lambda_P", 3e-6)
        self.lambda_S_final = cfg.get("lambda_S", 10.0)

        # Initialize decoder: W_d ~ U(-1/sqrt(n), 1/sqrt(n))
        self.W_dec = nn.Parameter(torch.empty(m, n))
        nn.init.uniform_(self.W_dec, -1 / math.sqrt(n), 1 / math.sqrt(n))

        # Initialize encoder: W_e = (n/m) * W_d^T for autoencoders
        self.W_enc = nn.Parameter(torch.empty(n, m))
        with torch.no_grad():
            self.W_enc.data = (n / m) * self.W_dec.data.T.clone()

        # Initialize log thresholds to 0.1 so exp(t) ≈ 1.105
        # Paper says "t: Initialized to 0.1"
        self.log_threshold = nn.Parameter(torch.full((m,), 0.1))

        # Biases
        self.b_dec = nn.Parameter(torch.zeros(n))
        self.b_enc = nn.Parameter(torch.zeros(m))

        # Track dead features
        self.num_batches_not_active = torch.zeros(m, device=cfg.get("device", "cpu"))

        self.to(cfg.get("dtype", torch.float32)).to(cfg.get("device", "cpu"))

    def initialize_b_enc_from_data(self, activation_store: ActivationsStore, target_l0: int):
        """
        Initialize b_enc so each feature activates ~(target_l0/m) of the time.
        Default target_l0 is 10000 (roughly 10000 features fire per datapoint).

        This is important for avoiding dead features at initialization.
        """
        m = self.cfg["dict_size"]
        target_activation_rate = target_l0 / m

        # Get a batch of data to estimate activation distribution
        with torch.no_grad():
            batch = activation_store.next_batch()
            batch, _, _ = self.preprocess_input(batch)

            # Compute pre-activations
            pre_acts = batch @ self.W_enc

            # For each feature, find the threshold that gives target activation rate
            # We want: P(pre_act + b_enc > exp(log_threshold)) = target_rate
            threshold = torch.exp(self.log_threshold)

            # Compute quantile: we want (1 - target_rate) quantile of pre_acts
            quantile_value = torch.quantile(pre_acts, 1 - target_activation_rate, dim=0)

            # Set b_enc so that pre_act + b_enc > threshold at target rate
            # i.e., b_enc = threshold - quantile_value
            self.b_enc.data = threshold - quantile_value

    def preprocess_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Normalize input.

        Paper: "Dataset is scaled by a single constant such that E[||x||_2] = sqrt(n)"
        This ensures lambda_S means the same thing across different sized models.
        """
        if self.cfg.get("input_unit_norm", False):
            # Per-sample normalization (alternative approach)
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        elif self.cfg.get("anthropic_norm", True):
            # Anthropic's normalization: scale so E[||x||_2] = sqrt(n)
            # This is a batch-level scaling
            n = x.shape[-1]
            target_norm = math.sqrt(n)
            actual_norm = x.norm(dim=-1).mean()
            scale = target_norm / (actual_norm + 1e-8)
            x = x * scale
            return x, None, scale
        return x, None, None

    def postprocess_output(
        self,
        x_reconstruct: torch.Tensor,
        x_mean: torch.Tensor | None,
        x_std_or_scale: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Reverse normalization."""
        if (
            self.cfg.get("input_unit_norm", False)
            and x_mean is not None
            and x_std_or_scale is not None
        ):
            x_reconstruct = x_reconstruct * x_std_or_scale + x_mean
        elif self.cfg.get("anthropic_norm", True) and x_std_or_scale is not None:
            # Reverse the scaling
            x_reconstruct = x_reconstruct / x_std_or_scale
        return x_reconstruct

    def forward(
        self, x: torch.Tensor, current_step: int = 0, total_steps: int = 1
    ) -> dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)

        # Encode
        pre_acts = x @ self.W_enc + self.b_enc

        # JumpReLU activation
        acts = JumpReLUFunction.apply(pre_acts, self.log_threshold, self.epsilon)

        # Decode
        x_reconstruct = acts @ self.W_dec + self.b_dec

        # Update dead feature tracking
        self.update_inactive_features(acts)

        # Get loss
        output = self.get_loss_dict(
            x, x_reconstruct, pre_acts, acts, x_mean, x_std, current_step, total_steps
        )
        return output

    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        pre_acts: torch.Tensor,
        acts: torch.Tensor,
        x_mean: torch.Tensor | None,
        x_std_or_scale: torch.Tensor | float | None,
        current_step: int,
        total_steps: int,
    ) -> dict[str, torch.Tensor]:
        # Reconstruction loss: ||y - y_hat||_2^2
        l2_loss = (x_reconstruct - x).pow(2).mean()

        # Sparsity penalty with tanh: lambda_S * sum_i tanh(c * |f_i| * ||W_d_i||_2)
        # Lambda_S is warmed up linearly over entire training
        lambda_S = self.lambda_S_final  # * (current_step / max(total_steps, 1))

        decoder_norms = self.W_dec.norm(dim=-1)  # [m]
        # acts: [batch, m], decoder_norms: [m]
        weighted_acts = torch.abs(acts) * decoder_norms.unsqueeze(0)
        sparsity_loss = lambda_S * torch.tanh(self.c * weighted_acts).sum(dim=-1).mean()

        # Pre-act loss: lambda_P * sum_i ReLU(exp(t_i) - pre_act_i) * ||W_d_i||_2
        # This applies gradient to features that don't fire
        threshold = torch.exp(self.log_threshold)
        pre_act_penalty = torch.relu(threshold.unsqueeze(0) - pre_acts)
        pre_act_loss = (
            self.lambda_P * (pre_act_penalty * decoder_norms.unsqueeze(0)).sum(dim=-1).mean()
        )

        # Total loss
        loss = l2_loss + sparsity_loss + pre_act_loss

        # Metrics
        l0_norm = (acts > 0).float().sum(dim=-1).mean()
        l1_norm = acts.abs().sum(dim=-1).mean()
        num_dead_features = (
            self.num_batches_not_active > self.cfg.get("n_batches_to_dead", 5)
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std_or_scale)

        return {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "sparsity_loss": sparsity_loss,
            "pre_act_loss": pre_act_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "l1_loss": sparsity_loss,  # For compatibility with logging
            "lambda_S": torch.tensor(lambda_S),
        }

    def update_inactive_features(self, acts: torch.Tensor) -> None:
        with torch.no_grad():
            active = acts.sum(0) > 0
            self.num_batches_not_active += (~active).float()
            self.num_batches_not_active[active] = 0

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Project decoder gradients to be orthogonal to decoder weights."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def normalize_decoder_post_training(self) -> None:
        """
        Post-training normalization to make decoder columns unit norm.

        Transforms:
        - W_e' = W_e * ||W_d||_2
        - b_e' = b_e * ||W_d||_2
        - W_d' = W_d / ||W_d||_2
        - b_d' = b_d (unchanged)
        """
        decoder_norms = self.W_dec.norm(dim=-1, keepdim=True)  # [m, 1]

        self.W_enc.data = self.W_enc.data * decoder_norms.T  # [n, m]
        self.b_enc.data = self.b_enc.data * decoder_norms.squeeze()  # [m]
        self.W_dec.data = self.W_dec.data / decoder_norms  # [m, n]
        # b_dec unchanged


def get_anthropic_cfg() -> dict:
    """Get default config for Anthropic JumpReLU SAE."""
    cfg = get_default_cfg()
    cfg["sae_type"] = "anthropic_jumprelu"

    # Anthropic hyperparameters from paper
    cfg["tanh_c"] = 4.0
    cfg["epsilon"] = 2.0
    cfg["lambda_P"] = 3e-6
    cfg["lambda_S"] = 20.0  # Final value, warmed up linearly

    # Initialization
    cfg["init_b_enc_from_data"] = True
    cfg["target_l0"] = 10000  # Target ~10k features active per datapoint

    # Data normalization - use Anthropic's approach by default
    cfg["input_unit_norm"] = False  # Disable per-sample normalization
    cfg["anthropic_norm"] = True  # Scale so E[||x||_2] = sqrt(n)

    # Training defaults from paper
    cfg["batch_size"] = 32768
    cfg["lr"] = 2e-4
    cfg["max_grad_norm"] = 1.0  # Paper uses gradient clipping to norm 1

    # Optimizer defaults (Adam with beta1=0.9, beta2=0.999, no weight decay)
    cfg["beta1"] = 0.9
    cfg["beta2"] = 0.999
    cfg["weight_decay"] = 0.0

    return cfg


@torch.no_grad()
def log_threshold_visualizations(
    wandb_run,
    step: int,
    activation_store: ActivationsStore,
    sae: AnthropicJumpReLUSAE,
) -> None:
    """Log visualizations of thresholds and activations."""
    cfg = sae.cfg
    batch_tokens = activation_store.get_batch_tokens()[: cfg["batch_size"] // cfg["seq_len"]]
    batch = activation_store.get_activations(batch_tokens).reshape(-1, cfg["act_size"])

    # Preprocess
    x, _, _ = sae.preprocess_input(batch)

    # Encode
    pre_acts = x @ sae.W_enc + sae.b_enc
    threshold = torch.exp(sae.log_threshold)
    acts = (pre_acts > threshold).float() * pre_acts

    # Move to CPU for plotting
    pre_acts_np = pre_acts.cpu().numpy()
    acts_np = acts.cpu().numpy()
    threshold_np = threshold.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Threshold distribution
    ax = axes[0, 0]
    ax.hist(threshold_np, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Threshold Value")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Threshold Distribution\nmean={threshold_np.mean():.4f}, std={threshold_np.std():.4f}"
    )
    ax.axvline(x=threshold_np.mean(), color="r", linestyle="--", alpha=0.7)

    # 2. Pre-activation distribution
    ax = axes[0, 1]
    flat_pre_acts = pre_acts_np.flatten()
    ax.hist(flat_pre_acts, bins=100, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Pre-activation Value")
    ax.set_ylabel("Count")
    ax.set_title(f"Pre-activation Distribution\nmean={flat_pre_acts.mean():.3f}")
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # 3. Post-activation (JumpReLU output) distribution
    ax = axes[0, 2]
    flat_acts = acts_np.flatten()
    nonzero_acts = flat_acts[flat_acts > 0]
    ax.hist(nonzero_acts, bins=100, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Activation Value (non-zero only)")
    ax.set_ylabel("Count")
    ax.set_title(f"Non-zero Activations\n{len(nonzero_acts) / len(flat_acts) * 100:.2f}% active")

    # 4. Per-feature activation rate
    ax = axes[1, 0]
    activation_rate = (acts_np > 0).mean(axis=0)
    sorted_rate = np.sort(activation_rate)[::-1]
    ax.semilogy(sorted_rate + 1e-10)
    ax.set_xlabel("Feature Index (sorted)")
    ax.set_ylabel("Activation Rate (log)")
    ax.set_title(f"Per-Feature Activation Rate\n{(activation_rate > 0).sum()} features ever active")

    # 5. Threshold vs mean pre-activation per feature
    ax = axes[1, 1]
    mean_pre_act = pre_acts_np.mean(axis=0)
    ax.scatter(mean_pre_act, threshold_np, alpha=0.3, s=5)
    ax.set_xlabel("Mean Pre-activation")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold vs Mean Pre-activation")
    # Add diagonal line where threshold = mean_pre_act
    lims = [
        min(mean_pre_act.min(), threshold_np.min()),
        max(mean_pre_act.max(), threshold_np.max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="threshold = pre_act")
    ax.legend()

    # 6. L0 per sample histogram
    ax = axes[1, 2]
    l0_per_sample = (acts_np > 0).sum(axis=1)
    ax.hist(l0_per_sample, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("L0 (active features)")
    ax.set_ylabel("Count")
    ax.set_title(f"L0 per Sample\nmean={l0_per_sample.mean():.1f}, std={l0_per_sample.std():.1f}")

    plt.tight_layout()
    wandb_run.log({"threshold_plots/overview": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Log scalar stats
    wandb_run.log(
        {
            "threshold_stats/mean": float(threshold_np.mean()),
            "threshold_stats/std": float(threshold_np.std()),
            "threshold_stats/min": float(threshold_np.min()),
            "threshold_stats/max": float(threshold_np.max()),
            "activation_stats/pct_active": float((flat_acts > 0).mean() * 100),
            "activation_stats/features_ever_active": int((activation_rate > 0).sum()),
            "activation_stats/mean_l0": float(l0_per_sample.mean()),
        },
        step=step,
    )


def train_anthropic_sae(
    sae: AnthropicJumpReLUSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
    cfg: dict,
) -> None:
    """
    Training loop with Anthropic's optimization techniques:
    - Adam with beta1=0.9, beta2=0.999, no weight decay
    - LR decay over last 20% of training
    - Gradient clipping to norm 1
    """
    num_batches = cfg["num_tokens"] // cfg["batch_size"]

    # Initialize b_enc from data
    if cfg.get("init_b_enc_from_data", True):
        print("Initializing b_enc from data...")
        sae.initialize_b_enc_from_data(activation_store, target_l0=cfg.get("target_l0", 10000))

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    # LR scheduler: linear decay over last 20%
    decay_start = int(0.8 * num_batches)

    def lr_lambda(step: int) -> float:
        if step < decay_start:
            return 1.0
        else:
            # Linear decay from 1.0 to 0.0 over last 20%
            progress = (step - decay_start) / (num_batches - decay_start)
            return 1.0 - progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pbar = tqdm.trange(num_batches)
    wandb_run = init_wandb(cfg)

    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch, current_step=i, total_steps=num_batches)

        # Log to wandb
        log_dict = {
            "loss": sae_output["loss"].item(),
            "l2_loss": sae_output["l2_loss"].item(),
            "sparsity_loss": sae_output["sparsity_loss"].item(),
            "pre_act_loss": sae_output["pre_act_loss"].item(),
            "l0_norm": sae_output["l0_norm"].item(),
            "l1_norm": sae_output["l1_norm"].item(),
            "num_dead_features": sae_output["num_dead_features"].item(),
            "lambda_S": sae_output["lambda_S"].item(),
            "lr": scheduler.get_last_lr()[0],
            "n_dead_in_batch": (sae_output["feature_acts"].sum(0) == 0).sum().item(),
        }
        wandb_run.log(log_dict, step=i)

        if i % cfg.get("perf_log_freq", 1000) == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)
            log_threshold_visualizations(wandb_run, i, activation_store, sae)

        if i % cfg.get("checkpoint_freq", 10000) == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm'].item():.1f}",
                "L2": f"{sae_output['l2_loss'].item():.4f}",
                "S": f"{sae_output['sparsity_loss'].item():.4f}",
                "P": f"{sae_output['pre_act_loss'].item():.6f}",
                "lS": f"{sae_output['lambda_S'].item():.2f}",
            }
        )

        loss.backward()

        # Gradient clipping to norm 1
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.get("max_grad_norm", 1.0))

        # Keep decoder unit norm
        sae.make_decoder_weights_and_grad_unit_norm()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Post-training: normalize decoder
    print("Applying post-training decoder normalization...")
    sae.normalize_decoder_post_training()

    save_checkpoint(wandb_run, sae, cfg, num_batches)
    print("Training complete!")


def train_comparison(
    jumprelu_sae: AnthropicJumpReLUSAE,
    topk_sae: BatchTopKSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
    cfg: dict,
) -> None:
    """
    Train JumpReLU and BatchTopK SAEs in parallel for comparison.

    Both SAEs see the same batches and use comparable training settings.
    Metrics are logged with prefixes 'jumprelu/' and 'topk/' for comparison.
    """
    num_batches = cfg["num_tokens"] // cfg["batch_size"]

    # Initialize JumpReLU SAE
    if cfg.get("init_b_enc_from_data", True):
        print("Initializing JumpReLU b_enc from data...")
        jumprelu_sae.initialize_b_enc_from_data(
            activation_store, target_l0=cfg.get("target_l0", 10000)
        )

    # Optimizers - same settings for fair comparison
    jumprelu_optimizer = torch.optim.Adam(
        jumprelu_sae.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    topk_optimizer = torch.optim.Adam(
        topk_sae.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    # LR scheduler for JumpReLU (decay over last 20%)
    decay_start = int(0.8 * num_batches)

    def lr_lambda(step: int) -> float:
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_batches - decay_start)
        return 1.0 - progress

    jumprelu_scheduler = torch.optim.lr_scheduler.LambdaLR(jumprelu_optimizer, lr_lambda)

    pbar = tqdm.trange(num_batches)
    wandb_run = init_wandb(cfg)

    for i in pbar:
        batch = activation_store.next_batch()

        # Forward pass for both SAEs
        jumprelu_output = jumprelu_sae(batch, current_step=i, total_steps=num_batches)
        topk_output = topk_sae(batch)

        # Log JumpReLU metrics
        jumprelu_log = {
            "jumprelu/loss": jumprelu_output["loss"].item(),
            "jumprelu/l2_loss": jumprelu_output["l2_loss"].item(),
            "jumprelu/sparsity_loss": jumprelu_output["sparsity_loss"].item(),
            "jumprelu/pre_act_loss": jumprelu_output["pre_act_loss"].item(),
            "jumprelu/l0_norm": jumprelu_output["l0_norm"].item(),
            "jumprelu/l1_norm": jumprelu_output["l1_norm"].item(),
            "jumprelu/num_dead_features": jumprelu_output["num_dead_features"].item(),
            "jumprelu/lambda_S": jumprelu_output["lambda_S"].item(),
            "jumprelu/n_dead_in_batch": (jumprelu_output["feature_acts"].sum(0) == 0).sum().item(),
        }

        # Log TopK metrics
        topk_log = {
            "topk/loss": topk_output["loss"].item(),
            "topk/l2_loss": topk_output["l2_loss"].item(),
            "topk/l0_norm": topk_output["l0_norm"].item(),
            "topk/l1_norm": topk_output["l1_norm"].item(),
            "topk/aux_loss": topk_output["aux_loss"].item(),
            "topk/num_dead_features": topk_output["num_dead_features"].item(),
            "topk/n_dead_in_batch": (topk_output["feature_acts"].sum(0) == 0).sum().item(),
        }

        # Combined logging
        log_dict = {**jumprelu_log, **topk_log, "lr": jumprelu_scheduler.get_last_lr()[0]}
        wandb_run.log(log_dict, step=i)

        # Performance logging
        if i % cfg.get("perf_log_freq", 1000) == 0:
            log_model_performance(
                wandb_run, i, model, activation_store, jumprelu_sae, index="jumprelu"
            )
            log_model_performance(wandb_run, i, model, activation_store, topk_sae, index="topk")
            log_threshold_visualizations(wandb_run, i, activation_store, jumprelu_sae)

        if i % cfg.get("checkpoint_freq", 10000) == 0:
            save_checkpoint(wandb_run, jumprelu_sae, {**cfg, "sae_type": "jumprelu"}, i)
            save_checkpoint(wandb_run, topk_sae, {**cfg, "sae_type": "topk"}, i)

        # Progress bar
        pbar.set_postfix(
            {
                "JR_L2": f"{jumprelu_output['l2_loss'].item():.4f}",
                "JR_L0": f"{jumprelu_output['l0_norm'].item():.1f}",
                "TK_L2": f"{topk_output['l2_loss'].item():.4f}",
                "TK_L0": f"{topk_output['l0_norm'].item():.1f}",
            }
        )

        # Backward and optimize JumpReLU
        jumprelu_output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(jumprelu_sae.parameters(), cfg.get("max_grad_norm", 1.0))
        jumprelu_sae.make_decoder_weights_and_grad_unit_norm()
        jumprelu_optimizer.step()
        jumprelu_scheduler.step()
        jumprelu_optimizer.zero_grad()

        # Backward and optimize TopK
        topk_output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(topk_sae.parameters(), cfg.get("max_grad_norm", 1.0))
        topk_sae.make_decoder_weights_and_grad_unit_norm()
        topk_optimizer.step()
        topk_optimizer.zero_grad()

    # Post-training normalization for JumpReLU
    print("Applying post-training decoder normalization to JumpReLU SAE...")
    jumprelu_sae.normalize_decoder_post_training()

    save_checkpoint(wandb_run, jumprelu_sae, {**cfg, "sae_type": "jumprelu"}, num_batches)
    save_checkpoint(wandb_run, topk_sae, {**cfg, "sae_type": "topk"}, num_batches)
    print("Comparison training complete!")


def main(
    # Training settings
    num_tokens: int = 100_000_000,
    batch_size: int = 4096,
    lr: float = 2e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    # Model settings
    model_name: str = "gpt2-small",
    layer: int = 8,
    site: str = "mlp_out",
    dict_size: int = 768 * 4,
    # Anthropic hyperparameters
    tanh_c: float = 4.0,
    epsilon: float = 2.0,
    lambda_P: float = 3e-6,
    lambda_S: float = 0.2,
    target_l0: int = 2000,
    init_b_enc_from_data: bool = True,
    # Comparison mode
    compare_with_topk: bool = True,
    top_k: int = 32,
    aux_penalty: float = 1 / 32,
    # Other
    seed: int = 42,
    device: str = "cuda:0",
    wandb_project: str = "anthropic_jumprelu_sae",
) -> None:
    """Run Anthropic JumpReLU SAE training experiment.

    Based on "Dictionary Learning Optimization Techniques" (January 2025)
    https://transformer-circuits.pub/2025/january-update/index.html#DL

    Args:
        num_tokens: Total tokens to train on
        batch_size: Batch size (paper recommends 32768)
        lr: Learning rate (paper default 2e-4)
        beta1: Adam beta1 (paper: 0.9)
        beta2: Adam beta2 (paper: 0.999)
        weight_decay: Weight decay (paper: 0)
        model_name: Transformer model name
        layer: Layer to extract activations from
        site: Hook site (resid_pre, resid_post, etc.)
        dict_size: SAE dictionary size (number of features)
        tanh_c: Tanh scaling factor (paper: 4)
        epsilon: Straight-through estimator bandwidth (paper: 2)
        lambda_P: Pre-act loss coefficient (paper: 3e-6)
        lambda_S: Final sparsity coefficient (paper: ~10-20)
        target_l0: Target L0 for b_enc initialization (paper: 10000)
        init_b_enc_from_data: Whether to initialize b_enc from data
        compare_with_topk: Train BatchTopK SAE in parallel for comparison
        top_k: K value for BatchTopK SAE (when compare_with_topk=True)
        aux_penalty: Auxiliary loss penalty for BatchTopK dead features
        seed: Random seed
        device: Device to train on
        wandb_project: W&B project name
    """
    # Build config
    cfg = get_anthropic_cfg()
    cfg["seed"] = seed
    cfg["num_tokens"] = num_tokens
    cfg["batch_size"] = batch_size
    cfg["lr"] = lr
    cfg["beta1"] = beta1
    cfg["beta2"] = beta2
    cfg["weight_decay"] = weight_decay
    cfg["model_name"] = model_name
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["dict_size"] = dict_size
    cfg["tanh_c"] = tanh_c
    cfg["epsilon"] = epsilon
    cfg["lambda_P"] = lambda_P
    cfg["lambda_S"] = lambda_S
    cfg["target_l0"] = target_l0
    cfg["init_b_enc_from_data"] = init_b_enc_from_data
    cfg["device"] = device
    cfg["wandb_project"] = wandb_project

    # TopK-specific config (used when compare_with_topk=True)
    cfg["top_k"] = top_k
    cfg["top_k_aux"] = min(512, dict_size)
    cfg["aux_penalty"] = aux_penalty
    cfg["l1_coeff"] = 0.0  # TopK doesn't need L1 regularization

    cfg = post_init_cfg(cfg)

    mode = "comparison (JumpReLU vs BatchTopK)" if compare_with_topk else "JumpReLU only"
    print(f"Training mode: {mode}")
    print(f"  Model: {model_name} layer {layer} ({site})")
    print(f"  dict_size: {dict_size}")
    print(
        f"  JumpReLU params: c={tanh_c}, epsilon={epsilon}, lambda_P={lambda_P}, lambda_S={lambda_S}"
    )
    if compare_with_topk:
        print(f"  TopK params: top_k={top_k}, aux_penalty={aux_penalty}")
    print(f"  target_l0: {target_l0}, init_b_enc_from_data: {init_b_enc_from_data}")
    print(f"  batch_size: {batch_size}, lr: {lr}")
    print(f"  num_tokens: {num_tokens:,}")

    # Load model
    model = HookedTransformer.from_pretrained(model_name).to(device)
    cfg["act_size"] = model.cfg.d_model

    # Create activation store
    activation_store = ActivationsStore(model, cfg)

    # Create JumpReLU SAE
    jumprelu_sae = AnthropicJumpReLUSAE(cfg)
    print(f"  JumpReLU SAE parameters: {sum(p.numel() for p in jumprelu_sae.parameters()):,}")

    if compare_with_topk:
        # Create BatchTopK SAE with same architecture
        topk_sae = BatchTopKSAE(cfg)
        print(f"  BatchTopK SAE parameters: {sum(p.numel() for p in topk_sae.parameters()):,}")

        # Train both in parallel
        train_comparison(jumprelu_sae, topk_sae, activation_store, model, cfg)
    else:
        # Train JumpReLU only
        train_anthropic_sae(jumprelu_sae, activation_store, model, cfg)

    print("Training complete!")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

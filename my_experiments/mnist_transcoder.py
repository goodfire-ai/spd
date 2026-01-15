"""Train a transcoder to imitate the MLP layer on MNIST.

A transcoder learns sparse features that map from the input of a layer
to its output, effectively imitating the layer's computation while
learning interpretable sparse features.

Based on BatchTopK implementation: https://github.com/bartbussmann/BatchTopK
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.log import logger
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification (same as in mnist_experiment.py)."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_fc1_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get the output of fc1 (post-ReLU activations)."""
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class Transcoder(nn.Module):
    """A transcoder that imitates an MLP layer.

    The transcoder maps from the input of a layer to its output through a
    sparse hidden representation. Supports multiple activation functions:
    - 'relu': Standard ReLU with L1 sparsity penalty
    - 'topk': Top-K activation per sample
    - 'batchtopk': Top-K activation across the entire batch

    Based on BatchTopK: https://github.com/bartbussmann/BatchTopK
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        n_features: int,
        activation: Literal["relu", "topk", "batchtopk"] = "topk",
        top_k: int = 32,
        input_unit_norm: bool = True,
        n_batches_to_dead: int = 5,
    ):
        """Initialize the transcoder.

        Args:
            d_input: Input dimension (e.g., 784 for flattened MNIST)
            d_output: Output dimension (e.g., 128 for hidden layer)
            n_features: Number of sparse features in the hidden layer
            activation: Activation type ('relu', 'topk', or 'batchtopk')
            top_k: Number of top activations to keep (for topk/batchtopk)
            input_unit_norm: Whether to normalize inputs to unit norm
            n_batches_to_dead: Number of batches without activation before feature is "dead"
        """
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_features = n_features
        self.activation = activation
        self.top_k = top_k
        self.input_unit_norm = input_unit_norm
        self.n_batches_to_dead = n_batches_to_dead

        # Encoder: d_input -> n_features
        self.W_enc = nn.Parameter(torch.empty(n_features, d_input))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: n_features -> d_output
        self.W_dec = nn.Parameter(torch.empty(d_output, n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_output))

        # Dead feature tracking
        self.register_buffer("num_batches_not_active", torch.zeros(n_features, dtype=torch.long))

        # For input normalization
        self.register_buffer("input_mean", torch.zeros(d_input))
        self.register_buffer("input_std", torch.ones(d_input))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform distribution."""
        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=np.sqrt(5))
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

        # Normalize decoder weights to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Normalize decoder weights and their gradients to unit norm.

        This is a key technique from the BatchTopK paper that ensures
        decoder columns remain unit norm throughout training.
        """
        # Normalize weights
        W_dec_normed = F.normalize(self.W_dec.data, dim=0)
        self.W_dec.data = W_dec_normed

        # Project gradients to be orthogonal to weights (if gradients exist)
        if self.W_dec.grad is not None:
            # Remove component parallel to weights
            dot = (self.W_dec.grad * W_dec_normed).sum(dim=0, keepdim=True)
            self.W_dec.grad = self.W_dec.grad - dot * W_dec_normed

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess input, optionally normalizing to unit norm.

        Returns:
            x_processed: Processed input
            x_norm: Original norm (for denormalization)
        """
        if self.input_unit_norm:
            x_norm = x.norm(dim=-1, keepdim=True)
            x_processed = x / (x_norm + 1e-8)
        else:
            x_norm = torch.ones(x.shape[0], 1, device=x.device)
            x_processed = x
        return x_processed, x_norm

    def postprocess(self, y: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """Reverse the preprocessing normalization."""
        if self.input_unit_norm:
            return y * x_norm
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to pre-activation features."""
        return x @ self.W_enc.T + self.b_enc

    def decode(self, hidden_acts: torch.Tensor) -> torch.Tensor:
        """Decode hidden activations to output."""
        return hidden_acts @ self.W_dec.T + self.b_dec

    def apply_activation(self, pre_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the activation function.

        Args:
            pre_acts: Pre-activation values (batch, n_features)

        Returns:
            hidden_acts: Post-activation values
            auxiliary_loss: Auxiliary loss for dead feature revival
        """
        if self.activation == "relu":
            hidden_acts = F.relu(pre_acts)
            auxiliary_loss = torch.tensor(0.0, device=pre_acts.device)

        elif self.activation == "topk":
            # Top-K per sample
            top_k = min(self.top_k, pre_acts.shape[-1])
            topk_values, topk_indices = torch.topk(pre_acts, k=top_k, dim=-1)
            hidden_acts = torch.zeros_like(pre_acts)
            hidden_acts.scatter_(-1, topk_indices, F.relu(topk_values))

            # Auxiliary loss for dead features
            auxiliary_loss = self._compute_auxiliary_loss(pre_acts, topk_indices)

        elif self.activation == "batchtopk":
            # Top-K across entire batch
            batch_size = pre_acts.shape[0]
            total_k = min(self.top_k * batch_size, pre_acts.numel())

            flat_pre_acts = pre_acts.view(-1)
            topk_values, topk_indices = torch.topk(flat_pre_acts, k=total_k)

            hidden_acts = torch.zeros_like(flat_pre_acts)
            hidden_acts.scatter_(0, topk_indices, F.relu(topk_values))
            hidden_acts = hidden_acts.view(batch_size, -1)

            # Track which features were active
            active_features = topk_indices % self.n_features
            auxiliary_loss = self._compute_auxiliary_loss_batchtopk(pre_acts, active_features)

        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Update dead feature tracking
        self._update_dead_feature_tracking(hidden_acts)

        return hidden_acts, auxiliary_loss

    def _compute_auxiliary_loss(
        self, pre_acts: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss to revive dead features (TopK version)."""
        dead_mask = self.num_batches_not_active >= self.n_batches_to_dead

        if not dead_mask.any():
            return torch.tensor(0.0, device=pre_acts.device)

        # For dead features, encourage them to activate
        dead_pre_acts = pre_acts[:, dead_mask]
        # Use top-k_aux from dead features
        k_aux = min(self.top_k * 16, dead_pre_acts.shape[-1])
        if k_aux == 0:
            return torch.tensor(0.0, device=pre_acts.device)

        topk_aux_values, _ = torch.topk(dead_pre_acts, k=k_aux, dim=-1)
        aux_acts = F.relu(topk_aux_values)

        # Reconstruct using only dead features and compute loss
        # This encourages dead features to learn useful representations
        aux_recon = aux_acts @ self.W_dec[:, dead_mask].T
        return aux_recon.pow(2).mean() * 0.03125  # 1/32 penalty

    def _compute_auxiliary_loss_batchtopk(
        self, pre_acts: torch.Tensor, active_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss for BatchTopK."""
        dead_mask = self.num_batches_not_active >= self.n_batches_to_dead

        if not dead_mask.any():
            return torch.tensor(0.0, device=pre_acts.device)

        dead_pre_acts = pre_acts[:, dead_mask]
        k_aux = min(self.top_k * 16, dead_pre_acts.numel())
        if k_aux == 0:
            return torch.tensor(0.0, device=pre_acts.device)

        flat_dead = dead_pre_acts.view(-1)
        topk_aux_values, _ = torch.topk(flat_dead, k=k_aux)
        aux_acts = F.relu(topk_aux_values)

        return aux_acts.pow(2).mean() * 0.03125

    @torch.no_grad()
    def _update_dead_feature_tracking(self, hidden_acts: torch.Tensor) -> None:
        """Update tracking of which features have been active."""
        if not self.training:
            return

        # Check which features were active in this batch
        feature_active = (hidden_acts > 0).any(dim=0)

        # Reset counter for active features, increment for inactive
        self.num_batches_not_active[feature_active] = 0
        self.num_batches_not_active[~feature_active] += 1

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the transcoder.

        Args:
            x: Input tensor of shape (batch, d_input)

        Returns:
            y_hat: Reconstructed output of shape (batch, d_output)
            hidden_acts: Hidden activations of shape (batch, n_features)
            auxiliary_loss: Loss for dead feature revival
            x_norm: Input norm (for loss computation)
        """
        # Preprocess input
        x_processed, x_norm = self.preprocess(x)

        # Encode
        pre_acts = self.encode(x_processed)

        # Apply activation
        hidden_acts, auxiliary_loss = self.apply_activation(pre_acts)

        # Decode
        y_hat = self.decode(hidden_acts)

        # Postprocess (reverse normalization)
        y_hat = self.postprocess(y_hat, x_norm)

        return y_hat, hidden_acts, auxiliary_loss, x_norm

    def get_feature_directions(self) -> torch.Tensor:
        """Get the encoder weight matrix (feature directions in input space).

        Returns:
            Tensor of shape (n_features, d_input)
        """
        return self.W_enc.detach()

    def get_num_dead_features(self) -> int:
        """Get the number of dead features."""
        return (self.num_batches_not_active >= self.n_batches_to_dead).sum().item()


def train_mlp(
    model: TwoLayerMLP,
    train_loader: DataLoader,
    device: str,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    log_wandb: bool = False,
) -> int:
    """Train the MLP on MNIST (same as in mnist_experiment.py)."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training MLP for {epochs} epochs...")
    total_steps = 0
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

            if log_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "mlp_train/loss": loss.item(),
                        "mlp_train/accuracy": 100.0 * correct / total,
                        "mlp_train/epoch": epoch,
                    },
                    step=total_steps,
                )

            total_steps += 1

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    model.eval()
    return total_steps


def train_transcoder(
    transcoder: Transcoder,
    target_model: TwoLayerMLP,
    train_loader: DataLoader,
    device: str,
    steps: int = 50000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    l1_coeff: float = 0.0,
    aux_penalty: float = 1 / 32,
    max_grad_norm: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99,
    log_wandb: bool = False,
    log_freq: int = 100,
    eval_freq: int = 1000,
    test_loader: DataLoader | None = None,
    step_offset: int = 0,
) -> None:
    """Train the transcoder to imitate the target MLP's fc1 layer.

    Based on BatchTopK training: https://github.com/bartbussmann/BatchTopK

    Args:
        transcoder: The transcoder model to train
        target_model: The target MLP model (frozen)
        train_loader: DataLoader for training data
        device: Device to train on
        steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        l1_coeff: L1 sparsity coefficient (mainly for 'relu' activation)
        aux_penalty: Auxiliary loss weight for dead feature revival
        max_grad_norm: Maximum gradient norm for clipping
        beta1: Adam beta1
        beta2: Adam beta2
        log_wandb: Whether to log to WandB
        log_freq: Frequency of logging
        eval_freq: Frequency of evaluation
        test_loader: DataLoader for test data (for evaluation)
        step_offset: Offset for step counting (for WandB logging continuity)
    """
    transcoder.train()
    target_model.eval()
    target_model.requires_grad_(False)

    optimizer = optim.Adam(transcoder.parameters(), lr=lr, betas=(beta1, beta2))

    logger.info(f"Training transcoder for {steps} steps...")
    logger.info(f"  Activation: {transcoder.activation}")
    if transcoder.activation in ["topk", "batchtopk"]:
        logger.info(f"  Top-K: {transcoder.top_k}")
    logger.info(f"  Input unit norm: {transcoder.input_unit_norm}")

    # Create an infinite data iterator
    def infinite_loader():
        while True:
            for images, _ in train_loader:
                yield images

    data_iter = iter(infinite_loader())

    pbar = tqdm(range(steps), desc="Training transcoder")
    running_loss = 0.0
    running_recon_loss = 0.0
    running_l1_loss = 0.0
    running_aux_loss = 0.0

    for step in pbar:
        # Get batch
        images = next(data_iter).to(device)
        images = images.view(images.size(0), -1)  # Flatten

        # Get target activations from the MLP
        with torch.no_grad():
            target_acts = target_model.get_fc1_output(images)

        # Forward through transcoder
        recon_acts, hidden_acts, aux_loss, x_norm = transcoder(images)

        # Compute losses
        # L2 reconstruction loss (normalized by input norm if using input_unit_norm)
        if transcoder.input_unit_norm:
            recon_loss = ((recon_acts - target_acts) / (x_norm + 1e-8)).pow(2).mean()
        else:
            recon_loss = F.mse_loss(recon_acts, target_acts)

        # L1 sparsity loss (mainly for ReLU activation)
        l1_loss = hidden_acts.abs().mean()

        # Total loss
        total_loss = recon_loss + l1_coeff * l1_loss + aux_penalty * aux_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(transcoder.parameters(), max_grad_norm)

        # Normalize decoder weights and gradients to unit norm
        transcoder.make_decoder_weights_and_grad_unit_norm()

        # Optimizer step
        optimizer.step()

        # Track running losses
        running_loss += total_loss.item()
        running_recon_loss += recon_loss.item()
        running_l1_loss += l1_loss.item()
        running_aux_loss += aux_loss.item()

        # Compute L0 (average number of active features)
        with torch.no_grad():
            l0 = (hidden_acts > 0).float().sum(dim=1).mean().item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "L0": f"{l0:.1f}",
                "dead": f"{transcoder.get_num_dead_features()}",
            }
        )

        # Log to WandB
        if log_wandb and step % log_freq == 0:
            n = log_freq if step > 0 else 1
            avg_loss = running_loss / n
            avg_recon = running_recon_loss / n
            avg_l1 = running_l1_loss / n
            avg_aux = running_aux_loss / n

            wandb.log(
                {
                    "transcoder/loss": avg_loss,
                    "transcoder/recon_loss": avg_recon,
                    "transcoder/l1_loss": avg_l1,
                    "transcoder/aux_loss": avg_aux,
                    "transcoder/l0": l0,
                    "transcoder/dead_features": transcoder.get_num_dead_features(),
                    "transcoder/lr": lr,
                },
                step=step + step_offset,
            )

            running_loss = 0.0
            running_recon_loss = 0.0
            running_l1_loss = 0.0
            running_aux_loss = 0.0

        # Evaluation
        if test_loader is not None and step % eval_freq == 0:
            eval_metrics = evaluate_transcoder(transcoder, target_model, test_loader, device)
            logger.info(
                f"Step {step}: eval_recon_loss={eval_metrics['recon_loss']:.4f}, "
                f"eval_l0={eval_metrics['l0']:.2f}, "
                f"dead_features={transcoder.get_num_dead_features()}"
            )
            if log_wandb:
                wandb.log(
                    {
                        "transcoder_eval/recon_loss": eval_metrics["recon_loss"],
                        "transcoder_eval/l0": eval_metrics["l0"],
                        "transcoder_eval/frac_variance_explained": eval_metrics[
                            "frac_variance_explained"
                        ],
                    },
                    step=step + step_offset,
                )

    transcoder.eval()


def evaluate_transcoder(
    transcoder: Transcoder,
    target_model: TwoLayerMLP,
    test_loader: DataLoader,
    device: str,
    n_batches: int = 10,
) -> dict:
    """Evaluate the transcoder on test data."""
    transcoder.eval()
    target_model.eval()

    total_recon_loss = 0.0
    total_l0 = 0.0
    total_variance = 0.0
    total_explained_variance = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            if batch_idx >= n_batches:
                break

            images = images.to(device)
            images = images.view(images.size(0), -1)

            target_acts = target_model.get_fc1_output(images)
            recon_acts, hidden_acts, _, _ = transcoder(images)

            # Reconstruction loss
            total_recon_loss += F.mse_loss(recon_acts, target_acts).item() * images.size(0)

            # L0 norm
            total_l0 += (hidden_acts > 0).float().sum(dim=1).mean().item() * images.size(0)

            # Variance explained
            total_variance += target_acts.var(dim=0).sum().item() * images.size(0)
            residual_variance = (target_acts - recon_acts).var(dim=0).sum().item()
            total_explained_variance += (
                target_acts.var(dim=0).sum().item() - residual_variance
            ) * images.size(0)

            n_samples += images.size(0)

    transcoder.train()

    frac_variance_explained = total_explained_variance / (total_variance + 1e-8)

    return {
        "recon_loss": total_recon_loss / n_samples,
        "l0": total_l0 / n_samples,
        "frac_variance_explained": frac_variance_explained,
    }


def plot_feature_directions(
    transcoder: Transcoder,
    out_dir: Path,
    n_features_to_show: int = 50,
    log_wandb: bool = False,
) -> None:
    """Plot transcoder feature directions as images.

    The encoder weights represent directions in input space (784-dim -> 28x28 images).
    """
    feature_dirs = transcoder.get_feature_directions().cpu()  # Shape: (n_features, 784)
    n_features_to_show = min(n_features_to_show, feature_dirs.shape[0])

    logger.info(f"Visualizing {n_features_to_show} transcoder features...")

    # Compute feature norms to sort by importance
    feature_norms = feature_dirs.norm(dim=1)
    top_features = torch.argsort(feature_norms, descending=True)[:n_features_to_show]

    n_cols = 10
    n_rows = (n_features_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_features):
        feature_direction = feature_dirs[feat_idx].reshape(28, 28).numpy()

        # Normalize for visualization
        vmax = max(abs(feature_direction.min()), abs(feature_direction.max()))
        if vmax > 1e-6:
            feature_direction = feature_direction / vmax

        axes[i].imshow(feature_direction, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i].set_title(f"F{feat_idx.item()}", fontsize=8)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_features_to_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Transcoder Feature Directions (Top by Norm)", fontsize=14, y=1.0)
    plt.tight_layout()

    plot_path = out_dir / "transcoder_features.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved transcoder features to {plot_path}")

    if log_wandb:
        wandb.log({"visualizations/transcoder_features": wandb.Image(str(plot_path))})


def plot_feature_activations_on_images(
    transcoder: Transcoder,
    test_dataset: datasets.MNIST,
    device: str,
    out_dir: Path,
    n_samples: int = 10,
    n_features_to_show: int = 20,
    log_wandb: bool = False,
) -> None:
    """Plot feature activations on sample MNIST images."""
    sample_indices = np.random.choice(len(test_dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 2))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    transcoder.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            image, label = test_dataset[int(sample_idx)]
            image_flat = image.view(1, -1).to(device)

            _, hidden_acts, _, _ = transcoder(image_flat)
            hidden_acts = hidden_acts[0].cpu().numpy()

            # Plot original image
            axes[idx, 0].imshow(image.squeeze().numpy(), cmap="gray")
            axes[idx, 0].set_title(f"Digit: {label}", fontsize=10)
            axes[idx, 0].axis("off")

            # Plot feature activations
            n_show = min(n_features_to_show, len(hidden_acts))
            top_features = np.argsort(hidden_acts)[-n_show:][::-1]
            top_activations = hidden_acts[top_features]

            # Only show non-zero activations
            nonzero_mask = top_activations > 0
            if nonzero_mask.any():
                top_features = top_features[nonzero_mask]
                top_activations = top_activations[nonzero_mask]

            axes[idx, 1].barh(range(len(top_features)), top_activations, color="steelblue")
            axes[idx, 1].set_yticks(range(len(top_features)))
            axes[idx, 1].set_yticklabels([f"F{f}" for f in top_features], fontsize=8)
            axes[idx, 1].set_xlabel("Activation", fontsize=9)
            axes[idx, 1].set_title("Top Feature Activations", fontsize=10)
            axes[idx, 1].grid(axis="x", alpha=0.3)

    plt.suptitle("Transcoder Feature Activations on Sample Images", fontsize=14, y=1.0)
    plt.tight_layout()

    plot_path = out_dir / "transcoder_activations_samples.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved transcoder activations plot to {plot_path}")

    if log_wandb:
        wandb.log({"visualizations/transcoder_activations_samples": wandb.Image(str(plot_path))})


def plot_digit_feature_heatmap(
    transcoder: Transcoder,
    test_dataset: datasets.MNIST,
    device: str,
    out_dir: Path,
    n_samples_per_digit: int = 50,
    n_features_to_show: int = 50,
    log_wandb: bool = False,
) -> None:
    """Plot a heatmap showing average feature activations per digit class."""
    n_features = transcoder.n_features

    # Collect activations per digit
    digit_activations = {digit: [] for digit in range(10)}

    transcoder.eval()
    with torch.no_grad():
        for digit in range(10):
            digit_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == digit]
            if len(digit_indices) == 0:
                continue

            sample_indices = np.random.choice(
                digit_indices, min(n_samples_per_digit, len(digit_indices)), replace=False
            )

            for idx in sample_indices:
                image, _ = test_dataset[int(idx)]
                image_flat = image.view(1, -1).to(device)

                _, hidden_acts, _, _ = transcoder(image_flat)
                digit_activations[digit].append(hidden_acts[0].cpu().numpy())

    # Compute average activations per digit
    avg_activations = np.zeros((10, n_features))
    for digit in range(10):
        if len(digit_activations[digit]) > 0:
            avg_activations[digit] = np.mean(digit_activations[digit], axis=0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(14, n_features_to_show * 0.3), 6))

    # Show top features by overall activation
    feature_importance = np.abs(avg_activations).mean(axis=0)
    top_features = np.argsort(feature_importance)[-n_features_to_show:][::-1]

    im = ax.imshow(
        avg_activations[:, top_features],
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Feature Index", fontsize=12)
    ax.set_ylabel("Digit Class", fontsize=12)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xticks(range(0, len(top_features), max(1, len(top_features) // 20)))
    ax.set_xticklabels(
        [
            f"F{top_features[i]}"
            for i in range(0, len(top_features), max(1, len(top_features) // 20))
        ],
        rotation=45,
        ha="right",
    )
    ax.set_title("Average Transcoder Feature Activations by Digit Class", fontsize=14)

    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()

    heatmap_path = out_dir / "transcoder_digit_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved digit-feature heatmap to {heatmap_path}")

    if log_wandb:
        wandb.log({"visualizations/transcoder_digit_heatmap": wandb.Image(str(heatmap_path))})


def main(
    hidden_size: int = 128,
    train_epochs: int = 50,
    train_lr: float = 0.001,
    train_weight_decay: float = 5e-4,
    n_features: int = 500,
    transcoder_steps: int = 50000,
    transcoder_lr: float = 3e-4,
    transcoder_l1_coeff: float = 0.0,
    transcoder_aux_penalty: float = 1 / 32,
    transcoder_activation: Literal["relu", "topk", "batchtopk"] = "batchtopk",
    transcoder_top_k: int = 32,
    transcoder_input_unit_norm: bool = True,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    output_dir: str | None = None,
    wandb_project: str | None = "mnist_transcoder",
    pretrained_mlp_path: str | None = None,
) -> None:
    """Main experiment function.

    Args:
        hidden_size: Hidden layer size for the MLP
        train_epochs: Number of epochs to train the MLP (if not using pretrained)
        train_lr: Learning rate for MLP training
        train_weight_decay: Weight decay for MLP training
        n_features: Number of features in the transcoder
        transcoder_steps: Number of steps for transcoder training
        transcoder_lr: Learning rate for transcoder training
        transcoder_l1_coeff: L1 sparsity coefficient (mainly for 'relu' activation)
        transcoder_aux_penalty: Auxiliary loss weight for dead feature revival
        transcoder_activation: Activation type ('relu', 'topk', or 'batchtopk')
        transcoder_top_k: Number of top activations to keep (for topk/batchtopk)
        transcoder_input_unit_norm: Whether to normalize inputs to unit norm
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        output_dir: Output directory (defaults to ./output/mnist_transcoder)
        wandb_project: WandB project name (None to disable wandb logging)
        pretrained_mlp_path: Path to pretrained MLP weights (if None, trains new MLP)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        output_dir = "./output/mnist_transcoder"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if wandb_project:
        import os

        from dotenv import load_dotenv

        load_dotenv(override=True)
        run_name = f"mnist_transcoder_{transcoder_activation}_h{hidden_size}_f{n_features}_k{transcoder_top_k}_s{seed}"

        wandb.init(
            project=wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            tags=["mnist", "transcoder", transcoder_activation],
        )
        assert wandb.run is not None

        wandb.config.update(
            {
                "hidden_size": hidden_size,
                "train_epochs": train_epochs,
                "train_lr": train_lr,
                "train_weight_decay": train_weight_decay,
                "n_features": n_features,
                "transcoder_steps": transcoder_steps,
                "transcoder_lr": transcoder_lr,
                "transcoder_l1_coeff": transcoder_l1_coeff,
                "transcoder_aux_penalty": transcoder_aux_penalty,
                "transcoder_activation": transcoder_activation,
                "transcoder_top_k": transcoder_top_k,
                "transcoder_input_unit_norm": transcoder_input_unit_norm,
                "max_grad_norm": max_grad_norm,
                "seed": seed,
            }
        )

    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create or load the target MLP
    logger.info("Creating 2-layer MLP...")
    mlp = TwoLayerMLP(input_size=784, hidden_size=hidden_size, num_classes=10)
    mlp = mlp.to(device)

    total_training_steps = 0
    if pretrained_mlp_path is not None:
        logger.info(f"Loading pretrained MLP from {pretrained_mlp_path}")
        mlp.load_state_dict(torch.load(pretrained_mlp_path, map_location=device, weights_only=True))
        mlp.eval()
    else:
        total_training_steps = train_mlp(
            mlp,
            train_loader,
            device,
            epochs=train_epochs,
            lr=train_lr,
            weight_decay=train_weight_decay,
            log_wandb=wandb_project is not None,
        )

        # Save the trained MLP
        mlp_path = out_path / "trained_mlp.pth"
        torch.save(mlp.state_dict(), mlp_path)
        logger.info(f"Saved trained MLP to {mlp_path}")

    # Evaluate MLP on test set
    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = mlp(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100.0 * correct / total
    logger.info(f"MLP Test accuracy: {test_accuracy:.2f}%")

    if wandb_project:
        wandb.log({"mlp_test/accuracy": test_accuracy}, step=total_training_steps)

    # Create and train the transcoder
    logger.info("Creating transcoder...")
    transcoder = Transcoder(
        d_input=784,
        d_output=hidden_size,
        n_features=n_features,
        activation=transcoder_activation,
        top_k=transcoder_top_k,
        input_unit_norm=transcoder_input_unit_norm,
    )
    transcoder = transcoder.to(device)

    logger.info(f"Transcoder: {784} -> {n_features} -> {hidden_size}")
    logger.info(f"Activation: {transcoder_activation}, Top-K: {transcoder_top_k}")
    logger.info(f"Total parameters: {sum(p.numel() for p in transcoder.parameters())}")

    # Train the transcoder
    train_transcoder(
        transcoder=transcoder,
        target_model=mlp,
        train_loader=train_loader,
        device=device,
        steps=transcoder_steps,
        lr=transcoder_lr,
        l1_coeff=transcoder_l1_coeff,
        aux_penalty=transcoder_aux_penalty,
        max_grad_norm=max_grad_norm,
        log_wandb=wandb_project is not None,
        test_loader=test_loader,
        step_offset=total_training_steps + 1,
    )

    # Save the trained transcoder
    transcoder_path = out_path / "trained_transcoder.pth"
    torch.save(transcoder.state_dict(), transcoder_path)
    logger.info(f"Saved trained transcoder to {transcoder_path}")

    # Final evaluation
    final_metrics = evaluate_transcoder(transcoder, mlp, test_loader, device, n_batches=100)
    logger.info("Final evaluation:")
    logger.info(f"  Reconstruction loss: {final_metrics['recon_loss']:.4f}")
    logger.info(f"  L0 (avg active features): {final_metrics['l0']:.2f}")
    logger.info(f"  Fraction variance explained: {final_metrics['frac_variance_explained']:.4f}")
    logger.info(f"  Dead features: {transcoder.get_num_dead_features()}")

    if wandb_project:
        wandb.log(
            {
                "transcoder_final/recon_loss": final_metrics["recon_loss"],
                "transcoder_final/l0": final_metrics["l0"],
                "transcoder_final/frac_variance_explained": final_metrics[
                    "frac_variance_explained"
                ],
                "transcoder_final/dead_features": transcoder.get_num_dead_features(),
            },
            step=total_training_steps + transcoder_steps,
        )

    # Create visualizations
    logger.info("Creating visualizations...")
    plot_feature_directions(
        transcoder=transcoder,
        out_dir=out_path,
        n_features_to_show=min(n_features, 50),
        log_wandb=wandb_project is not None,
    )

    plot_feature_activations_on_images(
        transcoder=transcoder,
        test_dataset=test_dataset,
        device=device,
        out_dir=out_path,
        n_samples=10,
        n_features_to_show=20,
        log_wandb=wandb_project is not None,
    )

    plot_digit_feature_heatmap(
        transcoder=transcoder,
        test_dataset=test_dataset,
        device=device,
        out_dir=out_path,
        n_samples_per_digit=50,
        n_features_to_show=50,
        log_wandb=wandb_project is not None,
    )

    logger.info("Experiment complete!")
    logger.info(f"All outputs saved to: {out_path}")

    if wandb_project:
        wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(main)

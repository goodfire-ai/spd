"""
Sparse Connection Transcoder

Learns a mapping from fc1 component activations to fc2 component activations,
with L1 sparsity on the encoder/decoder weights (connections) rather than on
the activations. This reveals which input components connect to which output
components through the ReLU nonlinearity.

The actual computation path in the decomposed model is:
    fc2_component_acts = V2^T @ ReLU(U1^T @ fc1_component_acts + b1)

So the transcoder architecture mirrors this:
    predicted = W_dec @ ReLU(W_enc @ input_acts + b_enc) + b_dec

With L1 penalties on W_enc and W_dec to discover sparse connectivity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.configs import ModulePatternInfoConfig
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SparseConnectionTranscoder(nn.Module):
    """Transcoder with L1-sparse weights mapping input component acts to output component acts.

    Architecture: input_acts -> W_enc -> ReLU -> W_dec -> predicted_output_acts

    Sparsity is on W_enc and W_dec entries, not on activations.
    """

    def __init__(self, n_input: int, n_output: int, hidden_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(n_input, hidden_dim)
        self.W_dec = nn.Linear(hidden_dim, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_dec(F.relu(self.W_enc(x)))

    def weight_l1(self) -> torch.Tensor:
        return self.W_enc.weight.abs().sum() + self.W_dec.weight.abs().sum()

    def connectivity_l1(self) -> torch.Tensor:
        """L1 on |W_dec| @ |W_enc|, the absolute-value product.

        This correctly captures nonlinear pathways through the ReLU. If entry
        (o, i) is zero, output o truly cannot depend on input i through any
        hidden unit, regardless of ReLU gating patterns.
        """
        return (self.W_dec.weight.abs() @ self.W_enc.weight.abs()).sum()

    def effective_connectivity(self) -> torch.Tensor:
        """(n_output, n_input) matrix of connection strengths (absolute-value product)."""
        return self.W_dec.weight.abs() @ self.W_enc.weight.abs()


class PairwiseTranscoder(nn.Module):
    """Each input-output pair gets its own scalar ReLU predictor.

    output_j = sum_i ReLU(input_i * w1[i,j] + b[i,j]) * w2[i,j]

    This gives n_input * n_output independent single-neuron pathways.
    Parameters: n_input * n_output * 3 (w1, b, w2 per pair).
    """

    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.w1 = nn.Parameter(torch.randn(n_input, n_output) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_input, n_output))
        self.w2 = nn.Parameter(torch.randn(n_input, n_output) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_input)
        x_expanded = x.unsqueeze(-1)  # (batch, n_input, 1)
        hidden = F.relu(x_expanded * self.w1 + self.b)  # (batch, n_input, n_output)
        return (hidden * self.w2).sum(dim=1)  # (batch, n_output)

    def weight_l1(self) -> torch.Tensor:
        return self.w1.abs().sum() + self.w2.abs().sum()

    def connectivity_l1(self) -> torch.Tensor:
        return (self.w1.abs() * self.w2.abs()).sum()

    def effective_connectivity(self) -> torch.Tensor:
        """(n_output, n_input) matrix of connection strengths."""
        return (self.w1.abs() * self.w2.abs()).T  # (n_output, n_input)

    def pair_strengths(self) -> torch.Tensor:
        """(n_input, n_output) matrix of |w1 * w2| per pair."""
        return (self.w1 * self.w2).abs()


class DirectSparseMap(nn.Module):
    """Single-layer sparse linear mapping (no hidden layer, no nonlinearity).

    Learns W such that output_acts ≈ input_acts @ W, with L1 on W.
    """

    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.W = nn.Linear(n_input, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x)

    def weight_l1(self) -> torch.Tensor:
        return self.W.weight.abs().sum()

    def connectivity_l1(self) -> torch.Tensor:
        return self.W.weight.abs().sum()

    def effective_connectivity(self) -> torch.Tensor:
        return self.W.weight


def load_component_model(out_dir: Path, device: str) -> tuple[ComponentModel, TwoLayerMLP]:
    model = TwoLayerMLP(input_size=784, hidden_size=128, num_classes=10)
    model.load_state_dict(
        torch.load(out_dir / "trained_mlp.pth", map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    module_info = [
        ModulePatternInfoConfig(module_pattern="fc1", C=500),
        ModulePatternInfoConfig(module_pattern="fc2", C=500),
    ]
    module_path_info = expand_module_patterns(model, module_info)
    component_model = ComponentModel(
        target_model=model,
        module_path_info=module_path_info,
        ci_fn_type="linear",
        ci_fn_hidden_dims=[256],
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    checkpoint = torch.load(out_dir / "model_200000.pth", map_location="cpu", weights_only=True)
    component_model.load_state_dict(checkpoint)
    component_model.to(device)
    component_model.eval()

    return component_model, model


def find_alive_components(
    component_model: ComponentModel,
    test_dataset: datasets.MNIST,
    device: str,
    ci_threshold: float = 0.1,
    n_batches: int = 100,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    fc1_components = component_model.components["fc1"]
    fc2_components = component_model.components["fc2"]
    n_fc1 = fc1_components.V.shape[1]
    n_fc2 = fc2_components.V.shape[1]

    fire_count_fc1 = torch.zeros(n_fc1, device=device)
    fire_count_fc2 = torch.zeros(n_fc2, device=device)
    n_samples = 0

    with torch.no_grad():
        for _ in range(n_batches):
            indices = torch.randint(0, len(test_dataset), (batch_size,))
            images = torch.stack([test_dataset[int(i)][0] for i in indices]).to(device)
            images_flat = images.view(batch_size, -1)

            fc1_acts = fc1_components.get_component_acts(images_flat)
            ci_fc1 = component_model.ci_fns["fc1"](fc1_acts)
            fire_count_fc1 += (ci_fc1 > ci_threshold).float().sum(dim=0)

            hidden = F.relu(component_model.target_model.fc1(images_flat))
            fc2_acts = fc2_components.get_component_acts(hidden)
            ci_fc2 = component_model.ci_fns["fc2"](fc2_acts)
            fire_count_fc2 += (ci_fc2 > ci_threshold).float().sum(dim=0)

            n_samples += batch_size

    fire_freq_fc1 = (fire_count_fc1 / n_samples).cpu().numpy()
    fire_freq_fc2 = (fire_count_fc2 / n_samples).cpu().numpy()

    alive_fc1 = np.where(fire_freq_fc1 > 0.001)[0]
    alive_fc2 = np.where(fire_freq_fc2 > 0.001)[0]

    return alive_fc1, alive_fc2


def collect_activation_pairs(
    component_model: ComponentModel,
    dataset: datasets.MNIST,
    device: str,
    alive_fc1: np.ndarray,
    alive_fc2: np.ndarray,
    n_samples: int = 50000,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect paired (fc1_component_acts, fc2_component_acts) for alive components."""
    fc1_components = component_model.components["fc1"]
    fc2_components = component_model.components["fc2"]

    alive_fc1_t = torch.tensor(alive_fc1, device=device)
    alive_fc2_t = torch.tensor(alive_fc2, device=device)

    all_fc1 = []
    all_fc2 = []
    n_collected = 0
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Collecting activations"):
            indices = torch.randint(0, len(dataset), (batch_size,))
            images = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
            images_flat = images.view(batch_size, -1)

            fc1_acts = fc1_components.get_component_acts(images_flat)
            all_fc1.append(fc1_acts[:, alive_fc1_t].cpu())

            hidden = F.relu(component_model.target_model.fc1(images_flat))
            fc2_acts = fc2_components.get_component_acts(hidden)
            all_fc2.append(fc2_acts[:, alive_fc2_t].cpu())

            n_collected += batch_size

    X = torch.cat(all_fc1, dim=0)[:n_samples]
    Y = torch.cat(all_fc2, dim=0)[:n_samples]
    return X, Y


Transcoder = SparseConnectionTranscoder | DirectSparseMap | PairwiseTranscoder


def _weight_sparsity(model: Transcoder, threshold: float = 0.01) -> float:
    """Average fraction of below-threshold weights per hidden neuron.

    For each hidden neuron, count incoming weights (W_enc row) and outgoing weights
    (W_dec column) that are below threshold, divided by total (n_input + n_output).
    Return the average across hidden neurons.

    For DirectSparseMap (no hidden layer), average fraction of below-threshold
    weights per output neuron.
    """
    with torch.no_grad():
        if isinstance(model, SparseConnectionTranscoder):
            W_enc = model.W_enc.weight.abs()  # (hidden_dim, n_input)
            W_dec = model.W_dec.weight.abs()  # (n_output, hidden_dim)
            n_input = W_enc.shape[1]
            n_output = W_dec.shape[0]
            incoming_dead = (W_enc < threshold).float().sum(dim=1)  # (hidden_dim,)
            outgoing_dead = (W_dec < threshold).float().sum(dim=0)  # (hidden_dim,)
            per_neuron = (incoming_dead + outgoing_dead) / (n_input + n_output)
            return per_neuron.mean().item()
        elif isinstance(model, PairwiseTranscoder):
            strengths = model.pair_strengths()  # (n_input, n_output)
            return (strengths < threshold).float().mean().item()
        else:
            W = model.W.weight.abs()  # (n_output, n_input)
            return (threshold > W).float().mean().item()


def train_transcoder(
    transcoder: Transcoder,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    device: str,
    l1_coeff: float = 0.01,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 512,
) -> list[dict[str, float]]:
    transcoder.to(device)
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history: list[dict[str, float]] = []

    warmup_end = int(0.8 * epochs)

    for epoch in range(epochs):
        l1_scale = min(epoch / warmup_end, 1.0) if warmup_end > 0 else 1.0
        transcoder.train()
        epoch_mse = 0.0
        epoch_l1 = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = transcoder(x_batch)
            mse_loss = F.mse_loss(pred, y_batch)
            l1_loss = transcoder.weight_l1()
            loss = mse_loss + l1_coeff * l1_scale * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            n_batches += 1

        scheduler.step()

        avg_mse = epoch_mse / n_batches
        avg_l1 = epoch_l1 / n_batches

        # Validation
        transcoder.eval()
        with torch.no_grad():
            val_pred = transcoder(val_X.to(device))
            val_mse = F.mse_loss(val_pred, val_Y.to(device)).item()

            ss_res = ((val_Y.to(device) - val_pred) ** 2).sum(dim=0)
            ss_tot = ((val_Y.to(device) - val_Y.to(device).mean(dim=0)) ** 2).sum(dim=0)
            r2_per_output = (1 - ss_res / ss_tot.clamp(min=1e-8)).cpu().numpy()
            mean_r2 = float(r2_per_output.mean())

            conn_sparsity = _weight_sparsity(transcoder)

        record = {
            "epoch": epoch,
            "train_mse": avg_mse,
            "train_l1": avg_l1,
            "val_mse": val_mse,
            "mean_r2": mean_r2,
            "conn_sparsity": conn_sparsity,
        }
        history.append(record)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_mse={avg_mse:.6f} | "
                f"val_mse={val_mse:.6f} | "
                f"R²={mean_r2:.4f} | "
                f"conn_sparse={conn_sparsity:.1%}"
            )

    return history


def prune_and_finetune(
    transcoder: Transcoder,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    device: str,
    threshold: float = 0.01,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 512,
) -> list[dict[str, float]]:
    """Hard-prune weights below threshold, then fine-tune surviving weights."""
    transcoder.to(device)

    # Build binary masks and zero out pruned weights
    masks: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in transcoder.named_parameters():
            if "weight" in name:
                mask = (param.abs() >= threshold).float()
                param.mul_(mask)
                masks[name] = mask

    n_total = sum(m.numel() for m in masks.values())
    n_alive = sum(m.sum().item() for m in masks.values())
    print(
        f"\nPruning: {n_total - n_alive:.0f}/{n_total} weights zeroed "
        f"({(n_total - n_alive) / n_total:.1%} pruned)"
    )

    optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)

    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        transcoder.train()
        epoch_mse = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = transcoder(x_batch)
            loss = F.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # Zero gradients for pruned weights so they stay dead
            with torch.no_grad():
                for name, param in transcoder.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name])

            optimizer.step()

            epoch_mse += loss.item()
            n_batches += 1

        avg_mse = epoch_mse / n_batches

        transcoder.eval()
        with torch.no_grad():
            val_pred = transcoder(val_X.to(device))
            val_mse = F.mse_loss(val_pred, val_Y.to(device)).item()
            ss_res = ((val_Y.to(device) - val_pred) ** 2).sum(dim=0)
            ss_tot = ((val_Y.to(device) - val_Y.to(device).mean(dim=0)) ** 2).sum(dim=0)
            mean_r2 = float((1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item())
            conn_sparsity = _weight_sparsity(transcoder, threshold)

        record = {
            "epoch": epoch,
            "train_mse": avg_mse,
            "train_l1": 0.0,
            "val_mse": val_mse,
            "mean_r2": mean_r2,
            "conn_sparsity": conn_sparsity,
        }
        history.append(record)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Finetune {epoch:3d} | "
                f"train_mse={avg_mse:.6f} | "
                f"val_mse={val_mse:.6f} | "
                f"R²={mean_r2:.4f} | "
                f"conn_sparse={conn_sparsity:.1%}"
            )

    return history


def _apply_topk_mask(weight: torch.Tensor, k: int) -> None:
    """Zero out all but the top-k entries by absolute value in each row."""
    with torch.no_grad():
        if k >= weight.shape[1]:
            return
        _, top_indices = weight.abs().topk(k, dim=1)
        mask = torch.zeros_like(weight)
        mask.scatter_(1, top_indices, 1.0)
        weight.mul_(mask)


def train_transcoder_topk(
    transcoder: SparseConnectionTranscoder,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    device: str,
    target_k: int = 3,
    lr: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 512,
) -> list[dict[str, float]]:
    """Train with top-K sparsification: per hidden neuron, keep only K largest weights.

    K anneals linearly from n_input (dense) to target_k over training.
    Applied to both encoder rows and decoder columns (transposed to rows).
    """
    transcoder.to(device)
    n_input = transcoder.W_enc.weight.shape[1]
    n_output = transcoder.W_dec.weight.shape[0]

    optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        # Anneal K from n_input -> target_k (encoder) and n_output -> target_k (decoder)
        frac = min(epoch / (0.8 * epochs), 1.0)
        k_enc = max(target_k, round(n_input - frac * (n_input - target_k)))
        k_dec = max(target_k, round(n_output - frac * (n_output - target_k)))

        transcoder.train()
        epoch_mse = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = transcoder(x_batch)
            loss = F.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Apply top-K mask after each step
            _apply_topk_mask(transcoder.W_enc.weight, k_enc)
            # Decoder: top-K per hidden neuron = per column, so transpose, mask, transpose back
            transcoder.W_dec.weight.data = transcoder.W_dec.weight.data.T.contiguous()
            _apply_topk_mask(transcoder.W_dec.weight, k_dec)
            transcoder.W_dec.weight.data = transcoder.W_dec.weight.data.T.contiguous()

            epoch_mse += loss.item()
            n_batches += 1

        scheduler.step()
        avg_mse = epoch_mse / n_batches

        transcoder.eval()
        with torch.no_grad():
            val_pred = transcoder(val_X.to(device))
            val_mse = F.mse_loss(val_pred, val_Y.to(device)).item()
            ss_res = ((val_Y.to(device) - val_pred) ** 2).sum(dim=0)
            ss_tot = ((val_Y.to(device) - val_Y.to(device).mean(dim=0)) ** 2).sum(dim=0)
            mean_r2 = float((1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item())
            conn_sparsity = _weight_sparsity(transcoder)

        record = {
            "epoch": epoch,
            "train_mse": avg_mse,
            "train_l1": 0.0,
            "val_mse": val_mse,
            "mean_r2": mean_r2,
            "conn_sparsity": conn_sparsity,
        }
        history.append(record)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_mse={avg_mse:.6f} | "
                f"val_mse={val_mse:.6f} | "
                f"R²={mean_r2:.4f} | "
                f"conn_sparse={conn_sparsity:.1%} | "
                f"k_enc={k_enc} k_dec={k_dec}"
            )

    return history


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_mse"] for h in history], label="Train MSE")
    axes[0].plot(epochs, [h["val_mse"] for h in history], label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(epochs, [h["mean_r2"] for h in history])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean R²")
    axes[1].set_title("Prediction Quality")
    axes[1].set_ylim(-0.1, 1.05)

    axes[2].plot(epochs, [h["conn_sparsity"] for h in history])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Fraction below threshold")
    axes[2].set_title("Weight Sparsity (per neuron avg)")
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_connectivity(
    transcoder: Transcoder,
    alive_fc1: np.ndarray,
    alive_fc2: np.ndarray,
    output_path: Path,
    threshold: float = 0.01,
) -> None:
    """Visualize the effective connectivity between input and output components."""
    with torch.no_grad():
        effective_W = transcoder.effective_connectivity().cpu().numpy()  # (n_output, n_input)

    fig, ax = plt.subplots(figsize=(max(8, len(alive_fc1) * 0.4), max(4, len(alive_fc2) * 0.4)))

    vmax = np.abs(effective_W).max()
    im = ax.imshow(
        effective_W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    ax.set_xlabel("Input component (fc1)")
    ax.set_ylabel("Output component (fc2)")
    ax.set_xticks(range(len(alive_fc1)))
    ax.set_xticklabels([f"C{c}" for c in alive_fc1], rotation=90, fontsize=7)
    ax.set_yticks(range(len(alive_fc2)))
    ax.set_yticklabels([f"C{c}" for c in alive_fc2], fontsize=8)
    ax.set_title("Effective connectivity")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved connectivity plot to {output_path}")

    # Print sparsity stats
    n_nonzero = (np.abs(effective_W) > threshold).sum()
    n_total = effective_W.size
    print(
        f"Effective connectivity: {n_nonzero}/{n_total} entries above {threshold} "
        f"({n_nonzero / n_total:.1%})"
    )

    n_inputs_per_output = (np.abs(effective_W) > threshold).sum(axis=1)
    print(
        f"Inputs per output component: mean={n_inputs_per_output.mean():.1f}, "
        f"median={np.median(n_inputs_per_output):.0f}, "
        f"max={n_inputs_per_output.max()}"
    )


def plot_per_component_r2(
    transcoder: Transcoder,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    alive_fc2: np.ndarray,
    device: str,
    output_path: Path,
) -> None:
    transcoder.eval()
    with torch.no_grad():
        val_pred = transcoder(val_X.to(device))
        ss_res = ((val_Y.to(device) - val_pred) ** 2).sum(dim=0)
        ss_tot = ((val_Y.to(device) - val_Y.to(device).mean(dim=0)) ** 2).sum(dim=0)
        r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).cpu().numpy()

    sorted_idx = np.argsort(r2)[::-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if v > 0.5 else "orange" if v > 0 else "red" for v in r2[sorted_idx]]
    ax.bar(range(len(r2)), r2[sorted_idx], color=colors, width=1.0)
    ax.set_xlabel("Output component (sorted)")
    ax.set_ylabel("R²")
    ax.set_title(f"Per-output-component R² (mean={r2.mean():.3f}, median={np.median(r2):.3f})")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="R²=0.5")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-component R² plot to {output_path}")
    print(f"R² > 0.5: {(r2 > 0.5).sum()}/{len(r2)} components")
    print(f"R² > 0.8: {(r2 > 0.8).sum()}/{len(r2)} components")


def _barycenter_order(adj: np.ndarray, fixed_order: np.ndarray) -> np.ndarray:
    """Order nodes to minimize crossings using barycenter heuristic.

    Args:
        adj: (n_free, n_fixed) adjacency matrix (absolute weights).
        fixed_order: positions of fixed-side nodes (0-indexed).
    Returns:
        Permutation of free-side node indices sorted by weighted barycenter.
    """
    n_free = adj.shape[0]
    barycenters = np.zeros(n_free)
    for i in range(n_free):
        weights = adj[i]
        total = weights.sum()
        if total > 0:
            barycenters[i] = (weights * fixed_order).sum() / total
        else:
            barycenters[i] = fixed_order.mean()
    return np.argsort(barycenters)


def plot_network_graph(
    transcoder: SparseConnectionTranscoder,
    alive_fc1: np.ndarray,
    alive_fc2: np.ndarray,
    output_path: Path,
    threshold: float = 0.01,
) -> None:
    """Draw the transcoder as a 3-layer network graph with alive edges."""
    with torch.no_grad():
        W_enc = transcoder.W_enc.weight.detach().cpu().numpy()  # (hidden, n_input)
        W_dec = transcoder.W_dec.weight.detach().cpu().numpy()  # (n_output, hidden)

    n_input = len(alive_fc1)
    n_hidden = W_enc.shape[0]
    n_output = len(alive_fc2)

    enc_alive = np.abs(W_enc) > threshold  # (hidden, n_input)
    dec_alive = np.abs(W_dec) > threshold  # (n_output, hidden)

    # Only keep hidden neurons that have at least one alive input AND one alive output
    hidden_alive_mask = enc_alive.any(axis=1) & dec_alive.any(axis=0)
    alive_hidden_idx = np.where(hidden_alive_mask)[0]
    n_alive_hidden = len(alive_hidden_idx)

    W_enc_alive = W_enc[alive_hidden_idx]  # (n_alive_hidden, n_input) - signed
    W_dec_alive = W_dec[:, alive_hidden_idx]  # (n_output, n_alive_hidden) - signed

    # Order layers to minimize crossings using barycenter heuristic (use abs for ordering).
    input_positions = np.arange(n_input, dtype=float)
    hidden_perm = _barycenter_order(np.abs(W_enc_alive), input_positions)
    W_enc_alive = W_enc_alive[hidden_perm]
    W_dec_alive = W_dec_alive[:, hidden_perm]
    alive_hidden_idx = alive_hidden_idx[hidden_perm]

    hidden_positions = np.arange(n_alive_hidden, dtype=float)
    output_perm = _barycenter_order(np.abs(W_dec_alive), hidden_positions)
    W_dec_alive = W_dec_alive[output_perm]
    output_labels = alive_fc2[output_perm]
    # Keep input labels in original order
    input_labels = alive_fc1

    # Layout: 3 columns
    x_in, x_hid, x_out = 0.0, 1.0, 2.0

    def y_positions(n: int) -> np.ndarray:
        if n == 1:
            return np.array([0.0])
        return np.linspace(0, 1, n)

    y_in = y_positions(n_input)
    y_hid = y_positions(n_alive_hidden)
    y_out = y_positions(n_output)

    max_enc = np.abs(W_enc_alive).max() if W_enc_alive.size > 0 else 1.0
    max_dec = np.abs(W_dec_alive).max() if W_dec_alive.size > 0 else 1.0

    fig_height = max(6, max(n_input, n_alive_hidden, n_output) * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Draw edges: encoder (input -> hidden)
    for h in range(n_alive_hidden):
        for i in range(n_input):
            w = W_enc_alive[h, i]
            aw = abs(w)
            if aw > threshold:
                alpha = 0.15 + 0.7 * (aw / max_enc)
                lw = 0.5 + 2.5 * (aw / max_enc)
                color = "steelblue" if w > 0 else "firebrick"
                ax.plot(
                    [x_in, x_hid],
                    [y_in[i], y_hid[h]],
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    zorder=1,
                )

    # Draw edges: decoder (hidden -> output)
    for o in range(n_output):
        for h in range(n_alive_hidden):
            w = W_dec_alive[o, h]
            aw = abs(w)
            if aw > threshold:
                alpha = 0.15 + 0.7 * (aw / max_dec)
                lw = 0.5 + 2.5 * (aw / max_dec)
                color = "steelblue" if w > 0 else "firebrick"
                ax.plot(
                    [x_hid, x_out],
                    [y_hid[h], y_out[o]],
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    zorder=1,
                )

    # Draw nodes
    node_size = 80
    ax.scatter(
        [x_in] * n_input,
        y_in,
        s=node_size,
        c="steelblue",
        zorder=3,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.scatter(
        [x_hid] * n_alive_hidden,
        y_hid,
        s=node_size,
        c="gray",
        zorder=3,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.scatter(
        [x_out] * n_output,
        y_out,
        s=node_size,
        c="firebrick",
        zorder=3,
        edgecolors="black",
        linewidths=0.5,
    )

    # Labels
    for i, lbl in enumerate(input_labels):
        ax.text(x_in - 0.05, y_in[i], f"C{lbl}", ha="right", va="center", fontsize=7)
    for i, idx in enumerate(alive_hidden_idx):
        ax.text(
            x_hid, y_hid[i] + 0.015, f"h{idx}", ha="center", va="bottom", fontsize=5, color="gray"
        )
    for i, lbl in enumerate(output_labels):
        ax.text(x_out + 0.05, y_out[i], f"C{lbl}", ha="left", va="center", fontsize=7)

    # Column headers
    ax.text(
        x_in,
        -0.06,
        f"fc1 inputs ({n_input})",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x_hid,
        -0.06,
        f"hidden ({n_alive_hidden}/{n_hidden})",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x_out,
        -0.06,
        f"fc2 outputs ({n_output})",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.1, 1.08)
    ax.axis("off")
    ax.set_title("Sparse Connection Transcoder", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    n_enc_edges = (np.abs(W_enc_alive) > threshold).sum()
    n_dec_edges = (np.abs(W_dec_alive) > threshold).sum()
    print(f"Saved network graph to {output_path}")
    print(f"  Alive hidden neurons: {n_alive_hidden}/{n_hidden}")
    print(f"  Encoder edges: {n_enc_edges}/{n_alive_hidden * n_input}")
    print(f"  Decoder edges: {n_dec_edges}/{n_alive_hidden * n_output}")


def evaluate_mnist_accuracy(
    transcoder: Transcoder,
    component_model: ComponentModel,
    test_dataset: datasets.MNIST,
    alive_fc1: np.ndarray,
    alive_fc2: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate MNIST accuracy by replacing the ReLU nonlinearity with the transcoder.

    Original model: logits = fc2(ReLU(fc1(input)))
    With transcoder: logits = transcoder(V1^T @ input)[alive] @ U2[alive] + b2

    Also computes the original model accuracy and the baseline (ground truth component path)
    for comparison.
    """
    fc1_components = component_model.components["fc1"]
    fc2_components = component_model.components["fc2"]

    V1 = fc1_components.V.to(device)  # (784, C_fc1)
    U2 = fc2_components.U.to(device)  # (C_fc2, 10)
    b2 = component_model.target_model.fc2.bias.to(device)  # (10,)
    b1 = component_model.target_model.fc1.bias.to(device)  # (128,)
    U1 = fc1_components.U.to(device)  # (C_fc1, 128)

    alive_fc1_t = torch.tensor(alive_fc1, device=device)
    alive_fc2_t = torch.tensor(alive_fc2, device=device)

    transcoder.to(device)
    transcoder.eval()

    correct_original = 0
    correct_transcoder = 0
    correct_baseline = 0
    total = 0

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images_flat = images.view(images.size(0), -1)

            # Original model
            original_logits = component_model.target_model(images)
            correct_original += (original_logits.argmax(dim=1) == labels).sum().item()

            # Baseline: ground truth component path
            fc1_acts = images_flat @ V1  # (batch, C_fc1)
            fc1_alive = fc1_acts[:, alive_fc1_t]
            hidden = F.relu(fc1_acts @ U1 + b1)  # (batch, 128)
            fc2_acts_gt = hidden @ fc2_components.V  # (batch, C_fc2)
            baseline_logits = fc2_acts_gt[:, alive_fc2_t] @ U2[alive_fc2_t] + b2
            correct_baseline += (baseline_logits.argmax(dim=1) == labels).sum().item()

            # Transcoder: replace nonlinearity
            fc2_acts_pred = transcoder(fc1_alive)  # (batch, n_alive_fc2)
            transcoder_logits = fc2_acts_pred @ U2[alive_fc2_t] + b2
            correct_transcoder += (transcoder_logits.argmax(dim=1) == labels).sum().item()

            total += labels.size(0)

    acc_original = correct_original / total
    acc_baseline = correct_baseline / total
    acc_transcoder = correct_transcoder / total

    print("\n" + "=" * 60)
    print("MNIST CLASSIFICATION ACCURACY")
    print("=" * 60)
    print(f"Original model:     {acc_original:.2%}")
    print(f"Baseline (V1→U1→ReLU→V2→U2, alive only): {acc_baseline:.2%}")
    print(f"Transcoder (V1→transcoder→U2):            {acc_transcoder:.2%}")

    return {
        "acc_original": acc_original,
        "acc_baseline": acc_baseline,
        "acc_transcoder": acc_transcoder,
    }


def main(
    output_dir: str = "output/mnist_experiment_v2",
    direct: bool = False,
    pairwise: bool = False,
    hidden_dim: int = 1024,
    l1_coeff: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 1000,
    n_train_samples: int = 50000,
    n_val_samples: int = 10000,
    topk: int | None = 1,
    seed: int = 42,
) -> None:
    """Train a sparse connection transcoder from fc1 to fc2 component activations.

    Args:
        output_dir: Directory containing the trained SPD model
        direct: If True, use a direct sparse linear map (no hidden layer).
        hidden_dim: Hidden dimension of the transcoder (ignored if direct=True)
        l1_coeff: L1 penalty coefficient on weights (applied to weight_l1() which uses sum)
        lr: Learning rate
        epochs: Number of training epochs
        n_train_samples: Number of training activation pairs to collect
        n_val_samples: Number of validation activation pairs
        topk: If set, use top-K sparsification instead of L1. K anneals from all inputs to this value.
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)

    print("Loading component model...")
    component_model, _ = load_component_model(out_dir, device)

    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    print("Finding alive components...")
    alive_fc1, alive_fc2 = find_alive_components(component_model, test_dataset, device)
    print(f"Alive fc1: {len(alive_fc1)}, alive fc2: {len(alive_fc2)}")

    print(f"Collecting {n_train_samples} training pairs...")
    train_X, train_Y = collect_activation_pairs(
        component_model, train_dataset, device, alive_fc1, alive_fc2, n_samples=n_train_samples
    )
    print(f"Collecting {n_val_samples} validation pairs...")
    val_X, val_Y = collect_activation_pairs(
        component_model, test_dataset, device, alive_fc1, alive_fc2, n_samples=n_val_samples
    )

    print(f"\nInput dim (alive fc1): {train_X.shape[1]}")
    print(f"Output dim (alive fc2): {train_Y.shape[1]}")

    # Baseline: ground truth path through U1, ReLU, V2
    with torch.no_grad():
        U1 = component_model.components["fc1"].U.cpu()
        V2 = component_model.components["fc2"].V.cpu()
        b1 = component_model.target_model.fc1.bias.cpu()

        alive_fc1_t = torch.tensor(alive_fc1)
        alive_fc2_t = torch.tensor(alive_fc2)

        U1_alive = U1[alive_fc1_t, :]
        V2_alive = V2[:, alive_fc2_t]

        baseline_pred = F.relu(val_X @ U1_alive + b1) @ V2_alive
        baseline_mse = F.mse_loss(baseline_pred, val_Y).item()

        ss_res = ((val_Y - baseline_pred) ** 2).sum(dim=0)
        ss_tot = ((val_Y - val_Y.mean(dim=0)) ** 2).sum(dim=0)
        baseline_r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).numpy()

    print("\n--- Baseline (ground truth path U1 @ V2) ---")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"Mean R²: {baseline_r2.mean():.4f}")

    # Build the model
    if pairwise:
        transcoder: Transcoder = PairwiseTranscoder(
            n_input=len(alive_fc1),
            n_output=len(alive_fc2),
        )
        mode_name = "pairwise"
        print("\nMode: pairwise (one ReLU neuron per input-output pair)")
    elif direct:
        transcoder = DirectSparseMap(
            n_input=len(alive_fc1),
            n_output=len(alive_fc2),
        )
        mode_name = "direct"
        print("\nMode: direct sparse linear map")
    else:
        transcoder = SparseConnectionTranscoder(
            n_input=len(alive_fc1),
            n_output=len(alive_fc2),
            hidden_dim=hidden_dim,
        )
        mode_name = f"transcoder_h{hidden_dim}"
        print(f"\nMode: transcoder with hidden_dim={hidden_dim}")

    n_params = sum(p.numel() for p in transcoder.parameters())
    print(f"Parameters: {n_params:,}")

    if pairwise:
        print("\n--- Training (pairwise, MSE only) ---")
        history = train_transcoder(
            transcoder=transcoder,
            train_X=train_X,
            train_Y=train_Y,
            val_X=val_X,
            val_Y=val_Y,
            device=device,
            l1_coeff=0.0,
            lr=lr,
            epochs=epochs,
        )
    elif topk is not None and isinstance(transcoder, SparseConnectionTranscoder):
        print(f"Top-K target: {topk}")
        print("\n--- Training (top-K) ---")
        history = train_transcoder_topk(
            transcoder=transcoder,
            train_X=train_X,
            train_Y=train_Y,
            val_X=val_X,
            val_Y=val_Y,
            device=device,
            target_k=topk,
            lr=lr,
            epochs=epochs,
        )
    else:
        print(f"L1 coefficient: {l1_coeff}")
        print("\n--- Training (L1) ---")
        history = train_transcoder(
            transcoder=transcoder,
            train_X=train_X,
            train_Y=train_Y,
            val_X=val_X,
            val_Y=val_Y,
            device=device,
            l1_coeff=l1_coeff,
            lr=lr,
            epochs=epochs,
        )

        # Prune and fine-tune
        print("\n--- Pruning + Fine-tuning ---")
        finetune_history = prune_and_finetune(
            transcoder=transcoder,
            train_X=train_X,
            train_Y=train_Y,
            val_X=val_X,
            val_Y=val_Y,
            device=device,
            epochs=500,
        )
        history.extend(finetune_history)

    # Save results
    results_dir = out_dir / f"sparse_transcoder_{mode_name}"
    results_dir.mkdir(exist_ok=True)

    plot_training_curves(history, results_dir / "training_curves.png")
    plot_connectivity(transcoder, alive_fc1, alive_fc2, results_dir / "connectivity.png")
    plot_per_component_r2(
        transcoder, val_X, val_Y, alive_fc2, device, results_dir / "per_component_r2.png"
    )
    if isinstance(transcoder, SparseConnectionTranscoder):
        plot_network_graph(transcoder, alive_fc1, alive_fc2, results_dir / "network_graph.png")

    torch.save(transcoder.state_dict(), results_dir / "transcoder.pth")
    print(f"\nSaved to {results_dir}")

    # Evaluate MNIST accuracy
    evaluate_mnist_accuracy(
        transcoder=transcoder,
        component_model=component_model,
        test_dataset=test_dataset,
        alive_fc1=alive_fc1,
        alive_fc2=alive_fc2,
        device=device,
    )

    # Summary
    final = history[-1]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (U1@V2 path): MSE={baseline_mse:.6f}, R²={baseline_r2.mean():.4f}")
    if topk is None:
        pre_prune = history[epochs - 1]
        print(
            f"Before pruning:        MSE={pre_prune['val_mse']:.6f}, R²={pre_prune['mean_r2']:.4f}, sparsity={pre_prune['conn_sparsity']:.1%}"
        )
        print(
            f"After prune+finetune:  MSE={final['val_mse']:.6f}, R²={final['mean_r2']:.4f}, sparsity={final['conn_sparsity']:.1%}"
        )
    else:
        print(
            f"Top-K (k={topk}):      MSE={final['val_mse']:.6f}, R²={final['mean_r2']:.4f}, sparsity={final['conn_sparsity']:.1%}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

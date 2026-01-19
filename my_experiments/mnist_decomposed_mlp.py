"""Train a decomposed MLP on MNIST from scratch.

Instead of training an MLP and then decomposing it, we train a decomposed model directly.
The model uses:
- LinearComponents (V, U matrices) instead of regular Linear layers
- CI functions to predict which components should be active
- Stochastic sampling based on CI during training
- CE loss + importance minimality loss to encourage sparsity
"""

from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.log import logger
from spd.metrics.importance_minimality_loss import importance_minimality_loss
from spd.models.components import LinearCiFn, MLPCiFn
from spd.models.sigmoids import SIGMOID_TYPES
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.module_utils import init_param_


class AliveTracker:
    """Track which components are alive based on recent mask activity.

    A component is considered alive if it has been active (mask > threshold) within
    the last n_batches_until_dead batches.
    """

    def __init__(
        self,
        module_to_c: dict[str, int],
        device: str,
        n_batches_until_dead: int,
        threshold: float = 0.5,
    ):
        self.n_batches_until_dead = n_batches_until_dead
        self.threshold = threshold
        self.n_batches_since_active: dict[str, Tensor] = {
            name: torch.zeros(c, dtype=torch.int64, device=device)
            for name, c in module_to_c.items()
        }

    def update(self, masks: dict[str, Float[Tensor, "... C"]]) -> None:
        """Update tracking based on mask values from a batch."""
        for name, mask in masks.items():
            # Component is active if mask > threshold for any sample in batch
            # threshold=0.5 works for both binary (0/1) and soft masks
            active = (mask > self.threshold).any(dim=0)  # (C,)
            self.n_batches_since_active[name] = torch.where(
                active,
                torch.zeros_like(self.n_batches_since_active[name]),
                self.n_batches_since_active[name] + 1,
            )

    def compute(self) -> dict[str, int]:
        """Return number of alive components per module."""
        return {
            name: int((counts < self.n_batches_until_dead).sum().item())
            for name, counts in self.n_batches_since_active.items()
        }


def calc_l0(mask: Float[Tensor, "... C"]) -> float:
    """Calculate L0: average number of active components per sample.

    Args:
        mask: Component mask with shape (..., C). Binary for Bernoulli sampling,
              soft values for deterministic.

    Returns:
        Average number of active components per sample (sum of mask values).
    """
    return mask.sum(-1).mean().item()


class DecomposedLinear(nn.Module):
    """A linear layer decomposed into C components with learned causal importance.

    Instead of a single weight matrix W, we have:
    - V: (d_in, C) - projects input to C component activations
    - U: (C, d_out) - projects component activations to output
    - ci_fn: predicts which components should be active given component activations

    The effective weight is V @ U (shape d_in x d_out -> transposed gives d_out x d_in).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        C: int,
        ci_fn_type: str,
        ci_fn_hidden_dims: list[int],
        use_normal_sigmoid: bool,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.C = C
        self.use_normal_sigmoid = use_normal_sigmoid

        # Component matrices
        self.V = nn.Parameter(torch.empty(d_in, C))
        self.U = nn.Parameter(torch.empty(C, d_out))
        init_param_(self.V, fan_val=d_in, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        # Bias (not decomposed, always enabled)
        self.bias = nn.Parameter(torch.zeros(d_out))

        # CI function
        if ci_fn_type == "linear":
            self.ci_fn = LinearCiFn(C=C)
        elif ci_fn_type == "mlp":
            self.ci_fn = MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)
        else:
            raise ValueError(f"Unknown ci_fn_type: {ci_fn_type}")

        # Sigmoid functions for CI
        if use_normal_sigmoid:
            self.sigmoid_fn = torch.sigmoid
        else:
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """Effective weight matrix (V @ U).T to match nn.Linear convention."""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        """Project input to component activations."""
        return einops.einsum(x, self.V, "... d_in, d_in C -> ... C")

    def forward(
        self,
        x: Float[Tensor, "... d_in"],
        sampling: str,
        return_extras: bool,
    ) -> (
        Float[Tensor, "... d_out"]
        | tuple[Float[Tensor, "... d_out"], Float[Tensor, "... C"], Float[Tensor, "... C"]]
    ):
        """Forward pass with optional stochastic component sampling.

        Args:
            x: Input tensor
            sampling: "none" for deterministic, "bernoulli" for stochastic sampling
            return_extras: Whether to return CI values and mask

        Returns:
            Output tensor, and optionally (ci_for_loss, mask)
        """
        # Get component activations
        component_acts = self.get_component_acts(x)  # (... C)

        # Compute CI values
        ci_pre_sigmoid = self.ci_fn(component_acts)  # (... C)

        if self.use_normal_sigmoid:
            # Normal sigmoid for both masking and loss
            ci = self.sigmoid_fn(ci_pre_sigmoid)
            if sampling == "bernoulli":
                mask = torch.bernoulli(ci)
            elif sampling == "none":
                mask = ci
            else:
                raise ValueError(f"Unknown sampling: {sampling}")
            ci_for_loss = ci
        else:
            # Leaky hard sigmoids (SPD default)
            if sampling == "bernoulli":
                # Stochastic sampling: sample from Bernoulli with CI as probability
                # Use reparameterization trick for gradients
                ci_for_sampling = 1.05 * ci_pre_sigmoid - 0.05 * torch.rand_like(ci_pre_sigmoid)
                ci_lower = self.lower_leaky_fn(ci_for_sampling)
                mask = torch.bernoulli(ci_lower)
            elif sampling == "none":
                # Deterministic: use soft CI values
                ci_lower = self.lower_leaky_fn(ci_pre_sigmoid)
                mask = ci_lower
            else:
                raise ValueError(f"Unknown sampling: {sampling}")
            ci_for_loss = self.upper_leaky_fn(ci_pre_sigmoid)

        # Apply mask to component activations
        masked_acts = component_acts * mask  # (... C)

        # Project to output
        out = einops.einsum(masked_acts, self.U, "... C, C d_out -> ... d_out") + self.bias

        if return_extras:
            return out, ci_for_loss, mask

        return out


class DecomposedMLP(nn.Module):
    """A 2-layer MLP with decomposed linear layers.

    Architecture: input -> DecomposedLinear -> ReLU -> DecomposedLinear -> output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        n_components: int,
        ci_fn_type: str,
        ci_fn_hidden_dims: list[int],
        use_normal_sigmoid: bool,
    ):
        super().__init__()
        self.fc1 = DecomposedLinear(
            d_in=input_size,
            d_out=hidden_size,
            C=n_components,
            ci_fn_type=ci_fn_type,
            ci_fn_hidden_dims=ci_fn_hidden_dims,
            use_normal_sigmoid=use_normal_sigmoid,
        )
        self.fc2 = DecomposedLinear(
            d_in=hidden_size,
            d_out=num_classes,
            C=n_components,
            ci_fn_type=ci_fn_type,
            ci_fn_hidden_dims=ci_fn_hidden_dims,
            use_normal_sigmoid=use_normal_sigmoid,
        )

    def forward(
        self,
        x: torch.Tensor,
        sampling: str,
        return_extras: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch, 784) or (batch, 1, 28, 28)
            sampling: "none" for deterministic, "bernoulli" for stochastic
            return_extras: Whether to return CI values and masks for each layer

        Returns:
            Logits, and optionally (dict of CI values, dict of masks) per layer
        """
        x = x.view(x.size(0), -1)  # Flatten

        if return_extras:
            h, ci1, mask1 = self.fc1(x, sampling=sampling, return_extras=True)
            h = F.relu(h)
            out, ci2, mask2 = self.fc2(h, sampling=sampling, return_extras=True)
            return out, {"fc1": ci1, "fc2": ci2}, {"fc1": mask1, "fc2": mask2}
        else:
            h = self.fc1(x, sampling=sampling, return_extras=False)
            h = F.relu(h)
            out = self.fc2(h, sampling=sampling, return_extras=False)
            return out


def train_decomposed_mlp(
    model: DecomposedMLP,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    importance_coeff: float,
    pnorm: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float,
    p_anneal_end_frac: float,
    sampling: str,
    n_batches_until_dead: int,
    log_wandb: bool,
) -> int:
    """Train the decomposed MLP on MNIST.

    Args:
        model: DecomposedMLP model
        train_loader: MNIST training data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        importance_coeff: Coefficient for importance minimality loss
        pnorm: Starting p-norm for importance minimality (smaller = sparser)
        p_anneal_start_frac: Fraction of training to start annealing p
        p_anneal_final_p: Final p value after annealing
        p_anneal_end_frac: Fraction of training to finish annealing p
        sampling: "bernoulli" for stochastic, "none" for deterministic
        n_batches_until_dead: Batches without activity before component is considered dead
        log_wandb: Whether to log to W&B

    Returns:
        Total number of training steps
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Initialize alive tracker (uses masks, not CI threshold)
    alive_tracker = AliveTracker(
        module_to_c={"fc1": model.fc1.C, "fc2": model.fc2.C},
        device=device,
        n_batches_until_dead=n_batches_until_dead,
    )

    logger.info(f"Training decomposed MLP for {epochs} epochs...")
    logger.info(f"  - Sampling: {sampling}")
    logger.info(f"  - Importance coeff: {importance_coeff}")
    logger.info(
        f"  - P-norm: {pnorm} -> {p_anneal_final_p} (anneal {p_anneal_start_frac}-{p_anneal_end_frac})"
    )

    total_steps = 0
    total_training_steps = epochs * len(train_loader)

    for epoch in range(epochs):
        total_ce_loss = 0.0
        total_im_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass with CI values and masks
            outputs, ci_values, masks = model(images, sampling=sampling, return_extras=True)

            # CE loss
            ce_loss = criterion(outputs, labels)

            # Importance minimality loss (using SPD's implementation)
            current_frac = total_steps / total_training_steps
            im_loss = importance_minimality_loss(
                ci_upper_leaky=ci_values,
                current_frac_of_training=current_frac,
                pnorm=pnorm,
                eps=1e-8,
                p_anneal_start_frac=p_anneal_start_frac,
                p_anneal_final_p=p_anneal_final_p,
                p_anneal_end_frac=p_anneal_end_frac,
            )

            # Total loss
            loss = ce_loss + importance_coeff * im_loss

            loss.backward()
            optimizer.step()

            total_ce_loss += ce_loss.item()
            total_im_loss += im_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Compute L0 from masks (actual active components)
            n_components_fc1 = masks["fc1"].shape[-1]
            n_components_fc2 = masks["fc2"].shape[-1]

            l0_fc1 = calc_l0(masks["fc1"])
            l0_fc2 = calc_l0(masks["fc2"])
            l0_total = l0_fc1 + l0_fc2

            # Update alive tracker (uses masks)
            alive_tracker.update(masks)
            alive_counts = alive_tracker.compute()
            n_alive_fc1 = alive_counts["fc1"]
            n_alive_fc2 = alive_counts["fc2"]
            alive_pct_fc1 = 100.0 * n_alive_fc1 / n_components_fc1
            alive_pct_fc2 = 100.0 * n_alive_fc2 / n_components_fc2

            mean_ci_fc1 = ci_values["fc1"].mean().item()
            mean_ci_fc2 = ci_values["fc2"].mean().item()
            avg_ci = (mean_ci_fc1 + mean_ci_fc2) / 2

            pbar.set_postfix(
                {
                    "ce": f"{ce_loss.item():.4f}",
                    "im": f"{im_loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.1f}%",
                    "l0": f"{l0_fc1:.0f}/{l0_fc2:.0f}",
                    "alive%": f"{alive_pct_fc1:.0f}/{alive_pct_fc2:.0f}",
                }
            )

            if log_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/ce_loss": ce_loss.item(),
                        "train/im_loss": im_loss.item(),
                        "train/total_loss": loss.item(),
                        "train/accuracy": 100.0 * correct / total,
                        "train/avg_ci": avg_ci,
                        "train/mean_ci_fc1": mean_ci_fc1,
                        "train/mean_ci_fc2": mean_ci_fc2,
                        "train/l0_fc1": l0_fc1,
                        "train/l0_fc2": l0_fc2,
                        "train/l0_total": l0_total,
                        "train/alive_pct_fc1": alive_pct_fc1,
                        "train/alive_pct_fc2": alive_pct_fc2,
                        "train/epoch": epoch,
                    },
                    step=total_steps,
                )

            total_steps += 1

        avg_ce = total_ce_loss / len(train_loader)
        avg_im = total_im_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        logger.info(f"Epoch {epoch + 1}: CE={avg_ce:.4f}, IM={avg_im:.4f}, Acc={accuracy:.2f}%")

        if log_wandb:
            wandb.log(
                {
                    "train/epoch_ce_loss": avg_ce,
                    "train/epoch_im_loss": avg_im,
                    "train/epoch_accuracy": accuracy,
                },
                step=total_steps - 1,
            )

    return total_steps


def evaluate(
    model: DecomposedMLP,
    test_loader: DataLoader,
    device: str,
    sampling: str,
) -> dict[str, float]:
    """Evaluate the model on test set.

    Returns:
        Dict with accuracy, L0, alive_pct, and CI metrics per layer.
    """
    model.eval()
    correct = 0
    total = 0
    l0_fc1_sum = 0.0
    l0_fc2_sum = 0.0
    ci_fc1_sum = 0.0
    ci_fc2_sum = 0.0
    n_batches = 0
    n_components_fc1 = model.fc1.C
    n_components_fc2 = model.fc2.C

    # Track which components have been active at least once (mask > 0.5)
    # Using 0.5 threshold works for both binary masks and soft masks
    ever_active_fc1 = torch.zeros(n_components_fc1, dtype=torch.bool, device=device)
    ever_active_fc2 = torch.zeros(n_components_fc2, dtype=torch.bool, device=device)
    active_threshold = 0.5

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, ci_values, masks = model(images, sampling=sampling, return_extras=True)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Track L0 from masks (actual active components)
            l0_fc1_sum += calc_l0(masks["fc1"])
            l0_fc2_sum += calc_l0(masks["fc2"])
            ci_fc1_sum += ci_values["fc1"].mean().item()
            ci_fc2_sum += ci_values["fc2"].mean().item()
            n_batches += 1

            # Update ever_active masks (component active if mask > threshold on any sample)
            ever_active_fc1 |= (masks["fc1"] > active_threshold).any(dim=0)
            ever_active_fc2 |= (masks["fc2"] > active_threshold).any(dim=0)

    accuracy = 100.0 * correct / total
    l0_fc1 = l0_fc1_sum / n_batches
    l0_fc2 = l0_fc2_sum / n_batches
    mean_ci_fc1 = ci_fc1_sum / n_batches
    mean_ci_fc2 = ci_fc2_sum / n_batches

    # Alive = active at least once across all test samples
    n_alive_fc1 = ever_active_fc1.sum().item()
    n_alive_fc2 = ever_active_fc2.sum().item()

    return {
        "accuracy": accuracy,
        "l0_fc1": l0_fc1,
        "l0_fc2": l0_fc2,
        "l0_total": l0_fc1 + l0_fc2,
        "alive_pct_fc1": 100.0 * n_alive_fc1 / n_components_fc1,
        "alive_pct_fc2": 100.0 * n_alive_fc2 / n_components_fc2,
        "mean_ci_fc1": mean_ci_fc1,
        "mean_ci_fc2": mean_ci_fc2,
        "avg_ci": (mean_ci_fc1 + mean_ci_fc2) / 2,
    }


def plot_component_directions(
    model: DecomposedMLP,
    out_dir: Path,
    n_components_to_show: int,
    log_wandb: bool,
) -> None:
    """Plot component directions as images for fc1."""
    V_fc1 = model.fc1.V.detach().cpu()  # (784, C)
    C = V_fc1.shape[1]
    n_show = min(n_components_to_show, C)

    # Compute global scale across all components to show
    directions = [V_fc1[:, i].reshape(28, 28).numpy() for i in range(n_show)]
    global_max = max(abs(d).max() for d in directions)

    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_show):
        im = axes[i].imshow(directions[i], cmap="RdBu_r", vmin=-global_max, vmax=global_max)
        axes[i].set_title(f"Component {i}", fontsize=10)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("FC1 Component Directions (Input Space)", fontsize=14, y=0.995)
    plt.tight_layout()

    plot_path = out_dir / "component_directions_fc1.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved component directions to {plot_path}")

    if log_wandb:
        wandb.log({"visualizations/component_directions_fc1": wandb.Image(str(plot_path))})


def plot_ci_distribution(
    model: DecomposedMLP,
    test_loader: DataLoader,
    device: str,
    out_dir: Path,
    log_wandb: bool,
) -> None:
    """Plot distribution of CI values across test set."""
    model.eval()

    all_ci_fc1 = []
    all_ci_fc2 = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _, ci_values, _ = model(images, sampling="none", return_extras=True)
            all_ci_fc1.append(ci_values["fc1"].cpu())
            all_ci_fc2.append(ci_values["fc2"].cpu())

    ci_fc1 = torch.cat(all_ci_fc1, dim=0)  # (N, C)
    ci_fc2 = torch.cat(all_ci_fc2, dim=0)  # (N, C)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # FC1 CI distribution
    mean_ci_fc1 = ci_fc1.mean(dim=0).numpy()
    axes[0].bar(range(len(mean_ci_fc1)), np.sort(mean_ci_fc1)[::-1], alpha=0.7)
    axes[0].set_xlabel("Component (sorted by CI)")
    axes[0].set_ylabel("Mean CI")
    axes[0].set_title(
        f"FC1 CI Distribution (active>0.5: {(mean_ci_fc1 > 0.5).sum()}/{len(mean_ci_fc1)})"
    )
    axes[0].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    # FC2 CI distribution
    mean_ci_fc2 = ci_fc2.mean(dim=0).numpy()
    axes[1].bar(range(len(mean_ci_fc2)), np.sort(mean_ci_fc2)[::-1], alpha=0.7)
    axes[1].set_xlabel("Component (sorted by CI)")
    axes[1].set_ylabel("Mean CI")
    axes[1].set_title(
        f"FC2 CI Distribution (active>0.5: {(mean_ci_fc2 > 0.5).sum()}/{len(mean_ci_fc2)})"
    )
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plot_path = out_dir / "ci_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved CI distribution to {plot_path}")

    if log_wandb:
        wandb.log({"visualizations/ci_distribution": wandb.Image(str(plot_path))})


def main(
    hidden_size: int = 128,
    n_components: int = 500,
    batch_size: int = 1024,
    epochs: int = 3000,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    importance_coeff: float = 3e-9,
    pnorm: float = 1.0,
    p_anneal_start_frac: float = 0.0,
    p_anneal_final_p: float = 1.0,
    p_anneal_end_frac: float = 0.5,
    sampling: str = "bernoulli",
    ci_fn_type: str = "linear",
    ci_fn_hidden_dims: list[int] | None = None,
    use_normal_sigmoid: bool = False,
    n_batches_until_dead: int = 100,
    seed: int = 42,
    output_dir: str | None = None,
    wandb_project: str | None = "mnist_predecomposed_mlp",
) -> None:
    """Main experiment function.

    Args:
        hidden_size: Hidden layer size
        n_components: Number of components per layer
        batch_size: Training batch size
        epochs: Training epochs
        lr: Learning rate
        weight_decay: Weight decay
        importance_coeff: Coefficient for importance minimality loss
        pnorm: Starting p-norm for sparsity (smaller = sparser)
        p_anneal_start_frac: Fraction of training to start annealing p
        p_anneal_final_p: Final p value after annealing
        p_anneal_end_frac: Fraction of training to finish annealing p
        sampling: "bernoulli" for stochastic, "none" for deterministic
        ci_fn_type: "linear" or "mlp"
        ci_fn_hidden_dims: Hidden dimensions for CI MLP (None uses defaults)
        use_normal_sigmoid: Use normal sigmoid instead of leaky hard sigmoids
        n_batches_until_dead: Batches without activity before component is considered dead
        seed: Random seed
        output_dir: Output directory
        wandb_project: W&B project name (None to disable)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        output_dir = "./output/mnist_decomposed"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if wandb_project:
        import os

        from dotenv import load_dotenv

        load_dotenv(override=True)
        run_name = f"decomposed_h{hidden_size}_c{n_components}_im{importance_coeff}_p{pnorm}"

        wandb.init(
            project=wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            tags=["mnist", "decomposed", "from_scratch"],
            config={
                "hidden_size": hidden_size,
                "n_components": n_components,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "importance_coeff": importance_coeff,
                "pnorm": pnorm,
                "p_anneal_start_frac": p_anneal_start_frac,
                "p_anneal_final_p": p_anneal_final_p,
                "p_anneal_end_frac": p_anneal_end_frac,
                "sampling": sampling,
                "ci_fn_type": ci_fn_type,
                "ci_fn_hidden_dims": ci_fn_hidden_dims,
                "use_normal_sigmoid": use_normal_sigmoid,
                "n_batches_until_dead": n_batches_until_dead,
                "seed": seed,
            },
        )

    # Load MNIST
    logger.info("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    logger.info(f"Creating decomposed MLP with {n_components} components per layer...")
    ci_fn_hidden_dims_resolved = ci_fn_hidden_dims if ci_fn_hidden_dims is not None else [256]
    model = DecomposedMLP(
        input_size=784,
        hidden_size=hidden_size,
        num_classes=10,
        n_components=n_components,
        ci_fn_type=ci_fn_type,
        ci_fn_hidden_dims=ci_fn_hidden_dims_resolved,
        use_normal_sigmoid=use_normal_sigmoid,
    )
    model = model.to(device)

    # Train
    total_steps = train_decomposed_mlp(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        importance_coeff=importance_coeff,
        pnorm=pnorm,
        p_anneal_start_frac=p_anneal_start_frac,
        p_anneal_final_p=p_anneal_final_p,
        p_anneal_end_frac=p_anneal_end_frac,
        sampling=sampling,
        n_batches_until_dead=n_batches_until_dead,
        log_wandb=wandb_project is not None,
    )

    # Evaluate
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, sampling="none")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(
        f"L0 (active components): fc1={test_metrics['l0_fc1']:.1f}, "
        f"fc2={test_metrics['l0_fc2']:.1f}, total={test_metrics['l0_total']:.1f}"
    )
    logger.info(
        f"Alive components: fc1={test_metrics['alive_pct_fc1']:.1f}%, "
        f"fc2={test_metrics['alive_pct_fc2']:.1f}%"
    )
    logger.info(f"Average CI: {test_metrics['avg_ci']:.4f}")

    if wandb_project:
        wandb.log(
            {
                "test/accuracy": test_metrics["accuracy"],
                "test/l0_fc1": test_metrics["l0_fc1"],
                "test/l0_fc2": test_metrics["l0_fc2"],
                "test/l0_total": test_metrics["l0_total"],
                "test/alive_pct_fc1": test_metrics["alive_pct_fc1"],
                "test/alive_pct_fc2": test_metrics["alive_pct_fc2"],
                "test/mean_ci_fc1": test_metrics["mean_ci_fc1"],
                "test/mean_ci_fc2": test_metrics["mean_ci_fc2"],
                "test/avg_ci": test_metrics["avg_ci"],
            },
            step=total_steps,
        )

    # Save model
    model_path = out_path / "decomposed_mlp.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # Visualizations
    logger.info("Creating visualizations...")
    plot_component_directions(
        model, out_path, n_components_to_show=20, log_wandb=wandb_project is not None
    )
    plot_ci_distribution(model, test_loader, device, out_path, log_wandb=wandb_project is not None)

    logger.info("Experiment complete!")
    logger.info(f"All outputs saved to: {out_path}")

    if wandb_project:
        wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(main)

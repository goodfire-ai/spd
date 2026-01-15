"""Train a 2-layer MLP on MNIST and then decompose it using SPD."""

from pathlib import Path

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

from spd.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    Config,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PGDReconSubsetLossConfig,
    ScheduleConfig,
    StochasticAccuracyLayerwiseConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconSubsetLossConfig,
    TMSTaskConfig,
    UVPlotsConfig,
)
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import ExecutionStamp, save_file


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTTargetDataset:
    """Dataset that generates (input, target_output) pairs for decomposition.

    This dataset wraps MNIST and generates batches where the target is the
    model's output on the input image.
    """

    def __init__(self, mnist_dataset, model: nn.Module, device: str, size: int = 100000):
        self.mnist_dataset = mnist_dataset
        self.model = model
        self.device = device
        self.size = size
        self.model.eval()

    def __len__(self) -> int:
        return self.size

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (input, target_output) pairs."""
        # Sample random indices from MNIST
        indices = torch.randint(0, len(self.mnist_dataset), (batch_size,))

        # Get images and flatten them
        images = []
        for idx in indices:
            image, _ = self.mnist_dataset[int(idx)]
            images.append(image)
        images = torch.stack(images).to(self.device)
        images = images.view(batch_size, -1)  # Flatten to (batch, 784)

        # Get target outputs from the model
        with torch.no_grad():
            target_outputs = self.model(images)

        return images, target_outputs


def train_mlp(
    model: TwoLayerMLP,
    train_loader: DataLoader,
    device: str,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    log_wandb: bool = False,
) -> int:
    """Train the MLP on MNIST.

    Returns:
        Total number of training steps completed
    """
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

            # Log to wandb periodically with mlp_train prefix to avoid conflicts
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

        # Log epoch summary to wandb
        if log_wandb:
            wandb.log(
                {
                    "mlp_train/epoch_loss": avg_loss,
                    "mlp_train/epoch_accuracy": accuracy,
                },
                step=total_steps - 1,  # Last step of the epoch
            )

    model.eval()
    return total_steps


def main(
    hidden_size: int = 128,
    train_epochs: int = 50,
    train_lr: float = 0.001,
    train_weight_decay: float = 5e-4,
    spd_steps: int = 200000,
    spd_lr: float = 0.001,
    n_components: int = 500,
    seed: int = 42,
    output_dir: str | None = None,
    wandb_project: str | None = None,
) -> None:
    """Main experiment function.

    Args:
        hidden_size: Hidden layer size for the MLP
        train_epochs: Number of epochs to train the MLP
        train_lr: Learning rate for MLP training
        train_weight_decay: Weight decay for MLP training
        spd_steps: Number of steps for SPD decomposition
        spd_lr: Learning rate for SPD decomposition
        n_components: Number of components per layer for decomposition
        seed: Random seed
        output_dir: Output directory (defaults to ./output/mnist_experiment)
        wandb_project: WandB project name (None to disable wandb logging)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        output_dir = "./output/mnist_experiment"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb for training phase
    run_id = None
    if wandb_project:
        import os

        from dotenv import load_dotenv

        load_dotenv(override=True)
        execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
        run_id = execution_stamp.run_id
        run_name = f"mnist_mlp_h{hidden_size}_e{train_epochs}_c{n_components}_s{seed}"

        wandb.init(
            id=run_id,
            project=wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            tags=["mnist", "mlp", "spd"],
        )
        assert wandb.run is not None

        # Log hyperparameters
        wandb.config.update(
            {
                "hidden_size": hidden_size,
                "train_epochs": train_epochs,
                "train_lr": train_lr,
                "train_weight_decay": train_weight_decay,
                "spd_steps": spd_steps,
                "spd_lr": spd_lr,
                "n_components": n_components,
                "seed": seed,
            }
        )

    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create and train the model
    logger.info("Creating 2-layer MLP...")
    model = TwoLayerMLP(input_size=784, hidden_size=hidden_size, num_classes=10)
    model = model.to(device)

    # Train the model
    total_training_steps = train_mlp(
        model,
        train_loader,
        device,
        epochs=train_epochs,
        lr=train_lr,
        weight_decay=train_weight_decay,
        log_wandb=wandb_project is not None,
    )

    # Log training completion step
    if wandb_project:
        wandb.log({"mlp_train/completed": 1}, step=total_training_steps - 1)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100.0 * correct / total
    logger.info(f"Test accuracy: {test_accuracy:.2f}%")

    # Log test accuracy to wandb
    if wandb_project:
        wandb.log({"mlp_test/accuracy": test_accuracy}, step=total_training_steps)

    # Save the trained model
    model_path = out_path / "trained_mlp.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved trained model to {model_path}")
    if wandb_project:
        wandb.save(str(model_path), base_path=str(out_path), policy="now")

    # Now set up for SPD decomposition
    logger.info("Setting up SPD decomposition...")
    model.eval()
    model.requires_grad_(False)

    # Create datasets for SPD
    train_target_dataset = MNISTTargetDataset(train_dataset, model, device, size=50000)
    eval_target_dataset = MNISTTargetDataset(test_dataset, model, device, size=10000)

    # Create data loaders compatible with SPD
    train_spd_loader = DatasetGeneratedDataLoader(
        train_target_dataset, batch_size=128, shuffle=True
    )
    eval_spd_loader = DatasetGeneratedDataLoader(eval_target_dataset, batch_size=128, shuffle=False)

    # Use the same output directory for everything
    out_dir = out_path
    logger.info(f"All outputs will be saved to: {out_dir}")

    # Create SPD config
    spd_config = Config(
        wandb_project=wandb_project,  # Use same wandb project
        seed=seed,
        n_mask_samples=1,
        ci_fn_type="linear",
        ci_fn_hidden_dims=[256],
        sigmoid_type="leaky_hard",
        module_info=[
            {"module_pattern": "fc1", "C": n_components},
            {"module_pattern": "fc2", "C": n_components},
        ],
        use_delta_component=True,
        loss_metric_configs=[
            ImportanceMinimalityLossConfig(
                coeff=5e-3,
                pnorm=2.0,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=0.5,
                p_anneal_end_frac=0.5,
            ),
            StochasticReconSubsetLossConfig(coeff=1.0),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=1.0),
            PGDReconSubsetLossConfig(
                coeff=1.0,
                init="random",
                step_size=1.0,
                n_steps=1,
                mask_scope="unique_per_datapoint",
            ),
        ],
        output_loss_type="kl",
        lr_schedule=ScheduleConfig(start_val=spd_lr, fn_type="cosine", final_val_frac=0.0),
        steps=spd_steps,
        batch_size=1024,
        gradient_accumulation_steps=1,
        faithfulness_warmup_steps=200,
        faithfulness_warmup_lr=0.01,
        faithfulness_warmup_weight_decay=0.1,
        train_log_freq=100,
        eval_freq=500,
        eval_batch_size=128,
        slow_eval_freq=2000,
        n_eval_steps=50,
        slow_eval_on_first_step=True,
        save_freq=None,
        eval_metric_configs=[
            PGDReconSubsetLossConfig(
                init="random",
                step_size=1.0,
                n_steps=20,
                mask_scope="unique_per_datapoint",
            ),
            CIMeanPerComponentConfig(),
            CIHistogramsConfig(n_batches_accum=5),
            ComponentActivationDensityConfig(),
            CI_L0Config(groups=None),
            UVPlotsConfig(identity_patterns=None, dense_patterns=None),
            StochasticAccuracyLayerwiseConfig(),
        ],
        ci_alive_threshold=0.1,
        n_examples_until_dead=1000000,
        pretrained_model_class="__main__.TwoLayerMLP",
        pretrained_model_path=None,  # We'll pass the model directly
        task_config=TMSTaskConfig(
            task_name="tms", feature_probability=0.1, data_generation_type="at_least_zero_active"
        ),
    )

    # Save config (SPDRunInfo expects final_config.yaml)
    config_path = out_dir / "final_config.yaml"
    save_file(spd_config.model_dump(mode="json"), config_path)
    logger.info(f"Saved SPD config to {config_path}")
    if wandb_project:
        wandb.save(str(config_path), base_path=str(out_dir), policy="now")

    # Run SPD decomposition with step offset
    logger.info("Starting SPD decomposition...")

    # Create a wrapper to offset SPD steps to continue from training
    original_wandb_log = None
    if wandb_project:
        original_wandb_log = wandb.log
        step_offset = total_training_steps + 1  # Start SPD steps after training

        def wandb_log_with_offset(*args, **kwargs):
            """Wrapper to offset SPD step numbers."""
            if "step" in kwargs:
                kwargs["step"] = kwargs["step"] + step_offset
            return original_wandb_log(*args, **kwargs)

        # Monkey-patch wandb.log for the duration of optimize()
        wandb.log = wandb_log_with_offset

    try:
        optimize(
            target_model=model,
            config=spd_config,
            device=device,
            train_loader=train_spd_loader,
            eval_loader=eval_spd_loader,
            n_eval_steps=spd_config.n_eval_steps,
            out_dir=out_dir,
        )
    finally:
        # Restore original wandb.log
        if wandb_project and original_wandb_log is not None:
            wandb.log = original_wandb_log

    logger.info("Experiment complete!")
    logger.info(f"All outputs saved to: {out_dir}")
    logger.info(f"  - Trained model: {model_path}")
    logger.info("  - SPD checkpoints: model_*.pth")
    logger.info("  - Config: final_config.yaml")
    logger.info("  - Visualizations: component_*.png")

    # Load the decomposed model and create visualizations
    logger.info("Creating component visualizations...")
    plot_component_directions(
        out_dir=out_dir,
        model=model,
        test_dataset=test_dataset,
        device=device,
        spd_config=spd_config,
        n_components_to_show=min(n_components, 20),  # Show up to 20 components
        log_wandb=wandb_project is not None,
    )

    logger.info("Visualizations complete!")

    if wandb_project:
        wandb.finish()


def plot_component_directions(
    out_dir: Path,
    model: TwoLayerMLP,
    test_dataset: datasets.MNIST,
    device: str,
    spd_config: Config,
    n_components_to_show: int = 20,
    log_wandb: bool = False,
) -> None:
    """Plot component directions as images.

    For fc1, the V matrix columns represent directions in input space (784-dim -> 28x28 images).
    We visualize these as images to see what patterns activate each component.
    """
    # Load the final component model checkpoint
    final_checkpoint = out_dir / f"model_{spd_config.steps}.pth"
    if not final_checkpoint.exists():
        # Try to find the latest checkpoint
        checkpoints = list(out_dir.glob("model_*.pth"))
        if not checkpoints:
            logger.warning("No component model checkpoint found. Skipping visualizations.")
            return
        final_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        logger.info(f"Using checkpoint: {final_checkpoint}")

    # Create ComponentModel with the same structure as during training
    model.eval()
    model.requires_grad_(False)

    module_path_info = expand_module_patterns(model, spd_config.all_module_info)
    component_model = ComponentModel(
        target_model=model,
        module_path_info=module_path_info,
        ci_fn_type=spd_config.ci_fn_type,
        ci_fn_hidden_dims=spd_config.ci_fn_hidden_dims,
        pretrained_model_output_attr=spd_config.pretrained_model_output_attr,
        sigmoid_type=spd_config.sigmoid_type,
    )

    # Load the checkpoint weights
    checkpoint_weights = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    component_model.load_state_dict(checkpoint_weights)
    component_model.to(device)
    component_model.eval()

    # Get fc1 components (input layer)
    fc1_components = component_model.components.get("fc1")
    if fc1_components is None:
        logger.warning("fc1 components not found. Skipping visualizations.")
        return

    # Extract V matrix for fc1: shape (784, C) where C is number of components
    V_fc1 = fc1_components.V.detach().cpu()  # Shape: (784, C)
    C = V_fc1.shape[1]
    n_components_to_show = min(n_components_to_show, C)

    logger.info(f"Visualizing {n_components_to_show} components from fc1 layer...")

    # Reshape each column to 28x28 and visualize
    n_cols = 5
    n_rows = (n_components_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_components_to_show):
        component_direction = V_fc1[:, i].reshape(28, 28).numpy()

        # Normalize for visualization
        vmin, vmax = component_direction.min(), component_direction.max()
        if vmax - vmin > 1e-6:
            component_direction = (component_direction - vmin) / (vmax - vmin)

        im = axes[i].imshow(component_direction, cmap="RdBu_r", vmin=0, vmax=1)
        axes[i].set_title(f"Component {i}", fontsize=10)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    # Hide unused subplots
    for i in range(n_components_to_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("FC1 Component Directions (Input Space)", fontsize=14, y=0.995)
    plt.tight_layout()

    fc1_plot_path = out_dir / "component_directions_fc1.png"
    plt.savefig(fc1_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved fc1 component directions to {fc1_plot_path}")

    # Log to wandb
    if log_wandb:
        wandb.log({"visualizations/component_directions_fc1": wandb.Image(str(fc1_plot_path))})

    # Also visualize component activations on sample images
    plot_component_activations_on_images(
        component_model=component_model,
        test_dataset=test_dataset,
        device=device,
        out_dir=out_dir,
        n_samples=10,
        n_components_to_show=n_components_to_show,
        log_wandb=log_wandb,
    )


def plot_component_activations_on_images(
    component_model: ComponentModel,
    test_dataset: datasets.MNIST,
    device: str,
    out_dir: Path,
    n_samples: int = 10,
    n_components_to_show: int = 20,
    log_wandb: bool = False,
) -> None:
    """Plot component activations on sample MNIST images.

    Shows which components activate most strongly for different digit images.
    """
    fc1_components = component_model.components.get("fc1")
    if fc1_components is None:
        return

    # Sample some images from test set
    sample_indices = np.random.choice(len(test_dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 2))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    component_model.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            image, label = test_dataset[int(sample_idx)]
            image_flat = image.view(1, -1).to(device)  # Shape: (1, 784)

            # Get component activations
            component_acts = fc1_components.get_component_acts(image_flat)  # Shape: (1, C)
            component_acts = component_acts[0].cpu().numpy()  # Shape: (C,)

            # Plot original image
            axes[idx, 0].imshow(image.squeeze().numpy(), cmap="gray")
            axes[idx, 0].set_title(f"Digit: {label}", fontsize=10)
            axes[idx, 0].axis("off")

            # Plot component activations
            n_show = min(n_components_to_show, len(component_acts))
            top_components = np.argsort(np.abs(component_acts))[-n_show:][::-1]
            top_activations = component_acts[top_components]

            axes[idx, 1].barh(range(len(top_components)), top_activations, color="steelblue")
            axes[idx, 1].set_yticks(range(len(top_components)))
            axes[idx, 1].set_yticklabels([f"C{c}" for c in top_components], fontsize=8)
            axes[idx, 1].set_xlabel("Activation", fontsize=9)
            axes[idx, 1].set_title(f"Top {n_show} Component Activations", fontsize=10)
            axes[idx, 1].grid(axis="x", alpha=0.3)

    plt.suptitle("Component Activations on Sample MNIST Images", fontsize=14, y=0.995)
    plt.tight_layout()

    activations_plot_path = out_dir / "component_activations_samples.png"
    plt.savefig(activations_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved component activations plot to {activations_plot_path}")

    # Log to wandb
    if log_wandb:
        wandb.log(
            {
                "visualizations/component_activations_samples": wandb.Image(
                    str(activations_plot_path)
                )
            }
        )

    # Create a heatmap showing which components activate for which digits
    plot_digit_component_heatmap(
        component_model=component_model,
        test_dataset=test_dataset,
        device=device,
        out_dir=out_dir,
        n_samples_per_digit=50,
        n_components_to_show=n_components_to_show,
        log_wandb=log_wandb,
    )


def plot_digit_component_heatmap(
    component_model: ComponentModel,
    test_dataset: datasets.MNIST,
    device: str,
    out_dir: Path,
    n_samples_per_digit: int = 50,
    n_components_to_show: int = 20,
    log_wandb: bool = False,
) -> None:
    """Plot a heatmap showing average component activations per digit class."""
    fc1_components = component_model.components.get("fc1")
    if fc1_components is None:
        return

    C = fc1_components.V.shape[1]
    n_components_to_show = min(n_components_to_show, C)

    # Collect activations per digit
    digit_activations = {digit: [] for digit in range(10)}

    component_model.eval()
    with torch.no_grad():
        for digit in range(10):
            # Find samples of this digit
            digit_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] == digit]
            if len(digit_indices) == 0:
                continue

            # Sample some examples
            sample_indices = np.random.choice(
                digit_indices, min(n_samples_per_digit, len(digit_indices)), replace=False
            )

            for idx in sample_indices:
                image, _ = test_dataset[int(idx)]
                image_flat = image.view(1, -1).to(device)

                # Get component activations
                component_acts = fc1_components.get_component_acts(image_flat)
                digit_activations[digit].append(component_acts[0].cpu().numpy())

    # Compute average activations per digit
    avg_activations = np.zeros((10, C))
    for digit in range(10):
        if len(digit_activations[digit]) > 0:
            avg_activations[digit] = np.mean(digit_activations[digit], axis=0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(12, n_components_to_show * 0.5), 6))

    # Show only top components
    component_importance = np.abs(avg_activations).mean(axis=0)
    top_components = np.argsort(component_importance)[-n_components_to_show:][::-1]

    im = ax.imshow(
        avg_activations[:, top_components],
        cmap="RdBu_r",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Component Index", fontsize=12)
    ax.set_ylabel("Digit Class", fontsize=12)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xticks(range(len(top_components)))
    ax.set_xticklabels([f"C{c}" for c in top_components], rotation=45, ha="right")
    ax.set_title("Average Component Activations by Digit Class", fontsize=14)

    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()

    heatmap_path = out_dir / "digit_component_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved digit-component heatmap to {heatmap_path}")

    # Log to wandb
    if log_wandb:
        wandb.log({"visualizations/digit_component_heatmap": wandb.Image(str(heatmap_path))})


if __name__ == "__main__":
    import fire

    fire.Fire(main)

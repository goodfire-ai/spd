"""Decompose the shapes/colors/sizes CNN using SPD.

Ground truth: 8 atomic mechanisms (3 shapes + 3 colors + 2 sizes)
If SPD finds components that separate by attribute type, that's evidence
it discovers atomic features rather than composed ones (like "red-square").
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from shapes_colours_dataset import (
    MultiAttributeCNNSingleHead,
    MultiAttributeShapesDataset,
)
from torch.utils.data import DataLoader
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


class ShapesTargetDataset:
    """Dataset that generates (input, target_output) pairs for decomposition.

    For the shapes network, inputs are flattened conv features and targets
    are the concatenated logits [shape_logits, color_logits, size_logits].
    """

    def __init__(
        self,
        shapes_dataset: MultiAttributeShapesDataset,
        model: MultiAttributeCNNSingleHead,
        device: str,
        size: int = 100000,
    ):
        self.shapes_dataset = shapes_dataset
        self.model = model
        self.device = device
        self.size = size
        self.model.eval()

    def __len__(self) -> int:
        return self.size

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (input, target_output) pairs.

        Returns:
            inputs: Flattened conv features (batch, flat_size)
            targets: Concatenated logits (batch, n_shapes + n_colors + n_sizes)
        """
        # Sample random indices from dataset
        indices = torch.randint(0, len(self.shapes_dataset), (batch_size,))

        # Get images
        images = []
        for idx in indices:
            image, _ = self.shapes_dataset[int(idx)]
            images.append(image)
        images = torch.stack(images).to(self.device)

        # Get conv features (input to MLP)
        with torch.no_grad():
            x = self.model.pool(F.relu(self.model.conv1(images)))
            x = self.model.pool(F.relu(self.model.conv2(x)))
            x = self.model.pool(F.relu(self.model.conv3(x)))
            conv_features = x.view(x.size(0), -1)  # Flatten

            # Get full model output as target
            outputs = self.model(images)
            # Concatenate all logits
            target_logits = torch.cat([outputs["shape"], outputs["color"], outputs["size"]], dim=1)

        return conv_features, target_logits


class ShapesMLP(nn.Module):
    """Just the MLP part of the shapes CNN for decomposition."""

    def __init__(self, flat_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(flat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_shapes_model(
    model: MultiAttributeCNNSingleHead,
    train_loader: DataLoader,
    device: str,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    log_wandb: bool = False,
) -> int:
    """Train the shapes CNN.

    Returns:
        Total number of training steps completed
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(f"Training shapes CNN for {epochs} epochs...")
    total_steps = 0

    for epoch in range(epochs):
        total_loss = 0.0
        correct = {"shape": 0, "color": 0, "size": 0}
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)

            loss = (
                F.cross_entropy(outputs["shape"], labels["shape"])
                + F.cross_entropy(outputs["color"], labels["color"])
                + F.cross_entropy(outputs["size"], labels["size"])
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for attr in ["shape", "color", "size"]:
                preds = outputs[attr].argmax(dim=1)
                correct[attr] += (preds == labels[attr]).sum().item()
            total += labels["shape"].size(0)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "shape": f"{100.0 * correct['shape'] / total:.1f}%",
                    "color": f"{100.0 * correct['color'] / total:.1f}%",
                    "size": f"{100.0 * correct['size'] / total:.1f}%",
                }
            )

            if log_wandb and batch_idx % 50 == 0:
                wandb.log(
                    {
                        "cnn_train/loss": loss.item(),
                        "cnn_train/acc_shape": 100.0 * correct["shape"] / total,
                        "cnn_train/acc_color": 100.0 * correct["color"] / total,
                        "cnn_train/acc_size": 100.0 * correct["size"] / total,
                    },
                    step=total_steps,
                )

            total_steps += 1

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f} | "
            f"Shape: {100.0 * correct['shape'] / total:.1f}% | "
            f"Color: {100.0 * correct['color'] / total:.1f}% | "
            f"Size: {100.0 * correct['size'] / total:.1f}%"
        )

    model.eval()
    return total_steps


def main(
    hidden_dim: int = 64,
    train_epochs: int = 20,
    train_lr: float = 0.001,
    train_weight_decay: float = 5e-4,
    spd_steps: int = 500000,
    spd_lr: float = 0.001,
    n_components: int = 500,
    seed: int = 42,
    n_train_samples: int = 10000,
    n_test_samples: int = 2000,
    output_dir: str | None = None,
    wandb_project: str | None = "shapes_decomposition",
) -> None:
    """Main experiment function.

    Args:
        hidden_dim: Hidden layer size for the MLP
        train_epochs: Number of epochs to train the CNN
        train_lr: Learning rate for CNN training
        train_weight_decay: Weight decay for CNN training
        spd_steps: Number of steps for SPD decomposition
        spd_lr: Learning rate for SPD decomposition
        n_components: Number of components per layer for decomposition
        seed: Random seed
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        output_dir: Output directory (defaults to ./output/shapes_decomposition)
        wandb_project: WandB project name (None to disable wandb logging)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        output_dir = "./output/shapes_decomposition"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    run_id = None
    if wandb_project:
        import os

        from dotenv import load_dotenv

        load_dotenv(override=True)
        execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
        run_id = execution_stamp.run_id
        run_name = f"shapes_cnn_h{hidden_dim}_e{train_epochs}_c{n_components}_s{seed}"

        wandb.init(
            id=run_id,
            project=wandb_project,
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            tags=["shapes", "cnn", "spd", "multi-attribute"],
        )
        assert wandb.run is not None

        wandb.config.update(
            {
                "hidden_dim": hidden_dim,
                "train_epochs": train_epochs,
                "train_lr": train_lr,
                "train_weight_decay": train_weight_decay,
                "spd_steps": spd_steps,
                "spd_lr": spd_lr,
                "n_components": n_components,
                "seed": seed,
                "n_train_samples": n_train_samples,
                "n_test_samples": n_test_samples,
            }
        )

    # Create datasets
    logger.info("Creating shapes datasets...")
    train_dataset = MultiAttributeShapesDataset(n_samples=n_train_samples, seed=seed)
    test_dataset = MultiAttributeShapesDataset(n_samples=n_test_samples, seed=seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create and train the full CNN model
    logger.info("Creating shapes CNN...")
    full_model = MultiAttributeCNNSingleHead(img_size=32, hidden_dim=hidden_dim)
    full_model = full_model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in full_model.parameters()):,}")

    total_training_steps = train_shapes_model(
        full_model,
        train_loader,
        device,
        epochs=train_epochs,
        lr=train_lr,
        weight_decay=train_weight_decay,
        log_wandb=wandb_project is not None,
    )

    # Evaluate on test set
    full_model.eval()
    correct = {"shape": 0, "color": 0, "size": 0}
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            outputs = full_model(images)
            for attr in ["shape", "color", "size"]:
                preds = outputs[attr].argmax(dim=1)
                correct[attr] += (preds == labels[attr]).sum().item()
            total += labels["shape"].size(0)

    test_acc_shape = 100.0 * correct["shape"] / total
    test_acc_color = 100.0 * correct["color"] / total
    test_acc_size = 100.0 * correct["size"] / total
    logger.info(
        f"Test accuracy: Shape={test_acc_shape:.1f}% | "
        f"Color={test_acc_color:.1f}% | "
        f"Size={test_acc_size:.1f}%"
    )

    # Log test accuracy to wandb
    if wandb_project:
        wandb.log(
            {
                "cnn_test/acc_shape": test_acc_shape,
                "cnn_test/acc_color": test_acc_color,
                "cnn_test/acc_size": test_acc_size,
            },
            step=total_training_steps,
        )

    # Save the trained full model
    full_model_path = out_path / "trained_shapes_cnn.pth"
    torch.save(full_model.state_dict(), full_model_path)
    logger.info(f"Saved trained CNN to {full_model_path}")
    if wandb_project:
        wandb.save(str(full_model_path), base_path=str(out_path), policy="now")

    # Now create the MLP-only model for decomposition
    logger.info("Creating MLP model for decomposition...")
    output_dim = full_model.n_shapes + full_model.n_colors + full_model.n_sizes  # 8
    mlp_model = ShapesMLP(
        flat_size=full_model.flat_size,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    # Copy weights from full model
    mlp_model.fc1.weight.data = full_model.fc1.weight.data.clone()
    mlp_model.fc1.bias.data = full_model.fc1.bias.data.clone()
    mlp_model.fc2.weight.data = full_model.fc2.weight.data.clone()
    mlp_model.fc2.bias.data = full_model.fc2.bias.data.clone()
    mlp_model = mlp_model.to(device)
    mlp_model.eval()
    mlp_model.requires_grad_(False)

    # Create datasets for SPD (using conv features as inputs)
    train_target_dataset = ShapesTargetDataset(train_dataset, full_model, device, size=50000)
    eval_target_dataset = ShapesTargetDataset(test_dataset, full_model, device, size=10000)

    train_spd_loader = DatasetGeneratedDataLoader(
        train_target_dataset, batch_size=128, shuffle=True
    )
    eval_spd_loader = DatasetGeneratedDataLoader(eval_target_dataset, batch_size=128, shuffle=False)

    # Create SPD config
    spd_config = Config(
        wandb_project=wandb_project,
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
        pretrained_model_class="__main__.ShapesMLP",
        pretrained_model_path=None,
        task_config=TMSTaskConfig(
            task_name="tms", feature_probability=0.1, data_generation_type="at_least_zero_active"
        ),
    )

    # Save config
    config_path = out_path / "final_config.yaml"
    save_file(spd_config.model_dump(mode="json"), config_path)
    logger.info(f"Saved SPD config to {config_path}")
    if wandb_project:
        wandb.save(str(config_path), base_path=str(out_path), policy="now")

    # Run SPD decomposition
    logger.info("Starting SPD decomposition...")

    original_wandb_log = None
    if wandb_project:
        original_wandb_log = wandb.log
        step_offset = total_training_steps + 1

        def wandb_log_with_offset(*args, **kwargs):
            if "step" in kwargs:
                kwargs["step"] = kwargs["step"] + step_offset
            return original_wandb_log(*args, **kwargs)

        wandb.log = wandb_log_with_offset

    try:
        optimize(
            target_model=mlp_model,
            config=spd_config,
            device=device,
            train_loader=train_spd_loader,
            eval_loader=eval_spd_loader,
            n_eval_steps=spd_config.n_eval_steps,
            out_dir=out_path,
        )
    finally:
        if wandb_project and original_wandb_log is not None:
            wandb.log = original_wandb_log

    logger.info("SPD decomposition complete!")
    logger.info(f"All outputs saved to: {out_path}")
    logger.info(f"  - Trained model: {full_model_path}")
    logger.info("  - SPD checkpoints: model_*.pth")
    logger.info("  - Config: final_config.yaml")

    # Create visualizations
    logger.info("Creating component visualizations...")
    plot_component_attribute_correlation(
        out_dir=out_path,
        mlp_model=mlp_model,
        full_model=full_model,
        test_dataset=test_dataset,
        device=device,
        spd_config=spd_config,
        log_wandb=wandb_project is not None,
    )

    logger.info("Visualizations complete!")

    if wandb_project:
        wandb.finish()

    logger.info("Experiment complete!")


def plot_component_attribute_correlation(
    out_dir: Path,
    mlp_model: ShapesMLP,
    full_model: MultiAttributeCNNSingleHead,
    test_dataset: MultiAttributeShapesDataset,
    device: str,
    spd_config: Config,
    log_wandb: bool = False,
) -> None:
    """Plot how components correlate with different attributes (shape/color/size).

    This is the key analysis: if SPD discovers atomic features, we should see
    components that activate specifically for shapes, colors, or sizes separately.
    """
    # Load the final component model
    final_checkpoint = out_dir / f"model_{spd_config.steps}.pth"
    if not final_checkpoint.exists():
        checkpoints = list(out_dir.glob("model_*.pth"))
        if not checkpoints:
            logger.warning("No checkpoint found. Skipping visualizations.")
            return
        final_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))

    mlp_model.eval()
    mlp_model.requires_grad_(False)

    module_path_info = expand_module_patterns(mlp_model, spd_config.all_module_info)
    component_model = ComponentModel(
        target_model=mlp_model,
        module_path_info=module_path_info,
        ci_fn_type=spd_config.ci_fn_type,
        ci_fn_hidden_dims=spd_config.ci_fn_hidden_dims,
        pretrained_model_output_attr=spd_config.pretrained_model_output_attr,
        sigmoid_type=spd_config.sigmoid_type,
    )

    checkpoint_weights = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    component_model.load_state_dict(checkpoint_weights)
    component_model.to(device)
    component_model.eval()

    fc1_components = component_model.components.get("fc1")
    if fc1_components is None:
        logger.warning("fc1 components not found.")
        return

    C = fc1_components.V.shape[1]

    # Collect component activations for each attribute value
    attr_activations = {
        "shape": {i: [] for i in range(3)},  # circle, square, triangle
        "color": {i: [] for i in range(3)},  # red, green, blue
        "size": {i: [] for i in range(2)},  # small, large
    }

    full_model.eval()
    component_model.eval()

    with torch.no_grad():
        for idx in range(min(len(test_dataset), 1000)):
            image, labels = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Get conv features
            x = full_model.pool(F.relu(full_model.conv1(image)))
            x = full_model.pool(F.relu(full_model.conv2(x)))
            x = full_model.pool(F.relu(full_model.conv3(x)))
            conv_features = x.view(1, -1)

            # Get component activations
            component_acts = fc1_components.get_component_acts(conv_features)
            component_acts = component_acts[0].cpu().numpy()

            # Store by attribute
            for attr in ["shape", "color", "size"]:
                attr_val = labels[attr].item()
                attr_activations[attr][attr_val].append(component_acts)

    # Compute average activations per attribute value
    avg_acts = {}
    for attr in ["shape", "color", "size"]:
        n_values = len(attr_activations[attr])
        avg_acts[attr] = np.zeros((n_values, C))
        for val in range(n_values):
            if attr_activations[attr][val]:
                avg_acts[attr][val] = np.mean(attr_activations[attr][val], axis=0)

    # Plot heatmaps for each attribute
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    attr_labels = {
        "shape": ["circle", "square", "triangle"],
        "color": ["red", "green", "blue"],
        "size": ["small", "large"],
    }

    for ax, attr in zip(axes, ["shape", "color", "size"], strict=True):
        im = ax.imshow(avg_acts[attr], cmap="RdBu_r", aspect="auto")
        ax.set_xlabel("Component Index")
        ax.set_ylabel(attr.capitalize())
        ax.set_yticks(range(len(attr_labels[attr])))
        ax.set_yticklabels(attr_labels[attr])
        ax.set_title(f"Avg Component Activation by {attr.capitalize()}")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Component-Attribute Correlations (fc1)", fontsize=14)
    plt.tight_layout()

    corr_path = out_dir / "component_attribute_correlation.png"
    plt.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation plot to {corr_path}")

    if log_wandb:
        wandb.log({"visualizations/component_attribute_correlation": wandb.Image(str(corr_path))})

    # Compute and plot selectivity: which components are selective for which attribute?
    plot_component_selectivity(
        avg_acts=avg_acts,
        out_dir=out_dir,
        attr_labels=attr_labels,
        log_wandb=log_wandb,
    )


def plot_component_selectivity(
    avg_acts: dict,
    out_dir: Path,
    attr_labels: dict,
    log_wandb: bool = False,
) -> None:
    """Identify which components are most selective for each attribute type.

    A component is selective for an attribute if its activations vary significantly
    across values of that attribute but not others.
    """
    C = avg_acts["shape"].shape[1]

    # Compute variance across attribute values for each component
    variance_per_attr = {}
    for attr in ["shape", "color", "size"]:
        variance_per_attr[attr] = np.var(avg_acts[attr], axis=0)

    # Normalize variances
    total_variance = sum(variance_per_attr.values())
    selectivity = {}
    for attr in ["shape", "color", "size"]:
        selectivity[attr] = variance_per_attr[attr] / (total_variance + 1e-8)

    # Stack into matrix for visualization
    selectivity_matrix = np.stack(
        [selectivity["shape"], selectivity["color"], selectivity["size"]], axis=0
    )

    # Find top selective components for each attribute
    top_k = min(10, C)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heatmap of selectivity
    ax = axes[0, 0]
    im = ax.imshow(selectivity_matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Attribute")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Shape", "Color", "Size"])
    ax.set_title("Component Selectivity by Attribute Type")
    plt.colorbar(im, ax=ax, label="Selectivity")

    # Bar chart of most shape-selective components
    ax = axes[0, 1]
    top_shape = np.argsort(selectivity["shape"])[-top_k:][::-1]
    ax.bar(range(top_k), selectivity["shape"][top_shape], color="coral")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"C{c}" for c in top_shape], rotation=45)
    ax.set_ylabel("Selectivity")
    ax.set_title(f"Top {top_k} Shape-Selective Components")

    # Bar chart of most color-selective components
    ax = axes[1, 0]
    top_color = np.argsort(selectivity["color"])[-top_k:][::-1]
    ax.bar(range(top_k), selectivity["color"][top_color], color="forestgreen")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"C{c}" for c in top_color], rotation=45)
    ax.set_ylabel("Selectivity")
    ax.set_title(f"Top {top_k} Color-Selective Components")

    # Bar chart of most size-selective components
    ax = axes[1, 1]
    top_size = np.argsort(selectivity["size"])[-top_k:][::-1]
    ax.bar(range(top_k), selectivity["size"][top_size], color="steelblue")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"C{c}" for c in top_size], rotation=45)
    ax.set_ylabel("Selectivity")
    ax.set_title(f"Top {top_k} Size-Selective Components")

    plt.suptitle("Component Selectivity Analysis", fontsize=14)
    plt.tight_layout()

    selectivity_path = out_dir / "component_selectivity.png"
    plt.savefig(selectivity_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved selectivity plot to {selectivity_path}")

    if log_wandb:
        wandb.log({"visualizations/component_selectivity": wandb.Image(str(selectivity_path))})

    # Print summary
    logger.info("\n=== Component Selectivity Summary ===")
    logger.info(f"Top shape-selective components: {list(top_shape)}")
    logger.info(f"Top color-selective components: {list(top_color)}")
    logger.info(f"Top size-selective components: {list(top_size)}")

    # Count how many components are primarily selective for each attribute
    primary_attr = np.argmax(selectivity_matrix, axis=0)
    attr_names = ["shape", "color", "size"]
    for i, attr in enumerate(attr_names):
        count = np.sum(primary_attr == i)
        logger.info(f"Components primarily selective for {attr}: {count}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

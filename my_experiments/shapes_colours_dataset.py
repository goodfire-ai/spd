"""
Synthetic Multi-Attribute Dataset for SPD ConvNet Experiments

Generates images with independent attributes:
- Shape: circle, square, triangle (3 classes)
- Color: red, green, blue (3 classes)
- Size: small, large (2 classes)

Ground truth: 3 + 3 + 2 = 8 atomic mechanisms
If SPD finds 8 components that separate by attribute, that's strong evidence
it discovers atomic features rather than composed ones (like "red-square").
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Dataset Generation
# =============================================================================


class MultiAttributeShapesDataset(Dataset):
    """
    Synthetic dataset of shapes with color and size attributes.

    Attributes:
        - shape: 0=circle, 1=square, 2=triangle
        - color: 0=red, 1=green, 2=blue
        - size: 0=small, 1=large
    """

    SHAPES = ["circle", "square", "triangle"]
    COLORS = ["red", "green", "blue"]
    SIZES = ["small", "large"]

    # RGB values for colors
    COLOR_MAP = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }

    def __init__(
        self,
        n_samples: int = 10000,
        img_size: int = 32,
        seed: int | None = 42,
        noise_level: float = 0.0,
        position_jitter: bool = True,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            img_size: Size of square images (img_size x img_size)
            seed: Random seed for reproducibility
            noise_level: Std of Gaussian noise to add (0.0 = no noise)
            position_jitter: If True, randomly offset shape position slightly
        """
        self.n_samples = n_samples
        self.img_size = img_size
        self.noise_level = noise_level
        self.position_jitter = position_jitter

        if seed is not None:
            np.random.seed(seed)

        # Generate all combinations uniformly
        self.labels = self._generate_labels()
        self.images = self._generate_images()

    def _generate_labels(self) -> dict[str, np.ndarray]:
        """Generate random attribute labels for all samples."""
        return {
            "shape": np.random.randint(0, 3, size=self.n_samples),
            "color": np.random.randint(0, 3, size=self.n_samples),
            "size": np.random.randint(0, 2, size=self.n_samples),
        }

    def _generate_images(self) -> np.ndarray:
        """Generate all images based on labels."""
        images = np.zeros((self.n_samples, 3, self.img_size, self.img_size), dtype=np.float32)

        for i in range(self.n_samples):
            img = self._render_single_image(
                shape=self.SHAPES[self.labels["shape"][i]],
                color=self.COLORS[self.labels["color"][i]],
                size=self.SIZES[self.labels["size"][i]],
            )
            images[i] = img

        return images

    def _render_single_image(self, shape: str, color: str, size: str) -> np.ndarray:
        """Render a single image with the given attributes."""
        img = Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Determine shape size
        radius = self.img_size // 6 if size == "small" else self.img_size // 3

        # Center position with optional jitter
        cx, cy = self.img_size // 2, self.img_size // 2
        if self.position_jitter:
            max_jitter = self.img_size // 8
            cx += np.random.randint(-max_jitter, max_jitter + 1)
            cy += np.random.randint(-max_jitter, max_jitter + 1)

        rgb = self.COLOR_MAP[color]

        # Draw shape
        if shape == "circle":
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, fill=rgb)

        elif shape == "square":
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.rectangle(bbox, fill=rgb)

        elif shape == "triangle":
            # Equilateral triangle pointing up
            points = [
                (cx, cy - radius),  # top
                (cx - radius, cy + radius),  # bottom left
                (cx + radius, cy + radius),  # bottom right
            ]
            draw.polygon(points, fill=rgb)

        # Convert to numpy array (C, H, W) format, normalized to [0, 1]
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add noise if specified
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, img_array.shape).astype(np.float32)
            img_array = np.clip(img_array + noise, 0, 1)

        return img_array

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns:
            image: (3, H, W) tensor
            labels: dict with 'shape', 'color', 'size' tensors
        """
        image = torch.from_numpy(self.images[idx])
        labels = {
            "shape": torch.tensor(self.labels["shape"][idx], dtype=torch.long),
            "color": torch.tensor(self.labels["color"][idx], dtype=torch.long),
            "size": torch.tensor(self.labels["size"][idx], dtype=torch.long),
        }
        return image, labels


def visualize_samples(dataset: MultiAttributeShapesDataset, n_samples: int = 16):
    """Visualize random samples from the dataset."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for ax, idx in zip(axes, indices, strict=True):
        img, labels = dataset[idx]
        # Convert from (C, H, W) to (H, W, C) for plotting
        img_np = img.numpy().transpose(1, 2, 0)
        ax.imshow(img_np)

        shape = dataset.SHAPES[labels["shape"].item()]
        color = dataset.COLORS[labels["color"].item()]
        size = dataset.SIZES[labels["size"].item()]
        ax.set_title(f"{size} {color} {shape}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150)
    plt.show()
    print("Saved visualization to dataset_samples.png")


# =============================================================================
# Model Architecture
# =============================================================================


class MultiAttributeCNNSingleHead(nn.Module):
    """
    Alternative: Single MLP head that predicts all attributes.

    This version is closer to standard SPD setup where we decompose
    a single MLP. The output is concatenated [shape_logits, color_logits, size_logits].
    """

    def __init__(
        self,
        img_size: int = 32,
        hidden_dim: int = 64,
        n_shapes: int = 3,
        n_colors: int = 3,
        n_sizes: int = 2,
    ):
        super().__init__()

        self.img_size = img_size
        self.n_shapes = n_shapes
        self.n_colors = n_colors
        self.n_sizes = n_sizes

        # Convolutional backbone (same as before)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flat_size = 64 * (img_size // 8) * (img_size // 8)

        # Single MLP for all predictions
        self.fc1 = nn.Linear(self.flat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_shapes + n_colors + n_sizes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Conv backbone
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten and MLP
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        # Split into attribute predictions
        return {
            "shape": logits[:, : self.n_shapes],
            "color": logits[:, self.n_shapes : self.n_shapes + self.n_colors],
            "size": logits[:, -self.n_sizes :],
        }


# =============================================================================
# Training
# =============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, list]:
    """Train the multi-attribute model."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_acc_shape": [],
        "test_acc_color": [],
        "test_acc_size": [],
    }

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)

            # Sum of cross-entropy losses for each attribute
            loss = (
                F.cross_entropy(outputs["shape"], labels["shape"])
                + F.cross_entropy(outputs["color"], labels["color"])
                + F.cross_entropy(outputs["size"], labels["size"])
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Evaluation
        model.eval()
        correct = {"shape": 0, "color": 0, "size": 0}
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(images)

                for attr in ["shape", "color", "size"]:
                    preds = outputs[attr].argmax(dim=1)
                    correct[attr] += (preds == labels[attr]).sum().item()
                total += labels["shape"].size(0)

        for attr in ["shape", "color", "size"]:
            acc = correct[attr] / total
            history[f"test_acc_{attr}"].append(acc)

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Shape: {history['test_acc_shape'][-1]:.3f} | "
            f"Color: {history['test_acc_color'][-1]:.3f} | "
            f"Size: {history['test_acc_size'][-1]:.3f}"
        )

    return history


def plot_training_history(history: dict[str, list]):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")

    ax2.plot(history["test_acc_shape"], label="Shape")
    ax2.plot(history["test_acc_color"], label="Color")
    ax2.plot(history["test_acc_size"], label="Size")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Test Accuracy by Attribute")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved training history to training_history.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultiAttributeShapesDataset(n_samples=10000, seed=42)
    test_dataset = MultiAttributeShapesDataset(n_samples=2000, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Visualize some samples
    print("\nVisualizing samples...")
    visualize_samples(train_dataset)

    # Create and train model
    print("\nTraining model...")
    model = MultiAttributeCNNSingleHead(img_size=32, hidden_dim=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(model, train_loader, test_loader, epochs=20)

    # Plot results
    plot_training_history(history)

    # Save model
    torch.save(model.state_dict(), "multi_attribute_cnn.pt")
    print("\nSaved model to multi_attribute_cnn.pt")

    # Print final results
    print("\n" + "=" * 50)
    print("Final Test Accuracies:")
    print(f"  Shape: {history['test_acc_shape'][-1]:.1%}")
    print(f"  Color: {history['test_acc_color'][-1]:.1%}")
    print(f"  Size:  {history['test_acc_size'][-1]:.1%}")
    print("=" * 50)

    print("\nGround truth: 8 atomic mechanisms (3 shapes + 3 colors + 2 sizes)")
    print("If SPD finds 8 components that separate by attribute type,")
    print("that's evidence for discovering atomic vs composed features.")

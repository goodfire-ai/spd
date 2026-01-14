"""Interactive dashboard for inspecting learned MNIST components."""

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComponentDashboard:
    """Dashboard for visualizing and analyzing learned components."""

    def __init__(
        self,
        experiment_dir: str,
        checkpoint_step: int | None = None,
        device: str = "cpu",
        n_samples: int = 1000,
    ):
        """Initialize the dashboard.

        Args:
            experiment_dir: Path to experiment output directory
            checkpoint_step: Which checkpoint to load (None = latest)
            device: Device to run on
            n_samples: Number of test samples to analyze
        """
        self.experiment_dir = Path(experiment_dir)
        self.device = device
        self.n_samples = n_samples

        # Load config
        config_path = self.experiment_dir / "final_config.yaml"
        with open(config_path) as f:
            import yaml

            config_dict = yaml.safe_load(f)
        self.config = Config(**config_dict)

        # Load the original model
        self.model = TwoLayerMLP(
            input_size=784,
            hidden_size=128,  # This should match what was used in training
            num_classes=10,
        )
        model_path = self.experiment_dir / "trained_mlp.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Create and load component model
        module_path_info = expand_module_patterns(self.model, self.config.all_module_info)
        self.component_model = ComponentModel(
            target_model=self.model,
            module_path_info=module_path_info,
            ci_fn_type=self.config.ci_fn_type,
            ci_fn_hidden_dims=self.config.ci_fn_hidden_dims,
            pretrained_model_output_attr=self.config.pretrained_model_output_attr,
            sigmoid_type=self.config.sigmoid_type,
        )

        # Find and load checkpoint
        if checkpoint_step is None:
            checkpoints = list(self.experiment_dir.glob("model_*.pth"))
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {self.experiment_dir}")
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        else:
            checkpoint_path = self.experiment_dir / f"model_{checkpoint_step}.pth"

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.component_model.load_state_dict(checkpoint_weights)
        self.component_model.to(device)
        self.component_model.eval()

        # Load MNIST test dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Pre-compute activations for test set
        print(f"Pre-computing component activations for {n_samples} test samples...")
        self._precompute_activations()
        print("Dashboard ready!")

    def _precompute_activations(self):
        """Pre-compute component activations for all test samples."""
        # Limit to n_samples
        indices = list(range(min(self.n_samples, len(self.test_dataset))))

        self.test_images = []
        self.test_labels = []
        self.activations_fc1 = []
        self.activations_fc2 = []
        self.causal_importance_fc1 = []
        self.causal_importance_fc2 = []
        self.pre_sigmoid_ci_fc1 = []
        self.pre_sigmoid_ci_fc2 = []

        with torch.no_grad():
            for idx in indices:
                image, label = self.test_dataset[idx]
                image_flat = image.view(1, -1).to(self.device)

                self.test_images.append(image.squeeze().numpy())
                self.test_labels.append(label)

                # Get activations and causal importance for both layers
                if "fc1" in self.component_model.components:
                    component_fc1 = self.component_model.components["fc1"]
                    acts_fc1 = component_fc1.get_component_acts(image_flat)
                    self.activations_fc1.append(acts_fc1[0].cpu().numpy())

                if "fc2" in self.component_model.components:
                    # Need to get hidden representation first
                    hidden = torch.relu(self.model.fc1(image_flat))
                    component_fc2 = self.component_model.components["fc2"]
                    acts_fc2 = component_fc2.get_component_acts(hidden)
                    self.activations_fc2.append(acts_fc2[0].cpu().numpy())

                # Get causal importance for all layers using ComponentModel
                pre_weight_acts = {}
                if "fc1" in self.component_model.components:
                    pre_weight_acts["fc1"] = image_flat
                if "fc2" in self.component_model.components:
                    hidden = torch.relu(self.model.fc1(image_flat))
                    pre_weight_acts["fc2"] = hidden

                ci_outputs = self.component_model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    sampling="continuous",
                    detach_inputs=False,
                )

                # Extract CI values for each layer (using lower_leaky which is the main CI)
                if "fc1" in ci_outputs.lower_leaky:
                    ci_fc1 = ci_outputs.lower_leaky["fc1"]
                    self.causal_importance_fc1.append(ci_fc1[0].cpu().numpy())

                if "fc2" in ci_outputs.lower_leaky:
                    ci_fc2 = ci_outputs.lower_leaky["fc2"]
                    self.causal_importance_fc2.append(ci_fc2[0].cpu().numpy())

                # Also store pre-sigmoid CI values
                if "fc1" in ci_outputs.pre_sigmoid:
                    pre_sigmoid_fc1 = ci_outputs.pre_sigmoid["fc1"]
                    self.pre_sigmoid_ci_fc1.append(pre_sigmoid_fc1[0].cpu().numpy())

                if "fc2" in ci_outputs.pre_sigmoid:
                    pre_sigmoid_fc2 = ci_outputs.pre_sigmoid["fc2"]
                    self.pre_sigmoid_ci_fc2.append(pre_sigmoid_fc2[0].cpu().numpy())

        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)
        if self.activations_fc1:
            self.activations_fc1 = np.array(self.activations_fc1)
            self.causal_importance_fc1 = np.array(self.causal_importance_fc1)
        if self.activations_fc2:
            self.activations_fc2 = np.array(self.activations_fc2)
            self.causal_importance_fc2 = np.array(self.causal_importance_fc2)
        if self.pre_sigmoid_ci_fc1:
            self.pre_sigmoid_ci_fc1 = np.array(self.pre_sigmoid_ci_fc1)
        if self.pre_sigmoid_ci_fc2:
            self.pre_sigmoid_ci_fc2 = np.array(self.pre_sigmoid_ci_fc2)

    def get_component_direction_plot(self, layer: str, component_idx: int):
        """Plot the component direction in input/output space."""
        if layer not in self.component_model.components:
            return None

        component = self.component_model.components[layer]
        V = component.V.detach().cpu().numpy()
        U = component.U.detach().cpu().numpy()

        if component_idx >= V.shape[1]:
            return None

        fig, ax = plt.subplots(figsize=(6, 6))

        if layer == "fc1":
            # For fc1, show V direction (input space) as 28x28 image
            direction = V[:, component_idx]
            direction_img = direction.reshape(28, 28)
            # Use symmetric colormap centered at 0 to show positive/negative contributions
            abs_max = max(abs(direction_img.min()), abs(direction_img.max()))
            if abs_max < 1e-6:
                abs_max = 1.0  # Avoid division by zero for dead components

            im = ax.imshow(direction_img, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
            ax.set_title(f"{layer} - Component {component_idx}\nDirection in Input Space (V)")
            plt.colorbar(im, ax=ax, label="Weight value")
            ax.axis("off")
        else:
            # For fc2, show U direction (output space = 10 classes)
            direction = U[component_idx, :]  # U shape is (C, d_out)
            colors = plt.cm.tab10(np.arange(10))
            ax.bar(range(10), direction, color=colors)
            ax.set_xlabel("Output Class")
            ax.set_ylabel("Weight")
            ax.set_xticks(range(10))
            ax.set_xticklabels([str(i) for i in range(10)])
            ax.set_title(f"{layer} - Component {component_idx}\nDirection in Output Space (U)")
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        return fig

    def get_activation_distribution(self, layer: str, component_idx: int):
        """Plot the distribution of activation values for this component."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(acts, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.axvline(acts.mean(), color="red", linestyle="--", label=f"Mean: {acts.mean():.3f}")
        ax1.axvline(
            np.median(acts), color="orange", linestyle="--", label=f"Median: {np.median(acts):.3f}"
        )
        ax1.set_xlabel("Inner Activation Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{layer} - Component {component_idx}\nInner Activation Distribution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Log histogram (if there are positive values)
        positive_acts = acts[acts > 0]
        if len(positive_acts) > 0:
            ax2.hist(np.log10(positive_acts + 1e-10), bins=50, color="green", alpha=0.7)
            ax2.set_xlabel("Log10(Inner Activation Value)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Log-scale Distribution (positive inner activations)")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No positive inner activations", ha="center", va="center")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def get_causal_importance_distribution(self, layer: str, component_idx: int):
        """Plot the distribution of causal importance values for this component."""
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0 or component_idx >= ci_values.shape[1]:
            return None

        ci = ci_values[:, component_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(ci, bins=50, color="purple", alpha=0.7, edgecolor="black")
        ax1.axvline(ci.mean(), color="red", linestyle="--", label=f"Mean: {ci.mean():.3f}")
        ax1.axvline(
            np.median(ci), color="orange", linestyle="--", label=f"Median: {np.median(ci):.3f}"
        )
        ax1.set_xlabel("Causal Importance")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{layer} - Component {component_idx}\nCausal Importance Distribution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # CDF plot
        sorted_ci = np.sort(ci)
        cdf = np.arange(1, len(sorted_ci) + 1) / len(sorted_ci)
        ax2.plot(sorted_ci, cdf, color="purple", linewidth=2)
        ax2.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Median")
        ax2.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90th percentile")
        ax2.set_xlabel("Causal Importance")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("Cumulative Distribution Function")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def get_top_activating_examples(
        self,
        layer: str,
        component_idx: int,
        n_examples: int = 16,
        show_inner_activation: bool = False,
    ):
        """Show examples with highest activation for this component.

        Args:
            layer: Which layer to show examples for
            component_idx: Which component to show
            n_examples: Number of examples to show
            show_inner_activation: If True and layer is fc1, show per-pixel contribution
                (image * V[:, component_idx]) instead of raw image
        """
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]
        top_indices = np.argsort(acts)[-n_examples:][::-1]

        # Get V matrix for fc1 if showing inner activation
        V_component = None
        if show_inner_activation and layer == "fc1" and "fc1" in self.component_model.components:
            V = self.component_model.components["fc1"].V.detach().cpu().numpy()
            V_component = V[:, component_idx].reshape(28, 28)

        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            if show_inner_activation and V_component is not None:
                # Show per-pixel contribution: image * V
                inner_act = self.test_images[idx] * V_component
                # Use symmetric colormap centered at 0
                abs_max = max(abs(inner_act.min()), abs(inner_act.max()))
                if abs_max < 1e-6:
                    abs_max = 1.0
                im = axes[i].imshow(inner_act, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
                plt.colorbar(im, ax=axes[i], fraction=0.046)
            else:
                axes[i].imshow(self.test_images[idx], cmap="gray")
            axes[i].set_title(f"Label: {self.test_labels[idx]}\nAct: {acts[idx]:.3f}", fontsize=9)
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        title_suffix = " (pixel × V)" if (show_inner_activation and V_component is not None) else ""
        plt.suptitle(
            f"{layer} - Component {component_idx}\nTop {n_examples} by Inner Activation{title_suffix}",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def get_top_ci_examples(self, layer: str, component_idx: int, n_examples: int = 16):
        """Show examples with highest causal importance for this component.

        Uses pre-sigmoid CI values for ranking and display.
        """
        pre_sigmoid_ci = self.pre_sigmoid_ci_fc1 if layer == "fc1" else self.pre_sigmoid_ci_fc2
        if len(pre_sigmoid_ci) == 0 or component_idx >= pre_sigmoid_ci.shape[1]:
            return None

        ci = pre_sigmoid_ci[:, component_idx]
        top_indices = np.argsort(ci)[-n_examples:][::-1]

        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            axes[i].imshow(self.test_images[idx], cmap="gray")
            axes[i].set_title(
                f"Label: {self.test_labels[idx]}\nCI (pre-σ): {ci[idx]:.3f}", fontsize=9
            )
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        plt.suptitle(
            f"{layer} - Component {component_idx}\nTop {n_examples} by Causal Importance (pre-sigmoid)",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def get_class_activation_analysis(self, layer: str, component_idx: int):
        """Analyze component activations by digit class."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]

        # Group by class
        class_activations = {digit: [] for digit in range(10)}
        for act, label in zip(acts, self.test_labels, strict=False):
            class_activations[label].append(act)

        # Compute statistics
        class_means = {digit: np.mean(vals) for digit, vals in class_activations.items()}
        class_stds = {digit: np.std(vals) for digit, vals in class_activations.items()}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot of means
        digits = list(range(10))
        means = [class_means[d] for d in digits]
        stds = [class_stds[d] for d in digits]

        ax1.bar(digits, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
        ax1.set_xlabel("Digit Class")
        ax1.set_ylabel("Mean Inner Activation")
        ax1.set_title(f"{layer} - Component {component_idx}\nMean Inner Activation by Class")
        ax1.set_xticks(digits)
        ax1.grid(axis="y", alpha=0.3)

        # Box plot
        box_data = [class_activations[d] for d in digits]
        bp = ax2.boxplot(box_data, labels=digits, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax2.set_xlabel("Digit Class")
        ax2.set_ylabel("Inner Activation Value")
        ax2.set_title(
            f"{layer} - Component {component_idx}\nInner Activation Distribution by Class"
        )
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def get_class_ci_analysis(self, layer: str, component_idx: int):
        """Analyze component causal importance by digit class."""
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0 or component_idx >= ci_values.shape[1]:
            return None

        ci = ci_values[:, component_idx]

        # Group by class
        class_ci = {digit: [] for digit in range(10)}
        for ci_val, label in zip(ci, self.test_labels, strict=False):
            class_ci[label].append(ci_val)

        # Compute statistics
        class_means = {digit: np.mean(vals) for digit, vals in class_ci.items()}
        class_stds = {digit: np.std(vals) for digit, vals in class_ci.items()}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot of means
        digits = list(range(10))
        means = [class_means[d] for d in digits]
        stds = [class_stds[d] for d in digits]

        ax1.bar(digits, means, yerr=stds, capsize=5, color="purple", alpha=0.7)
        ax1.set_xlabel("Digit Class")
        ax1.set_ylabel("Mean Causal Importance")
        ax1.set_title(f"{layer} - Component {component_idx}\nMean Causal Importance by Class")
        ax1.set_xticks(digits)
        ax1.grid(axis="y", alpha=0.3)

        # Box plot
        box_data = [class_ci[d] for d in digits]
        bp = ax2.boxplot(box_data, labels=digits, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("plum")
        ax2.set_xlabel("Digit Class")
        ax2.set_ylabel("Causal Importance")
        ax2.set_title(f"{layer} - Component {component_idx}\nCI Distribution by Class")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def get_selectivity_metrics(self, layer: str, component_idx: int):
        """Compute selectivity metrics for the component."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2

        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return "Component not found"

        acts = activations[:, component_idx]
        ci = ci_values[:, component_idx]

        # Group by class
        class_activations = {digit: [] for digit in range(10)}
        class_ci = {digit: [] for digit in range(10)}
        for act, ci_val, label in zip(acts, ci, self.test_labels, strict=False):
            class_activations[label].append(act)
            class_ci[label].append(ci_val)

        class_act_means = np.array([np.mean(class_activations[d]) for d in range(10)])
        class_ci_means = np.array([np.mean(class_ci[d]) for d in range(10)])

        # Compute activation metrics
        sparsity = (acts == 0).mean()
        mean_act = acts.mean()
        max_act = acts.max()
        std_act = acts.std()

        # Compute CI metrics
        ci_sparsity = (ci < 0.01).mean()  # Effectively zero CI
        mean_ci = ci.mean()
        max_ci = ci.max()
        std_ci = ci.std()
        median_ci = np.median(ci)
        percentile_90_ci = np.percentile(ci, 90)

        # Selectivity index (how concentrated is activation on one class)
        max_class_act_mean = class_act_means.max()
        act_selectivity_idx = max_class_act_mean / (class_act_means.mean() + 1e-10)

        max_class_ci_mean = class_ci_means.max()
        ci_selectivity_idx = max_class_ci_mean / (class_ci_means.mean() + 1e-10)

        # Preferred class
        preferred_class_act = class_act_means.argmax()
        preferred_class_ci = class_ci_means.argmax()

        metrics_text = f"""
**Component {component_idx} - {layer}**

**Inner Activation Statistics:**
- Mean: {mean_act:.4f}
- Max: {max_act:.4f}
- Std: {std_act:.4f}
- Sparsity (% zeros): {sparsity * 100:.2f}%

**Causal Importance Statistics:**
- Mean CI: {mean_ci:.4f}
- Median CI: {median_ci:.4f}
- Max CI: {max_ci:.4f}
- Std CI: {std_ci:.4f}
- 90th Percentile: {percentile_90_ci:.4f}
- Sparsity (CI < 0.01): {ci_sparsity * 100:.2f}%

**Inner Activation Class Selectivity:**
- Preferred Class: {preferred_class_act}
- Selectivity Index: {act_selectivity_idx:.3f}

**CI Class Selectivity:**
- Preferred Class: {preferred_class_ci}
- Selectivity Index: {ci_selectivity_idx:.3f}

**Class-wise Mean CI:**
"""
        for digit in range(10):
            metrics_text += f"- Digit {digit}: {class_ci_means[digit]:.4f}\n"

        return metrics_text

    def get_ci_difference_image(self, layer: str, component_idx: int, threshold: float = 0.01):
        """Show difference between mean images where CI > threshold vs CI < threshold.

        This reveals what input patterns the component is sensitive to.
        """
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0 or component_idx >= ci_values.shape[1]:
            return None

        ci = ci_values[:, component_idx]

        # Split images by CI threshold
        high_ci_mask = ci > threshold
        low_ci_mask = ci <= threshold

        high_ci_images = self.test_images[high_ci_mask]
        low_ci_images = self.test_images[low_ci_mask]

        # Compute mean images
        mean_high = np.zeros((28, 28)) if len(high_ci_images) == 0 else high_ci_images.mean(axis=0)

        mean_low = np.zeros((28, 28)) if len(low_ci_images) == 0 else low_ci_images.mean(axis=0)

        # Compute difference
        diff = mean_high - mean_low

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Mean image where CI > threshold
        im0 = axes[0].imshow(mean_high, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Mean Image (CI > {threshold})\nn={high_ci_mask.sum()}")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Mean image where CI <= threshold
        im1 = axes[1].imshow(mean_low, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Mean Image (CI ≤ {threshold})\nn={low_ci_mask.sum()}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Difference (symmetric colormap centered at 0)
        abs_max = max(abs(diff.min()), abs(diff.max()))
        if abs_max < 1e-6:
            abs_max = 1.0
        im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        axes[2].set_title("Difference (High CI - Low CI)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Δ intensity")

        plt.suptitle(
            f"{layer} - Component {component_idx}\nMean Input Images by Causal Importance",
            fontsize=14,
        )
        plt.tight_layout()
        return fig


def create_dashboard(experiment_dir: str, checkpoint_step: int | None = None):
    """Create the Gradio dashboard interface."""
    dashboard = ComponentDashboard(
        experiment_dir=experiment_dir,
        checkpoint_step=checkpoint_step,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_samples=1000,
    )

    # Get available layers and components
    available_layers = list(dashboard.component_model.components.keys())
    max_components = {
        layer: dashboard.component_model.components[layer].V.shape[1] for layer in available_layers
    }

    # Compute alive components for each layer (CI sparsity < 99.5%)
    def get_alive_components(layer: str, threshold: float = 0.01) -> list[int]:
        """Get list of component indices that are not dead (have CI > threshold for >0.5% of samples)."""
        ci_values = (
            dashboard.causal_importance_fc1 if layer == "fc1" else dashboard.causal_importance_fc2
        )
        if len(ci_values) == 0:
            return list(range(max_components[layer]))

        alive = []
        for comp_idx in range(ci_values.shape[1]):
            ci = ci_values[:, comp_idx]
            # Component is alive if more than 0.5% of samples have CI > threshold
            # (i.e., sparsity < 99.5%)
            if (ci > threshold).mean() > 0.005:
                alive.append(comp_idx)
        return alive

    alive_components = {layer: get_alive_components(layer) for layer in available_layers}

    # Log alive component counts
    for layer in available_layers:
        total = max_components[layer]
        alive = len(alive_components[layer])
        print(f"{layer}: {alive}/{total} components alive ({100 * alive / total:.1f}%)")

    def update_component_viz(layer, component_idx, show_inner_activation):
        """Update all visualizations for selected component."""
        if component_idx >= max_components.get(layer, 0):
            component_idx = 0

        direction_plot = dashboard.get_component_direction_plot(layer, component_idx)
        dist_plot = dashboard.get_activation_distribution(layer, component_idx)
        ci_dist_plot = dashboard.get_causal_importance_distribution(layer, component_idx)
        top_examples = dashboard.get_top_activating_examples(
            layer, component_idx, show_inner_activation=show_inner_activation
        )
        top_ci_examples = dashboard.get_top_ci_examples(layer, component_idx)
        class_analysis = dashboard.get_class_activation_analysis(layer, component_idx)
        class_ci_analysis = dashboard.get_class_ci_analysis(layer, component_idx)
        metrics = dashboard.get_selectivity_metrics(layer, component_idx)
        ci_diff_image = dashboard.get_ci_difference_image(layer, component_idx)

        return (
            direction_plot,
            dist_plot,
            ci_dist_plot,
            top_examples,
            top_ci_examples,
            class_analysis,
            class_ci_analysis,
            metrics,
            ci_diff_image,
        )

    # Create Gradio interface
    with gr.Blocks(title="MNIST Component Dashboard") as demo:
        gr.Markdown("# MNIST Component Inspector Dashboard")
        gr.Markdown(
            f"Exploring components from: `{experiment_dir}`\n\n"
            f"Available layers: {', '.join(available_layers)}"
        )

        with gr.Row():
            layer_dropdown = gr.Dropdown(
                choices=available_layers,
                value=available_layers[0],
                label="Layer",
            )
            component_dropdown = gr.Dropdown(
                choices=[(f"Component {i}", i) for i in alive_components[available_layers[0]]],
                value=alive_components[available_layers[0]][0]
                if alive_components[available_layers[0]]
                else 0,
                label="Component Index",
            )
            hide_dead_toggle = gr.Checkbox(
                value=True,
                label="Hide dead components (CI sparsity >= 99.5%)",
            )

        # Stats about alive components
        alive_stats = gr.Markdown(
            f"**{available_layers[0]}**: {len(alive_components[available_layers[0]])}/{max_components[available_layers[0]]} "
            f"components alive ({100 * len(alive_components[available_layers[0]]) / max_components[available_layers[0]]:.1f}%)"
        )

        # Update component dropdown when layer or toggle changes
        def update_component_dropdown(layer, hide_dead):
            if hide_dead:
                choices = [(f"Component {i}", i) for i in alive_components[layer]]
            else:
                choices = [(f"Component {i}", i) for i in range(max_components[layer])]

            # Stats text
            alive_count = len(alive_components[layer])
            total_count = max_components[layer]
            stats = f"**{layer}**: {alive_count}/{total_count} components alive ({100 * alive_count / total_count:.1f}%)"

            default_value = choices[0][1] if choices else 0
            return gr.Dropdown(choices=choices, value=default_value), stats

        layer_dropdown.change(
            update_component_dropdown,
            inputs=[layer_dropdown, hide_dead_toggle],
            outputs=[component_dropdown, alive_stats],
        )
        hide_dead_toggle.change(
            update_component_dropdown,
            inputs=[layer_dropdown, hide_dead_toggle],
            outputs=[component_dropdown, alive_stats],
        )

        with gr.Row():
            with gr.Column():
                direction_plot = gr.Plot(label="Component Direction")
            with gr.Column():
                metrics_text = gr.Markdown(label="Selectivity Metrics")

        gr.Markdown("## Causal Importance Statistics")
        with gr.Row():
            ci_dist_plot = gr.Plot(label="Causal Importance Distribution")
            class_ci_analysis = gr.Plot(label="Causal Importance by Class")

        with gr.Row():
            top_ci_examples = gr.Plot(label="Top CI Examples")

        with gr.Row():
            ci_diff_image = gr.Plot(label="Mean Image Difference by CI")

        gr.Markdown("## Inner Activation Statistics")
        with gr.Row():
            dist_plot = gr.Plot(label="Inner Activation Distribution")
            class_analysis = gr.Plot(label="Inner Activation by Class")

        with gr.Row():
            show_inner_act_toggle = gr.Checkbox(
                value=True,
                label="Show inner activation (pixel × V) - fc1 only",
            )
        with gr.Row():
            top_examples = gr.Plot(label="Top by Inner Activation")

        # Update visualizations when component selection changes
        component_dropdown.change(
            update_component_viz,
            inputs=[layer_dropdown, component_dropdown, show_inner_act_toggle],
            outputs=[
                direction_plot,
                dist_plot,
                ci_dist_plot,
                top_examples,
                top_ci_examples,
                class_analysis,
                class_ci_analysis,
                metrics_text,
                ci_diff_image,
            ],
        )

        # Update when show inner activation toggle changes
        show_inner_act_toggle.change(
            update_component_viz,
            inputs=[layer_dropdown, component_dropdown, show_inner_act_toggle],
            outputs=[
                direction_plot,
                dist_plot,
                ci_dist_plot,
                top_examples,
                top_ci_examples,
                class_analysis,
                class_ci_analysis,
                metrics_text,
                ci_diff_image,
            ],
        )

        # Initial load
        demo.load(
            update_component_viz,
            inputs=[layer_dropdown, component_dropdown, show_inner_act_toggle],
            outputs=[
                direction_plot,
                dist_plot,
                ci_dist_plot,
                top_examples,
                top_ci_examples,
                class_analysis,
                class_ci_analysis,
                metrics_text,
                ci_diff_image,
            ],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch MNIST Component Dashboard")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing checkpoints and config",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific checkpoint step to load (default: latest)",
    )
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")

    args = parser.parse_args()

    demo = create_dashboard(
        experiment_dir=args.experiment_dir,
        checkpoint_step=args.checkpoint_step,
    )

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",  # Listen on all network interfaces
    )

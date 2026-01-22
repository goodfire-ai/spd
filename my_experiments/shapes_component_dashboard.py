"""Interactive dashboard for inspecting learned shapes/colors/sizes components."""

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapes_colours_dataset import MultiAttributeShapesDataset

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


class MultiAttributeCNNSingleHead(nn.Module):
    """CNN for multi-attribute classification (same as in shapes_colours_dataset.py)."""

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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flat_size = 64 * (img_size // 8) * (img_size // 8)

        self.fc1 = nn.Linear(self.flat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_shapes + n_colors + n_sizes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return {
            "shape": logits[:, : self.n_shapes],
            "color": logits[:, self.n_shapes : self.n_shapes + self.n_colors],
            "size": logits[:, -self.n_sizes :],
        }


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


# Attribute labels
SHAPE_NAMES = ["circle", "square", "triangle"]
COLOR_NAMES = ["red", "green", "blue"]
SIZE_NAMES = ["small", "large"]


class ShapesComponentDashboard:
    """Dashboard for visualizing and analyzing learned components from shapes experiment."""

    def __init__(
        self,
        experiment_dir: str,
        checkpoint_step: int | None = None,
        device: str = "cpu",
        n_samples: int = 1000,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.device = device
        self.n_samples = n_samples

        # Load config
        config_path = self.experiment_dir / "final_config.yaml"
        with open(config_path) as f:
            import yaml

            config_dict = yaml.safe_load(f)
        self.config = Config(**config_dict)

        # Load the full CNN model
        self.cnn_model = MultiAttributeCNNSingleHead(img_size=32, hidden_dim=64)
        cnn_path = self.experiment_dir / "trained_shapes_cnn.pth"
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        self.cnn_model = self.cnn_model.to(device)
        self.cnn_model.eval()
        self.cnn_model.requires_grad_(False)

        # Create the MLP model (for decomposition)
        output_dim = self.cnn_model.n_shapes + self.cnn_model.n_colors + self.cnn_model.n_sizes
        self.mlp_model = ShapesMLP(
            flat_size=self.cnn_model.flat_size,
            hidden_dim=64,
            output_dim=output_dim,
        )
        # Copy weights from CNN
        self.mlp_model.fc1.weight.data = self.cnn_model.fc1.weight.data.clone()
        self.mlp_model.fc1.bias.data = self.cnn_model.fc1.bias.data.clone()
        self.mlp_model.fc2.weight.data = self.cnn_model.fc2.weight.data.clone()
        self.mlp_model.fc2.bias.data = self.cnn_model.fc2.bias.data.clone()
        self.mlp_model = self.mlp_model.to(device)
        self.mlp_model.eval()
        self.mlp_model.requires_grad_(False)

        # Create and load component model
        module_path_info = expand_module_patterns(self.mlp_model, self.config.all_module_info)
        self.component_model = ComponentModel(
            target_model=self.mlp_model,
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

        # Create test dataset
        self.test_dataset = MultiAttributeShapesDataset(n_samples=n_samples, seed=999)

        # Pre-compute activations
        print(f"Pre-computing component activations for {n_samples} test samples...")
        self._precompute_activations()
        print("Dashboard ready!")

    def _get_conv_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get flattened conv features from images."""
        with torch.no_grad():
            x = self.cnn_model.pool(F.relu(self.cnn_model.conv1(images)))
            x = self.cnn_model.pool(F.relu(self.cnn_model.conv2(x)))
            x = self.cnn_model.pool(F.relu(self.cnn_model.conv3(x)))
            return x.view(x.size(0), -1)

    def _precompute_activations(self):
        """Pre-compute component activations for all test samples."""
        indices = list(range(min(self.n_samples, len(self.test_dataset))))

        self.test_images = []
        self.test_labels = {"shape": [], "color": [], "size": []}
        self.activations_fc1 = []
        self.activations_fc2 = []
        self.causal_importance_fc1 = []
        self.causal_importance_fc2 = []
        self.pre_sigmoid_ci_fc1 = []
        self.pre_sigmoid_ci_fc2 = []

        with torch.no_grad():
            for idx in indices:
                image, labels = self.test_dataset[idx]
                image = image.unsqueeze(0).to(self.device)

                # Store image and labels
                self.test_images.append(image.squeeze().cpu().numpy())
                for attr in ["shape", "color", "size"]:
                    self.test_labels[attr].append(labels[attr].item())

                # Get conv features
                conv_features = self._get_conv_features(image)

                # Get activations for fc1
                if "fc1" in self.component_model.components:
                    component_fc1 = self.component_model.components["fc1"]
                    acts_fc1 = component_fc1.get_component_acts(conv_features)
                    self.activations_fc1.append(acts_fc1[0].cpu().numpy())

                # Get activations for fc2
                if "fc2" in self.component_model.components:
                    hidden = F.relu(self.mlp_model.fc1(conv_features))
                    component_fc2 = self.component_model.components["fc2"]
                    acts_fc2 = component_fc2.get_component_acts(hidden)
                    self.activations_fc2.append(acts_fc2[0].cpu().numpy())

                # Get causal importance
                pre_weight_acts = {}
                if "fc1" in self.component_model.components:
                    pre_weight_acts["fc1"] = conv_features
                if "fc2" in self.component_model.components:
                    hidden = F.relu(self.mlp_model.fc1(conv_features))
                    pre_weight_acts["fc2"] = hidden

                ci_outputs = self.component_model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    sampling="continuous",
                    detach_inputs=False,
                )

                if "fc1" in ci_outputs.lower_leaky:
                    self.causal_importance_fc1.append(
                        ci_outputs.lower_leaky["fc1"][0].cpu().numpy()
                    )
                if "fc2" in ci_outputs.lower_leaky:
                    self.causal_importance_fc2.append(
                        ci_outputs.lower_leaky["fc2"][0].cpu().numpy()
                    )
                if "fc1" in ci_outputs.pre_sigmoid:
                    self.pre_sigmoid_ci_fc1.append(ci_outputs.pre_sigmoid["fc1"][0].cpu().numpy())
                if "fc2" in ci_outputs.pre_sigmoid:
                    self.pre_sigmoid_ci_fc2.append(ci_outputs.pre_sigmoid["fc2"][0].cpu().numpy())

        self.test_images = np.array(self.test_images)
        for attr in ["shape", "color", "size"]:
            self.test_labels[attr] = np.array(self.test_labels[attr])

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

    def get_activation_distribution(self, layer: str, component_idx: int):
        """Plot the distribution of activation values for this component."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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

        positive_acts = acts[acts > 0]
        if len(positive_acts) > 0:
            ax2.hist(np.log10(positive_acts + 1e-10), bins=50, color="green", alpha=0.7)
            ax2.set_xlabel("Log10(Inner Activation Value)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Log-scale Distribution (positive)")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No positive inner activations", ha="center", va="center")

        plt.tight_layout()
        return fig

    def get_causal_importance_distribution(self, layer: str, component_idx: int):
        """Plot the distribution of causal importance values."""
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0 or component_idx >= ci_values.shape[1]:
            return None

        ci = ci_values[:, component_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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

        sorted_ci = np.sort(ci)
        cdf = np.arange(1, len(sorted_ci) + 1) / len(sorted_ci)
        ax2.plot(sorted_ci, cdf, color="purple", linewidth=2)
        ax2.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Median")
        ax2.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90th percentile")
        ax2.set_xlabel("Causal Importance")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("CDF")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def get_top_activating_examples(self, layer: str, component_idx: int, n_examples: int = 16):
        """Show examples with highest activation for this component."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]
        top_indices = np.argsort(acts)[-n_examples:][::-1]

        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            img = self.test_images[idx].transpose(1, 2, 0)  # CHW -> HWC
            axes[i].imshow(img)
            shape = SHAPE_NAMES[self.test_labels["shape"][idx]]
            color = COLOR_NAMES[self.test_labels["color"][idx]]
            size = SIZE_NAMES[self.test_labels["size"][idx]]
            axes[i].set_title(f"{size} {color} {shape}\nAct: {acts[idx]:.3f}", fontsize=8)
            axes[i].axis("off")

        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        plt.suptitle(
            f"{layer} - Component {component_idx}\nTop {n_examples} by Inner Activation",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def get_top_ci_examples(self, layer: str, component_idx: int, n_examples: int = 16):
        """Show examples with highest causal importance."""
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
            img = self.test_images[idx].transpose(1, 2, 0)
            axes[i].imshow(img)
            shape = SHAPE_NAMES[self.test_labels["shape"][idx]]
            color = COLOR_NAMES[self.test_labels["color"][idx]]
            size = SIZE_NAMES[self.test_labels["size"][idx]]
            axes[i].set_title(f"{size} {color} {shape}\nCI: {ci[idx]:.3f}", fontsize=8)
            axes[i].axis("off")

        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        plt.suptitle(
            f"{layer} - Component {component_idx}\nTop {n_examples} by Causal Importance",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def get_attribute_activation_analysis(self, layer: str, component_idx: int, attribute: str):
        """Analyze component activations by attribute value."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]
        labels = self.test_labels[attribute]

        if attribute == "shape":
            names = SHAPE_NAMES
            n_values = 3
            color = "coral"
        elif attribute == "color":
            names = COLOR_NAMES
            n_values = 3
            color = "forestgreen"
        else:  # size
            names = SIZE_NAMES
            n_values = 2
            color = "steelblue"

        # Group by attribute value
        class_activations = {i: [] for i in range(n_values)}
        for act, label in zip(acts, labels, strict=False):
            class_activations[label].append(act)

        class_means = {i: np.mean(vals) for i, vals in class_activations.items()}
        class_stds = {i: np.std(vals) for i, vals in class_activations.items()}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Bar plot
        values = list(range(n_values))
        means = [class_means[v] for v in values]
        stds = [class_stds[v] for v in values]

        ax1.bar(values, means, yerr=stds, capsize=5, color=color, alpha=0.7)
        ax1.set_xlabel(attribute.capitalize())
        ax1.set_ylabel("Mean Inner Activation")
        ax1.set_title(
            f"{layer} - Component {component_idx}\nMean Activation by {attribute.capitalize()}"
        )
        ax1.set_xticks(values)
        ax1.set_xticklabels(names)
        ax1.grid(axis="y", alpha=0.3)

        # Box plot
        box_data = [class_activations[v] for v in values]
        bp = ax2.boxplot(box_data, labels=names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax2.set_xlabel(attribute.capitalize())
        ax2.set_ylabel("Inner Activation")
        ax2.set_title(f"Distribution by {attribute.capitalize()}")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def get_attribute_ci_analysis(self, layer: str, component_idx: int, attribute: str):
        """Analyze causal importance by attribute value."""
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0 or component_idx >= ci_values.shape[1]:
            return None

        ci = ci_values[:, component_idx]
        labels = self.test_labels[attribute]

        if attribute == "shape":
            names = SHAPE_NAMES
            n_values = 3
            color = "coral"
        elif attribute == "color":
            names = COLOR_NAMES
            n_values = 3
            color = "forestgreen"
        else:
            names = SIZE_NAMES
            n_values = 2
            color = "steelblue"

        class_ci = {i: [] for i in range(n_values)}
        for ci_val, label in zip(ci, labels, strict=False):
            class_ci[label].append(ci_val)

        class_means = {i: np.mean(vals) for i, vals in class_ci.items()}
        class_stds = {i: np.std(vals) for i, vals in class_ci.items()}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        values = list(range(n_values))
        means = [class_means[v] for v in values]
        stds = [class_stds[v] for v in values]

        ax1.bar(values, means, yerr=stds, capsize=5, color=color, alpha=0.7)
        ax1.set_xlabel(attribute.capitalize())
        ax1.set_ylabel("Mean Causal Importance")
        ax1.set_title(f"{layer} - Component {component_idx}\nMean CI by {attribute.capitalize()}")
        ax1.set_xticks(values)
        ax1.set_xticklabels(names)
        ax1.grid(axis="y", alpha=0.3)

        box_data = [class_ci[v] for v in values]
        bp = ax2.boxplot(box_data, labels=names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax2.set_xlabel(attribute.capitalize())
        ax2.set_ylabel("Causal Importance")
        ax2.set_title(f"CI Distribution by {attribute.capitalize()}")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def get_all_attributes_summary(self, layer: str, component_idx: int):
        """Summary plot showing activation patterns across all attributes."""
        activations = self.activations_fc1 if layer == "fc1" else self.activations_fc2
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2

        if len(activations) == 0 or component_idx >= activations.shape[1]:
            return None

        acts = activations[:, component_idx]
        ci = ci_values[:, component_idx]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        attrs = [("shape", SHAPE_NAMES, 3), ("color", COLOR_NAMES, 3), ("size", SIZE_NAMES, 2)]
        colors = ["coral", "forestgreen", "steelblue"]

        for col, (attr, names, n_vals) in enumerate(attrs):
            labels = self.test_labels[attr]

            # Activation means
            act_means = [np.mean(acts[labels == i]) for i in range(n_vals)]
            axes[0, col].bar(range(n_vals), act_means, color=colors[col], alpha=0.7)
            axes[0, col].set_xticks(range(n_vals))
            axes[0, col].set_xticklabels(names, rotation=45)
            axes[0, col].set_ylabel("Mean Activation")
            axes[0, col].set_title(f"Activation by {attr.capitalize()}")
            axes[0, col].grid(axis="y", alpha=0.3)

            # CI means
            ci_means = [np.mean(ci[labels == i]) for i in range(n_vals)]
            axes[1, col].bar(range(n_vals), ci_means, color=colors[col], alpha=0.7)
            axes[1, col].set_xticks(range(n_vals))
            axes[1, col].set_xticklabels(names, rotation=45)
            axes[1, col].set_ylabel("Mean CI")
            axes[1, col].set_title(f"CI by {attr.capitalize()}")
            axes[1, col].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"{layer} - Component {component_idx}\nAttribute Analysis Summary", fontsize=14
        )
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

        # Basic statistics
        sparsity = (acts == 0).mean()
        ci_sparsity = (ci < 0.01).mean()

        metrics_text = f"""
**Component {component_idx} - {layer}**

**Inner Activation Statistics:**
- Mean: {acts.mean():.4f}
- Max: {acts.max():.4f}
- Std: {acts.std():.4f}
- Sparsity (% zeros): {sparsity * 100:.2f}%

**Causal Importance Statistics:**
- Mean CI: {ci.mean():.4f}
- Median CI: {np.median(ci):.4f}
- Max CI: {ci.max():.4f}
- 90th Percentile: {np.percentile(ci, 90):.4f}
- Sparsity (CI < 0.01): {ci_sparsity * 100:.2f}%

**Selectivity by Attribute:**
"""

        for attr, names, n_vals in [
            ("shape", SHAPE_NAMES, 3),
            ("color", COLOR_NAMES, 3),
            ("size", SIZE_NAMES, 2),
        ]:
            labels = self.test_labels[attr]
            ci_means = np.array([np.mean(ci[labels == i]) for i in range(n_vals)])

            max_idx = ci_means.argmax()
            selectivity = ci_means[max_idx] / (ci_means.mean() + 1e-10)

            metrics_text += f"\n*{attr.capitalize()}:*\n"
            for i, name in enumerate(names):
                marker = " **←**" if i == max_idx else ""
                metrics_text += f"  - {name}: {ci_means[i]:.4f}{marker}\n"
            metrics_text += f"  - Selectivity Index: {selectivity:.3f}\n"

        return metrics_text

    def get_component_attribute_heatmap(self, layer: str):
        """Heatmap showing which components are selective for which attributes."""
        ci_values = self.causal_importance_fc1 if layer == "fc1" else self.causal_importance_fc2
        if len(ci_values) == 0:
            return None

        n_components = ci_values.shape[1]

        # Compute selectivity for each attribute
        selectivity_matrix = np.zeros((8, n_components))  # 3 shapes + 3 colors + 2 sizes

        row = 0
        for attr, _names, n_vals in [
            ("shape", SHAPE_NAMES, 3),
            ("color", COLOR_NAMES, 3),
            ("size", SIZE_NAMES, 2),
        ]:
            labels = self.test_labels[attr]
            for val in range(n_vals):
                for comp_idx in range(n_components):
                    ci = ci_values[:, comp_idx]
                    selectivity_matrix[row, comp_idx] = np.mean(ci[labels == val])
                row += 1

        fig, ax = plt.subplots(figsize=(min(20, n_components // 5 + 4), 6))

        im = ax.imshow(selectivity_matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Component Index")
        ax.set_ylabel("Attribute Value")

        y_labels = SHAPE_NAMES + COLOR_NAMES + SIZE_NAMES
        ax.set_yticks(range(8))
        ax.set_yticklabels(y_labels)

        # Add horizontal lines to separate attribute types
        ax.axhline(2.5, color="white", linewidth=2)
        ax.axhline(5.5, color="white", linewidth=2)

        plt.colorbar(im, ax=ax, label="Mean CI")
        ax.set_title(f"{layer} - Component Selectivity Heatmap\n(Mean CI by attribute value)")

        plt.tight_layout()
        return fig


def create_dashboard(experiment_dir: str, checkpoint_step: int | None = None):
    """Create the Gradio dashboard interface."""
    dashboard = ShapesComponentDashboard(
        experiment_dir=experiment_dir,
        checkpoint_step=checkpoint_step,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_samples=1000,
    )

    available_layers = list(dashboard.component_model.components.keys())
    max_components = {
        layer: dashboard.component_model.components[layer].V.shape[1] for layer in available_layers
    }

    def get_alive_components(layer: str, threshold: float = 0.01) -> list[int]:
        ci_values = (
            dashboard.causal_importance_fc1 if layer == "fc1" else dashboard.causal_importance_fc2
        )
        if len(ci_values) == 0:
            return list(range(max_components[layer]))

        alive = []
        for comp_idx in range(ci_values.shape[1]):
            ci = ci_values[:, comp_idx]
            if (ci > threshold).mean() > 0.005:
                alive.append(comp_idx)
        return alive

    alive_components = {layer: get_alive_components(layer) for layer in available_layers}

    for layer in available_layers:
        total = max_components[layer]
        alive = len(alive_components[layer])
        print(f"{layer}: {alive}/{total} components alive ({100 * alive / total:.1f}%)")

    def update_component_viz(layer, component_idx, attribute):
        if component_idx >= max_components.get(layer, 0):
            component_idx = 0

        dist_plot = dashboard.get_activation_distribution(layer, component_idx)
        ci_dist_plot = dashboard.get_causal_importance_distribution(layer, component_idx)
        top_examples = dashboard.get_top_activating_examples(layer, component_idx)
        top_ci_examples = dashboard.get_top_ci_examples(layer, component_idx)
        attr_act_plot = dashboard.get_attribute_activation_analysis(layer, component_idx, attribute)
        attr_ci_plot = dashboard.get_attribute_ci_analysis(layer, component_idx, attribute)
        summary_plot = dashboard.get_all_attributes_summary(layer, component_idx)
        metrics = dashboard.get_selectivity_metrics(layer, component_idx)

        return (
            dist_plot,
            ci_dist_plot,
            top_examples,
            top_ci_examples,
            attr_act_plot,
            attr_ci_plot,
            summary_plot,
            metrics,
        )

    def update_heatmap(layer):
        return dashboard.get_component_attribute_heatmap(layer)

    with gr.Blocks(title="Shapes Component Dashboard") as demo:
        gr.Markdown("# Shapes/Colors/Sizes Component Inspector Dashboard")
        gr.Markdown(
            f"Exploring components from: `{experiment_dir}`\n\n"
            f"**Ground truth:** 8 atomic mechanisms (3 shapes + 3 colors + 2 sizes)\n\n"
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
            attribute_dropdown = gr.Dropdown(
                choices=["shape", "color", "size"],
                value="shape",
                label="Attribute for Analysis",
            )
            hide_dead_toggle = gr.Checkbox(
                value=True,
                label="Hide dead components",
            )

        alive_stats = gr.Markdown(
            f"**{available_layers[0]}**: {len(alive_components[available_layers[0]])}/{max_components[available_layers[0]]} "
            f"components alive"
        )

        def update_component_dropdown(layer, hide_dead):
            if hide_dead:
                choices = [(f"Component {i}", i) for i in alive_components[layer]]
            else:
                choices = [(f"Component {i}", i) for i in range(max_components[layer])]

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

        gr.Markdown("## Component Selectivity Heatmap")
        heatmap_plot = gr.Plot(label="Selectivity Heatmap")
        layer_dropdown.change(update_heatmap, inputs=[layer_dropdown], outputs=[heatmap_plot])

        with gr.Row():
            with gr.Column():
                summary_plot = gr.Plot(label="All Attributes Summary")
            with gr.Column():
                metrics_text = gr.Markdown(label="Selectivity Metrics")

        gr.Markdown("## Causal Importance Analysis")
        with gr.Row():
            ci_dist_plot = gr.Plot(label="CI Distribution")
            attr_ci_plot = gr.Plot(label="CI by Selected Attribute")

        with gr.Row():
            top_ci_examples = gr.Plot(label="Top CI Examples")

        gr.Markdown("## Inner Activation Analysis")
        with gr.Row():
            dist_plot = gr.Plot(label="Activation Distribution")
            attr_act_plot = gr.Plot(label="Activation by Selected Attribute")

        with gr.Row():
            top_examples = gr.Plot(label="Top Activating Examples")

        # Update on component/attribute change
        for trigger in [component_dropdown, attribute_dropdown]:
            trigger.change(
                update_component_viz,
                inputs=[layer_dropdown, component_dropdown, attribute_dropdown],
                outputs=[
                    dist_plot,
                    ci_dist_plot,
                    top_examples,
                    top_ci_examples,
                    attr_act_plot,
                    attr_ci_plot,
                    summary_plot,
                    metrics_text,
                ],
            )

        # Initial load
        demo.load(
            update_component_viz,
            inputs=[layer_dropdown, component_dropdown, attribute_dropdown],
            outputs=[
                dist_plot,
                ci_dist_plot,
                top_examples,
                top_ci_examples,
                attr_act_plot,
                attr_ci_plot,
                summary_plot,
                metrics_text,
            ],
        )
        demo.load(update_heatmap, inputs=[layer_dropdown], outputs=[heatmap_plot])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Shapes Component Dashboard")
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
        server_name="0.0.0.0",
    )

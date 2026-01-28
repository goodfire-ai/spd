"""Metric for visualizing component directions during training."""

import io
from typing import Any, ClassVar, override

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel


def _render_figure(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


class ComponentDirections(Metric):
    """Visualize the most active component directions as images.

    For fc1 layer, V columns represent directions in input space (784-dim).
    We visualize the top components by mean CI as 28x28 images.
    """

    metric_section: ClassVar[str] = "figures"
    slow: ClassVar[bool] = True

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        layer_name: str = "fc1",
        n_components: int = 20,
        image_shape: tuple[int, int] = (28, 28),
    ) -> None:
        self.model = model
        self.device = device
        self.layer_name = layer_name
        self.n_components = n_components
        self.image_shape = image_shape

        # Accumulate CI values to find most active components
        self.ci_sum: Tensor | None = None
        self.n_samples = 0

    @override
    def update(
        self,
        *,
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        if self.layer_name not in ci.lower_leaky:
            return

        layer_ci = ci.lower_leaky[self.layer_name]
        # Average over batch and any sequence dimensions
        mean_ci = layer_ci.mean(dim=tuple(range(layer_ci.dim() - 1)))

        if self.ci_sum is None:
            self.ci_sum = mean_ci.detach()
        else:
            self.ci_sum = self.ci_sum + mean_ci.detach()
        self.n_samples += 1

    @override
    def compute(self) -> dict[str, Image.Image]:
        if self.ci_sum is None or self.n_samples == 0:
            return {}

        if self.layer_name not in self.model.components:
            return {}

        components = self.model.components[self.layer_name]
        V = components.V.detach().cpu()  # Shape: (d_in, C)

        # Get mean CI and find top components
        mean_ci = self.ci_sum / self.n_samples
        top_indices = torch.argsort(mean_ci, descending=True)[: self.n_components]
        top_ci_values = mean_ci[top_indices].cpu().numpy()

        # Create visualization
        n_cols = 5
        n_rows = (self.n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, (comp_idx, ci_val) in enumerate(zip(top_indices, top_ci_values, strict=False)):
            if i >= len(axes):
                break

            # Get component direction and reshape to image
            direction = V[:, comp_idx].numpy()

            # Check if dimensions match expected image shape
            expected_size = self.image_shape[0] * self.image_shape[1]
            if direction.shape[0] == expected_size:
                direction = direction.reshape(self.image_shape)
            else:
                # Can't reshape, skip visualization
                axes[i].text(
                    0.5,
                    0.5,
                    f"C{comp_idx.item()}\nCI={ci_val:.3f}\n(wrong dim)",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                axes[i].axis("off")
                continue

            # Normalize for visualization
            vmax = max(abs(direction.min()), abs(direction.max()))
            if vmax > 1e-6:
                direction = direction / vmax

            axes[i].imshow(direction, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[i].set_title(f"C{comp_idx.item()} (CI={ci_val:.3f})", fontsize=8)
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"Top {self.n_components} Active Components ({self.layer_name})", fontsize=10)
        plt.tight_layout()

        # Convert to PIL Image
        img = _render_figure(fig)
        plt.close(fig)

        return {f"component_directions_{self.layer_name}": img}

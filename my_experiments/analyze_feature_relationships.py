"""Analyze relationships between input (fc1) and output (fc2) component activations.

For each output feature, finds the two most predictive input features and creates
heatmap visualizations showing the relationship.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.models.component_model import ComponentModel
from spd.utils.module_utils import ModulePathInfo


class TwoLayerMLP(nn.Module):
    """A simple 2-layer MLP for MNIST classification."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_component_model(output_dir: Path, device: str) -> ComponentModel:
    """Load the trained component model."""
    # Load the target MLP
    mlp_path = output_dir / "trained_mlp.pth"
    model = TwoLayerMLP()
    model.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
    model.eval()
    model.requires_grad_(False)

    # Find the latest checkpoint
    checkpoints = list(output_dir.glob("model_*.pth"))
    assert checkpoints, f"No checkpoints found in {output_dir}"
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    print(f"Loading checkpoint: {latest_checkpoint}")

    # Determine C from checkpoint
    checkpoint_weights = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
    # Find C from the V matrix shape
    v_key = [k for k in checkpoint_weights if ".V" in k][0]
    C = checkpoint_weights[v_key].shape[1]
    print(f"Number of components: {C}")

    # Create component model
    module_path_info = [
        ModulePathInfo(module_path="fc1", C=C),
        ModulePathInfo(module_path="fc2", C=C),
    ]

    comp_model = ComponentModel(
        target_model=model,
        module_path_info=module_path_info,
        ci_fn_type="linear",
        ci_fn_hidden_dims=[],
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr=None,
    )

    comp_model.load_state_dict(checkpoint_weights)
    comp_model.to(device)
    comp_model.eval()

    return comp_model


def collect_component_activations(
    comp_model: ComponentModel,
    data_loader: DataLoader,
    device: str,
    max_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect fc1 and fc2 component activations on the dataset."""
    fc1_acts_list = []
    fc2_acts_list = []
    n_samples = 0

    fc1_components = comp_model.components["fc1"]
    fc2_components = comp_model.components["fc2"]

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Collecting activations"):
            if n_samples >= max_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # Flatten images
            x = images.view(batch_size, -1)

            # Get fc1 component activations: x @ V1
            fc1_acts = fc1_components.get_component_acts(x)  # (batch, C)

            # Compute hidden activations: ReLU(x @ V1 @ U1)
            hidden = torch.relu(fc1_acts @ fc1_components.U)
            if fc1_components.bias is not None:
                hidden = hidden + fc1_components.bias

            # Get fc2 component activations: hidden @ V2
            fc2_acts = fc2_components.get_component_acts(hidden)  # (batch, C)

            fc1_acts_list.append(fc1_acts.cpu().numpy())
            fc2_acts_list.append(fc2_acts.cpu().numpy())
            n_samples += batch_size

    fc1_acts = np.concatenate(fc1_acts_list, axis=0)[:max_samples]
    fc2_acts = np.concatenate(fc2_acts_list, axis=0)[:max_samples]

    print(f"Collected {fc1_acts.shape[0]} samples")
    print(f"fc1_acts shape: {fc1_acts.shape}, fc2_acts shape: {fc2_acts.shape}")

    return fc1_acts, fc2_acts


def find_alive_components(
    acts: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Find components that are 'alive' (have meaningful activity).

    Args:
        acts: Component activations of shape (N, C)
        threshold: Minimum mean absolute activation to be considered alive

    Returns:
        Boolean array of shape (C,) indicating which components are alive
    """
    mean_abs_act = np.abs(acts).mean(axis=0)
    return mean_abs_act > threshold


def find_top_predictors(
    fc1_acts: np.ndarray,
    fc2_acts: np.ndarray,
    n_output_components: int = 20,
    n_top_predictors: int = 2,
    alive_threshold: float = 0.1,
) -> tuple[list[tuple[int, list[int], float]], np.ndarray, np.ndarray]:
    """Find the top predictive input features for each output feature using Lasso.

    Only considers alive components (those with mean abs activation > threshold).

    Returns:
        results: list of (output_idx, [input_idx1, input_idx2, ...], r2_score)
        alive_inputs: boolean array of alive input components
        alive_outputs: boolean array of alive output components
    """
    # Find alive components
    alive_inputs = find_alive_components(fc1_acts, alive_threshold)
    alive_outputs = find_alive_components(fc2_acts, alive_threshold)

    n_alive_inputs = alive_inputs.sum()
    n_alive_outputs = alive_outputs.sum()
    print(f"Alive input components: {n_alive_inputs} / {fc1_acts.shape[1]}")
    print(f"Alive output components: {n_alive_outputs} / {fc2_acts.shape[1]}")

    # Get indices of alive components
    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    # Filter to only alive input components for prediction
    fc1_acts_alive = fc1_acts[:, alive_inputs]

    # Find the most active output components among alive ones
    output_activity = np.abs(fc2_acts[:, alive_outputs]).mean(axis=0)
    top_alive_output_order = np.argsort(output_activity)[-n_output_components:][::-1]
    top_output_indices = alive_output_indices[top_alive_output_order]

    results = []

    for out_idx in tqdm(top_output_indices, desc="Finding predictors"):
        y = fc2_acts[:, out_idx]

        # Skip if output has no variance
        if np.std(y) < 1e-6:
            continue

        # Use LassoCV to find sparse predictors among alive inputs only
        lasso = LassoCV(cv=5, max_iter=2000, n_jobs=-1)
        lasso.fit(fc1_acts_alive, y)

        # Get coefficients and find top predictors by absolute value
        coefs = np.abs(lasso.coef_)
        top_alive_input_order = np.argsort(coefs)[-n_top_predictors:][::-1]
        # Map back to original indices
        top_input_indices = alive_input_indices[top_alive_input_order].tolist()

        # Compute R² score
        y_pred = lasso.predict(fc1_acts_alive)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results.append((out_idx, top_input_indices, r2))

    return results, alive_inputs, alive_outputs


def create_heatmap(
    fc1_acts: np.ndarray,
    fc2_acts: np.ndarray,
    out_idx: int,
    in_idx1: int,
    in_idx2: int,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D heatmap of average output activation given two input activations."""
    x1 = fc1_acts[:, in_idx1]
    x2 = fc1_acts[:, in_idx2]
    y = fc2_acts[:, out_idx]

    # Create bins
    x1_bins = np.linspace(np.percentile(x1, 2), np.percentile(x1, 98), n_bins + 1)
    x2_bins = np.linspace(np.percentile(x2, 2), np.percentile(x2, 98), n_bins + 1)

    # Compute average y in each bin
    heatmap = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    x1_digitized = np.digitize(x1, x1_bins) - 1
    x2_digitized = np.digitize(x2, x2_bins) - 1

    for i in range(len(y)):
        i1 = min(max(x1_digitized[i], 0), n_bins - 1)
        i2 = min(max(x2_digitized[i], 0), n_bins - 1)
        heatmap[i2, i1] += y[i]  # Note: i2 is row (y-axis), i1 is col (x-axis)
        counts[i2, i1] += 1

    # Average
    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap = np.where(counts > 0, heatmap / counts, np.nan)

    return heatmap, x1_bins, x2_bins


def plot_relationships(
    fc1_acts: np.ndarray,
    fc2_acts: np.ndarray,
    results: list[tuple[int, list[int], float]],
    output_dir: Path,
    n_plots: int = 16,
):
    """Create heatmap plots for the top relationships."""
    n_plots = min(n_plots, len(results))

    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    # First pass: compute all heatmaps to find global min/max
    heatmaps = []
    for i, (out_idx, in_indices, r2) in enumerate(results[:n_plots]):  # noqa: B007
        in_idx1, in_idx2 = in_indices[0], in_indices[1]
        heatmap, x1_bins, x2_bins = create_heatmap(fc1_acts, fc2_acts, out_idx, in_idx1, in_idx2)
        heatmaps.append((heatmap, x1_bins, x2_bins))

    # Find global scale (symmetric around 0)
    all_values = np.concatenate([h[0][~np.isnan(h[0])].flatten() for h in heatmaps])
    vmax = np.percentile(np.abs(all_values), 98)  # Use 98th percentile to avoid outliers
    vmin = -vmax

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    for i, (out_idx, in_indices, r2) in enumerate(results[:n_plots]):  # noqa: B007
        ax = axes[i]

        in_idx1, in_idx2 = in_indices[0], in_indices[1]
        heatmap, x1_bins, x2_bins = heatmaps[i]

        # Plot heatmap with shared scale, zero = white
        im = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            extent=[x1_bins[0], x1_bins[-1], x2_bins[0], x2_bins[-1]],
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel(f"Input C{in_idx1}", fontsize=9)
        ax.set_ylabel(f"Input C{in_idx2}", fontsize=9)
        ax.set_title(f"Output C{out_idx}\nR²={r2:.3f}", fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Output Feature vs Top 2 Predictive Input Features", fontsize=12, y=1.02)
    plt.tight_layout()

    plot_path = output_dir / "feature_relationship_heatmaps.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap plot to {plot_path}")

    # Also create a summary plot showing R² scores
    fig, ax = plt.subplots(figsize=(10, 4))

    out_indices = [r[0] for r in results]
    r2_scores = [r[2] for r in results]

    ax.bar(range(len(results)), r2_scores, color="steelblue")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"C{idx}" for idx in out_indices], rotation=45, ha="right")
    ax.set_xlabel("Output Component")
    ax.set_ylabel("R² Score (top 2 predictors)")
    ax.set_title("How well can output components be predicted from top 2 input components?")
    ax.grid(axis="y", alpha=0.3)

    summary_path = output_dir / "feature_predictability_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {summary_path}")


def plot_global_connectivity(
    fc1_acts: np.ndarray,
    fc2_acts: np.ndarray,
    alive_inputs: np.ndarray,
    alive_outputs: np.ndarray,
    output_dir: Path,
):
    """Create a global connectivity map showing which inputs predict which outputs."""
    from sklearn.linear_model import Lasso

    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    n_alive_in = len(alive_input_indices)
    n_alive_out = len(alive_output_indices)

    # Filter to alive components
    fc1_alive = fc1_acts[:, alive_inputs]
    fc2_acts[:, alive_outputs]

    # Build connectivity matrix using Lasso coefficients
    # Rows = output components, Cols = input components
    connectivity = np.zeros((n_alive_out, n_alive_in))

    print(f"Computing connectivity matrix ({n_alive_out} outputs x {n_alive_in} inputs)...")
    for i, out_idx in enumerate(tqdm(alive_output_indices, desc="Fitting models")):
        y = fc2_acts[:, out_idx]
        if np.std(y) < 1e-6:
            continue

        # Fit Lasso to get sparse coefficients (higher alpha = sparser)
        lasso = Lasso(alpha=0.5, max_iter=2000)
        lasso.fit(fc1_alive, y)
        connectivity[i, :] = lasso.coef_

    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, n_alive_in * 0.4), max(6, n_alive_out * 0.4)))

    # Use symmetric colormap centered at 0
    vmax = np.percentile(np.abs(connectivity), 98)
    vmin = -vmax

    im = ax.imshow(
        connectivity,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    # Labels
    ax.set_xticks(range(n_alive_in))
    ax.set_xticklabels([f"C{idx}" for idx in alive_input_indices], rotation=90, fontsize=8)
    ax.set_yticks(range(n_alive_out))
    ax.set_yticklabels([f"C{idx}" for idx in alive_output_indices], fontsize=8)

    ax.set_xlabel("Input Components (fc1)", fontsize=11)
    ax.set_ylabel("Output Components (fc2)", fontsize=11)
    ax.set_title("Input → Output Feature Connectivity\n(Lasso coefficients, white=0)", fontsize=12)

    plt.colorbar(im, ax=ax, label="Coefficient")
    plt.tight_layout()

    connectivity_path = output_dir / "feature_connectivity_matrix.png"
    plt.savefig(connectivity_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved connectivity matrix to {connectivity_path}")

    # Also create a graph-style visualization for the strongest connections
    plot_connectivity_graph(
        connectivity,
        alive_input_indices,
        alive_output_indices,
        output_dir,
        top_k=2,  # Top 2 inputs per output
    )

    return connectivity, alive_input_indices, alive_output_indices


def plot_connectivity_graph(
    connectivity: np.ndarray,
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    output_dir: Path,
    top_k: int = 3,
):
    """Create a bipartite graph visualization of top connections."""
    n_out, n_in = connectivity.shape

    fig, ax = plt.subplots(figsize=(14, 8))

    # Position nodes
    in_y = np.linspace(0, 1, n_in)
    out_y = np.linspace(0, 1, n_out)

    in_x = 0
    out_x = 1

    # Draw connections (top_k per output)
    max_coef = np.abs(connectivity).max()
    for i in range(n_out):
        coefs = connectivity[i, :]
        top_inputs = np.argsort(np.abs(coefs))[-top_k:][::-1]

        for j in top_inputs:
            coef = coefs[j]
            if np.abs(coef) < 0.01:  # Skip very weak connections
                continue

            # Line width and color based on coefficient
            width = 0.5 + 3 * np.abs(coef) / max_coef
            color = "red" if coef > 0 else "blue"
            alpha = 0.3 + 0.7 * np.abs(coef) / max_coef

            ax.plot([in_x, out_x], [in_y[j], out_y[i]], color=color, linewidth=width, alpha=alpha)

    # Draw input nodes
    for j in range(n_in):
        ax.scatter(in_x, in_y[j], s=100, c="steelblue", zorder=5)
        ax.text(in_x - 0.05, in_y[j], f"C{input_indices[j]}", ha="right", va="center", fontsize=8)

    # Draw output nodes
    for i in range(n_out):
        ax.scatter(out_x, out_y[i], s=100, c="coral", zorder=5)
        ax.text(out_x + 0.05, out_y[i], f"C{output_indices[i]}", ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        in_x, -0.05, "Input Features\n(fc1)", ha="center", va="top", fontsize=11, fontweight="bold"
    )
    ax.text(
        out_x,
        -0.05,
        "Output Features\n(fc2)",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"Feature Connectivity Graph (top {top_k} inputs per output)\nRed=positive, Blue=negative",
        fontsize=12,
    )

    graph_path = output_dir / "feature_connectivity_graph.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved connectivity graph to {graph_path}")


def stepwise_forward_regression(
    fc1_acts: np.ndarray,
    fc2_acts: np.ndarray,
    alive_inputs: np.ndarray,
    alive_outputs: np.ndarray,
    output_dir: Path,
    r2_threshold: float = 0.9,
    max_features: int = 10,
):
    """Stepwise forward regression with linear models.

    For each output feature, greedily add input features until R² >= threshold.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    n_alive_in = len(alive_input_indices)
    n_alive_out = len(alive_output_indices)

    fc1_alive = fc1_acts[:, alive_inputs]

    print(f"\nStepwise forward regression with linear models (target R² = {r2_threshold})...")
    print(f"Analyzing {n_alive_out} output components with {n_alive_in} candidate inputs\n")

    results = []  # List of (out_idx, selected_input_indices, r2_values_at_each_step)

    for out_global_idx in tqdm(alive_output_indices, desc="Stepwise regression"):
        y = fc2_acts[:, out_global_idx]

        if np.std(y) < 1e-6:
            continue

        selected_local = []  # Local indices into alive inputs
        selected_global = []  # Global component indices
        r2_history = []
        remaining = set(range(n_alive_in))

        current_r2 = 0.0

        while current_r2 < r2_threshold and len(selected_local) < max_features and remaining:
            best_feature = None
            best_r2 = current_r2

            # Try adding each remaining feature
            for feat_local in remaining:
                trial_features = selected_local + [feat_local]
                X_trial = fc1_alive[:, trial_features]

                # Use linear regression
                model = LinearRegression()
                # Use cross-validation to get robust R²
                scores = cross_val_score(model, X_trial, y, cv=3, scoring="r2")
                mean_r2 = scores.mean()

                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_feature = feat_local

            if best_feature is None:
                break

            selected_local.append(best_feature)
            selected_global.append(alive_input_indices[best_feature])
            remaining.remove(best_feature)
            current_r2 = best_r2
            r2_history.append(current_r2)

        results.append((out_global_idx, selected_global, r2_history))

    # Print results
    print("\n" + "=" * 70)
    print(f"Stepwise Forward Regression Results (target R² = {r2_threshold})")
    print("=" * 70)

    n_features_needed = []
    for out_idx, selected, r2_hist in results:  # noqa: B007
        n_feat = len(selected)
        final_r2 = r2_hist[-1] if r2_hist else 0
        features_str = ", ".join([f"C{idx}" for idx in selected[:5]])
        if len(selected) > 5:
            features_str += f", ... ({len(selected)} total)"
        print(f"Output C{out_idx:3d}: {n_feat} features → R²={final_r2:.3f}  [{features_str}]")
        n_features_needed.append(n_feat)

    print(f"\nAverage features needed: {np.mean(n_features_needed):.1f}")
    print(f"Median features needed: {np.median(n_features_needed):.1f}")

    # Plot results
    plot_stepwise_results(results, output_dir, r2_threshold)

    return results


def plot_stepwise_results(
    results: list[tuple[int, list[int], list[float]]],
    output_dir: Path,
    r2_threshold: float,
):
    """Plot stepwise regression results."""
    n_outputs = len(results)

    # Plot 1: Number of features needed per output
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of features needed
    ax = axes[0]
    out_indices = [r[0] for r in results]
    n_features = [len(r[1]) for r in results]
    final_r2 = [r[2][-1] if r[2] else 0 for r in results]

    colors = ["green" if r2 >= r2_threshold else "orange" for r2 in final_r2]
    ax.bar(range(n_outputs), n_features, color=colors)
    ax.set_xticks(range(n_outputs))
    ax.set_xticklabels([f"C{idx}" for idx in out_indices], rotation=45, ha="right")
    ax.set_xlabel("Output Component")
    ax.set_ylabel("# Input Features Needed")
    ax.set_title(
        f"Features needed to reach R² ≥ {r2_threshold}\n(green=achieved, orange=not reached)"
    )
    ax.axhline(
        y=np.mean(n_features), color="red", linestyle="--", label=f"Mean={np.mean(n_features):.1f}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: R² curves for each output
    ax = axes[1]
    for out_idx, selected, r2_hist in results:  # noqa: B007
        if r2_hist:
            ax.plot(range(1, len(r2_hist) + 1), r2_hist, "o-", label=f"C{out_idx}", alpha=0.7)

    ax.axhline(y=r2_threshold, color="red", linestyle="--", label=f"Target R²={r2_threshold}")
    ax.set_xlabel("Number of Input Features")
    ax.set_ylabel("R² Score (CV)")
    ax.set_title("R² vs Number of Features (Stepwise Forward Selection)")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    stepwise_path = output_dir / "stepwise_regression_results.png"
    plt.savefig(stepwise_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved stepwise regression plot to {stepwise_path}")

    # Plot 3: Which input features are most commonly selected
    all_selected = []
    for _, selected, _ in results:
        all_selected.extend(selected)

    if all_selected:
        from collections import Counter

        counts = Counter(all_selected)

        fig, ax = plt.subplots(figsize=(12, 5))
        sorted_inputs = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        sorted_counts = [counts[x] for x in sorted_inputs]

        ax.bar(range(len(sorted_inputs)), sorted_counts, color="steelblue")
        ax.set_xticks(range(len(sorted_inputs)))
        ax.set_xticklabels([f"C{idx}" for idx in sorted_inputs], rotation=45, ha="right")
        ax.set_xlabel("Input Component")
        ax.set_ylabel("# Outputs it predicts")
        ax.set_title("Most Predictive Input Features (across all outputs)")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        importance_path = output_dir / "input_feature_importance.png"
        plt.savefig(importance_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved input importance plot to {importance_path}")


def collect_ci_scores(
    comp_model: ComponentModel,
    data_loader: DataLoader,
    device: str,
    max_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect CI scores for fc1 and fc2 components on the dataset."""
    fc1_ci_list = []
    fc2_ci_list = []
    n_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Collecting CI scores"):
            if n_samples >= max_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)
            x = images.view(batch_size, -1)

            # Run forward pass with caching to get pre_weight_acts
            output_with_cache = comp_model(x, cache_type="input")

            # Get CI scores
            ci_outputs = comp_model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                sampling="continuous",
                detach_inputs=True,
            )

            # Extract CI for fc1 and fc2 (using lower_leaky which is the standard CI)
            fc1_ci = ci_outputs.lower_leaky["fc1"].cpu().numpy()  # (batch, C)
            fc2_ci = ci_outputs.lower_leaky["fc2"].cpu().numpy()  # (batch, C)

            fc1_ci_list.append(fc1_ci)
            fc2_ci_list.append(fc2_ci)
            n_samples += batch_size

    fc1_ci = np.concatenate(fc1_ci_list, axis=0)[:max_samples]
    fc2_ci = np.concatenate(fc2_ci_list, axis=0)[:max_samples]

    print(f"Collected CI scores for {fc1_ci.shape[0]} samples")
    print(f"fc1_ci shape: {fc1_ci.shape}, fc2_ci shape: {fc2_ci.shape}")

    return fc1_ci, fc2_ci


def compute_causal_effects(
    comp_model: ComponentModel,
    data_loader: DataLoader,
    device: str,
    alive_inputs: np.ndarray,
    alive_outputs: np.ndarray,
    output_dir: Path,
    ci_threshold: float = 0.1,
    max_samples: int = 5000,
):
    """Compute causal effects of input components on output components.

    For each (i, j) pair:
    - Filter samples where CI(j) > ci_threshold
    - Compute E[output_j | do(mechanism_i = 1)] / E[output_j | do(mechanism_i = 0)]

    This measures the causal effect of "turning on" vs "turning off" each input component.
    """
    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    n_in = len(alive_input_indices)
    n_out = len(alive_output_indices)

    fc1_components = comp_model.components["fc1"]
    fc2_components = comp_model.components["fc2"]

    print(f"Computing causal effects for {n_in} input x {n_out} output components...")

    # First, collect all the data we need
    all_x = []
    all_fc2_ci = []
    n_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Collecting samples"):
            if n_samples >= max_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)
            x = images.view(batch_size, -1)

            # Get CI scores for fc2
            output_with_cache = comp_model(x, cache_type="input")
            ci_outputs = comp_model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                sampling="continuous",
                detach_inputs=True,
            )
            fc2_ci = ci_outputs.lower_leaky["fc2"]  # (batch, C)

            all_x.append(x.cpu())
            all_fc2_ci.append(fc2_ci.cpu())
            n_samples += batch_size

    all_x = torch.cat(all_x, dim=0)[:max_samples]
    all_fc2_ci = torch.cat(all_fc2_ci, dim=0)[:max_samples]

    print(f"Collected {all_x.shape[0]} samples")

    # Now compute causal effects for each (i, j) pair
    # causal_effect[i, j] = E[output_j | do(i=1)] / E[output_j | do(i=0)]
    causal_effect_matrix = np.zeros((n_in, n_out))
    sample_counts = np.zeros((n_in, n_out))  # How many samples used for each pair

    V1 = fc1_components.V  # (784, C1)
    U1 = fc1_components.U  # (C1, 128)
    V2 = fc2_components.V  # (128, C2)
    bias1 = fc1_components.bias  # (128,) or None

    with torch.no_grad():
        for j_local, j_global in enumerate(tqdm(alive_output_indices, desc="Output components")):
            # Filter samples where CI(j) > threshold
            ci_j = all_fc2_ci[:, j_global]
            valid_mask = ci_j > ci_threshold
            n_valid = valid_mask.sum().item()

            if n_valid < 10:
                print(f"  Skipping output C{j_global}: only {n_valid} valid samples")
                continue

            x_valid = all_x[valid_mask].to(device)

            for i_local, i_global in enumerate(alive_input_indices):
                # Compute fc1 component activations: x @ V1
                fc1_acts = x_valid @ V1  # (n_valid, C1)

                # Create masks for intervention
                # mask_on: component i is ON (=1), others normal
                # mask_off: component i is OFF (=0), others normal
                C1 = fc1_acts.shape[1]

                # Intervention: set component i to 1 (on)
                torch.ones(C1, device=device)
                # The component activation when "on" is just the normal activation
                # We compute: hidden = ReLU((fc1_acts * mask) @ U1)
                # With mask_on, component i contributes normally

                # Intervention: set component i to 0 (off) - ablate it
                mask_off = torch.ones(C1, device=device)
                mask_off[i_global] = 0.0

                # Compute hidden activations for both interventions
                hidden_on = torch.relu(fc1_acts @ U1)
                if bias1 is not None:
                    hidden_on = hidden_on + bias1

                hidden_off = torch.relu((fc1_acts * mask_off) @ U1)
                if bias1 is not None:
                    hidden_off = hidden_off + bias1

                # Compute fc2 component j activations
                # fc2_acts = hidden @ V2[:, j]
                fc2_j_on = hidden_on @ V2[:, j_global]  # (n_valid,)
                fc2_j_off = hidden_off @ V2[:, j_global]  # (n_valid,)

                # Compute both ratio and absolute difference
                mean_on = fc2_j_on.mean().item()
                mean_off = fc2_j_off.mean().item()

                # Absolute difference: E[j | do(i=1)] - E[j | do(i=0)]
                causal_effect = mean_on - mean_off

                causal_effect_matrix[i_local, j_local] = causal_effect
                sample_counts[i_local, j_local] = n_valid

    # Plot the causal effect matrix (absolute difference)
    fig, ax = plt.subplots(figsize=(max(10, n_out * 0.8), max(8, n_in * 0.3)))

    # Use full range, symmetric around 0
    vmax = max(abs(causal_effect_matrix.min()), abs(causal_effect_matrix.max()))
    vmin = -vmax

    im = ax.imshow(
        causal_effect_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(n_out))
    ax.set_xticklabels([f"C{idx}" for idx in alive_output_indices], rotation=90, fontsize=8)
    ax.set_yticks(range(n_in))
    ax.set_yticklabels([f"C{idx}" for idx in alive_input_indices], fontsize=8)

    ax.set_xlabel("Output Components (fc2)", fontsize=11)
    ax.set_ylabel("Input Components (fc1)", fontsize=11)
    ax.set_title(
        f"Causal Effect: E[output_j | input_i ON] - E[output_j | input_i OFF]\n"
        f"(filtered to samples where CI(j) > {ci_threshold})",
        fontsize=11,
    )

    plt.colorbar(im, ax=ax, label="Absolute Change")
    plt.tight_layout()

    causal_path = output_dir / "causal_effect_matrix.png"
    plt.savefig(causal_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal effect matrix to {causal_path}")

    # Create sparse version: keep effects until X% of total per output
    sparse_matrix = np.zeros_like(causal_effect_matrix)
    sparsity_threshold = 0.8

    print(
        f"\nSparsifying causal matrix (keeping {sparsity_threshold * 100:.0f}% of effect per output):"
    )
    for j in range(n_out):
        col = causal_effect_matrix[:, j]
        abs_col = np.abs(col)
        total = abs_col.sum()

        if total < 1e-6:
            continue

        # Sort by absolute value descending
        sorted_indices = np.argsort(abs_col)[::-1]
        cumsum = 0
        n_kept = 0

        for idx in sorted_indices:
            cumsum += abs_col[idx]
            sparse_matrix[idx, j] = col[idx]
            n_kept += 1
            if cumsum >= sparsity_threshold * total:
                break

        out_comp = alive_output_indices[j]
        print(f"  Output C{out_comp}: kept {n_kept}/{n_in} inputs ({n_kept / n_in * 100:.0f}%)")

    # Plot the sparse causal effect matrix
    fig, ax = plt.subplots(figsize=(max(10, n_out * 0.8), max(8, n_in * 0.3)))

    # Use full range, symmetric around 0
    vmax = max(abs(sparse_matrix.min()), abs(sparse_matrix.max()))
    vmin = -vmax

    im = ax.imshow(
        sparse_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(n_out))
    ax.set_xticklabels([f"C{idx}" for idx in alive_output_indices], rotation=90, fontsize=8)
    ax.set_yticks(range(n_in))
    ax.set_yticklabels([f"C{idx}" for idx in alive_input_indices], fontsize=8)

    ax.set_xlabel("Output Components (fc2)", fontsize=11)
    ax.set_ylabel("Input Components (fc1)", fontsize=11)
    ax.set_title(
        f"Sparse Causal Effects (top inputs covering {sparsity_threshold * 100:.0f}% effect per output)\n"
        f"E[output_j | input_i ON] - E[output_j | input_i OFF]",
        fontsize=11,
    )

    plt.colorbar(im, ax=ax, label="Absolute Change")
    plt.tight_layout()

    sparse_path = output_dir / "causal_effect_matrix_sparse.png"
    plt.savefig(sparse_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved sparse causal effect matrix to {sparse_path}")

    # Count total non-zero entries
    n_nonzero = (sparse_matrix != 0).sum()
    total_entries = sparse_matrix.size
    print(
        f"Sparsity: {n_nonzero}/{total_entries} entries ({n_nonzero / total_entries * 100:.1f}% non-zero)"
    )

    # Create graph visualization with minimized crossings
    plot_causal_graph(
        sparse_matrix,
        alive_input_indices,
        alive_output_indices,
        output_dir,
    )

    # Print top causal effects
    print("\nTop 10 excitatory effects (input ON increases output):")
    flat_effects = causal_effect_matrix.flatten()
    sorted_indices = np.argsort(flat_effects)[::-1]  # Most positive first

    for k in range(min(10, len(sorted_indices))):
        idx = sorted_indices[k]
        i, j = np.unravel_index(idx, causal_effect_matrix.shape)
        in_comp = alive_input_indices[i]
        out_comp = alive_output_indices[j]
        effect = causal_effect_matrix[i, j]
        n_samp = int(sample_counts[i, j])
        print(f"  C{in_comp} → C{out_comp}: {effect:+.4f} (n={n_samp})")

    print("\nTop 10 inhibitory effects (input ON decreases output):")
    for k in range(min(10, len(sorted_indices))):
        idx = sorted_indices[-(k + 1)]
        i, j = np.unravel_index(idx, causal_effect_matrix.shape)
        in_comp = alive_input_indices[i]
        out_comp = alive_output_indices[j]
        effect = causal_effect_matrix[i, j]
        n_samp = int(sample_counts[i, j])
        print(f"  C{in_comp} → C{out_comp}: {effect:+.4f} (n={n_samp})")

    return causal_effect_matrix, alive_input_indices, alive_output_indices


def plot_causal_graph(
    sparse_matrix: np.ndarray,
    input_indices: np.ndarray,
    output_indices: np.ndarray,
    output_dir: Path,
):
    """Create a bipartite graph visualization with minimized edge crossings.

    Uses barycenter heuristic with many iterations to order nodes and reduce crossings.
    """
    n_in, n_out = sparse_matrix.shape

    # Get edges (non-zero entries)
    edges = []
    for i in range(n_in):
        for j in range(n_out):
            if sparse_matrix[i, j] != 0:
                edges.append((i, j, sparse_matrix[i, j]))

    if not edges:
        print("No edges to plot")
        return

    # Filter to only inputs that have connections
    connected_input_set = set(i for i, j, _ in edges)
    connected_inputs_initial = sorted(connected_input_set)

    # Barycenter heuristic to minimize crossings
    # Start with outputs ordered by index
    output_order = list(range(n_out))
    input_order = connected_inputs_initial.copy()

    # Try multiple random restarts with barycenter/median refinement
    best_crossings = float("inf")
    best_input_order = input_order.copy()
    best_output_order = output_order.copy()

    for restart in range(50):
        # Random initial order for restarts > 0
        if restart > 0:
            import random

            input_order = connected_inputs_initial.copy()
            random.shuffle(input_order)
            output_order = list(range(n_out))
            random.shuffle(output_order)

        for _iteration in range(10):
            # Create position mapping for outputs
            output_pos = {idx: pos for pos, idx in enumerate(output_order)}

            # Reorder inputs by median position of connected outputs
            _output_pos = output_pos
            _output_order = output_order

            def input_median(i, _op=_output_pos, _oo=_output_order):  # noqa: B023
                positions = [_op[j] for j in range(n_out) if sparse_matrix[i, j] != 0]
                if not positions:
                    return len(_oo) / 2
                positions.sort()
                mid = len(positions) // 2
                if len(positions) % 2 == 0:
                    return (positions[mid - 1] + positions[mid]) / 2
                return positions[mid]

            input_order = sorted(input_order, key=input_median)

            # Create position mapping for inputs
            input_pos = {idx: pos for pos, idx in enumerate(input_order)}

            # Reorder outputs by median position of connected inputs
            _input_pos = input_pos
            _input_order = input_order

            def output_median(j, _ip=_input_pos, _io=_input_order):  # noqa: B023
                positions = [_ip[i] for i in connected_input_set if sparse_matrix[i, j] != 0]
                if not positions:
                    return len(_io) / 2
                positions.sort()
                mid = len(positions) // 2
                if len(positions) % 2 == 0:
                    return (positions[mid - 1] + positions[mid]) / 2
                return positions[mid]

            output_order = sorted(output_order, key=output_median)

        # Count crossings with current order
        in_y_temp = {idx: pos for pos, idx in enumerate(input_order)}
        out_y_temp = {idx: pos for pos, idx in enumerate(output_order)}
        crossings = count_crossings(edges, in_y_temp, out_y_temp)

        if crossings < best_crossings:
            best_crossings = crossings
            best_input_order = input_order.copy()
            best_output_order = output_order.copy()

    input_order = best_input_order
    output_order = best_output_order

    # Local optimization: try swapping adjacent pairs
    improved = True
    while improved:
        improved = False

        # Try swapping adjacent inputs
        for i in range(len(input_order) - 1):
            # Try swap
            input_order[i], input_order[i + 1] = input_order[i + 1], input_order[i]
            in_y_temp = {idx: pos for pos, idx in enumerate(input_order)}
            out_y_temp = {idx: pos for pos, idx in enumerate(output_order)}
            new_crossings = count_crossings(edges, in_y_temp, out_y_temp)

            if new_crossings < best_crossings:
                best_crossings = new_crossings
                improved = True
            else:
                # Swap back
                input_order[i], input_order[i + 1] = input_order[i + 1], input_order[i]

        # Try swapping adjacent outputs
        for j in range(len(output_order) - 1):
            output_order[j], output_order[j + 1] = output_order[j + 1], output_order[j]
            in_y_temp = {idx: pos for pos, idx in enumerate(input_order)}
            out_y_temp = {idx: pos for pos, idx in enumerate(output_order)}
            new_crossings = count_crossings(edges, in_y_temp, out_y_temp)

            if new_crossings < best_crossings:
                best_crossings = new_crossings
                improved = True
            else:
                output_order[j], output_order[j + 1] = output_order[j + 1], output_order[j]

    connected_inputs = input_order
    print(f"Best crossing count after optimization: {best_crossings}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, max(10, len(connected_inputs) * 0.4)))

    # Position nodes
    in_x = 0
    out_x = 1

    # Y positions based on order
    in_y = {
        idx: pos / (len(connected_inputs) - 1) if len(connected_inputs) > 1 else 0.5
        for pos, idx in enumerate(connected_inputs)
    }
    out_y = {idx: pos / (n_out - 1) if n_out > 1 else 0.5 for pos, idx in enumerate(output_order)}

    # Find max absolute effect for scaling
    max_effect = max(abs(e[2]) for e in edges)

    # Draw edges
    for i, j, effect in edges:
        if i not in in_y:
            continue

        # Line properties based on effect
        width = 0.5 + 3 * abs(effect) / max_effect
        color = "#d62728" if effect > 0 else "#1f77b4"  # Red for excitatory, blue for inhibitory
        alpha = 0.4 + 0.5 * abs(effect) / max_effect

        ax.plot(
            [in_x, out_x], [in_y[i], out_y[j]], color=color, linewidth=width, alpha=alpha, zorder=1
        )

    # Draw input nodes
    for idx in connected_inputs:
        ax.scatter(in_x, in_y[idx], s=150, c="#2ca02c", zorder=5, edgecolors="white", linewidths=1)
        ax.text(
            in_x - 0.03,
            in_y[idx],
            f"C{input_indices[idx]}",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Draw output nodes
    for idx in output_order:
        ax.scatter(
            out_x, out_y[idx], s=150, c="#ff7f0e", zorder=5, edgecolors="white", linewidths=1
        )
        ax.text(
            out_x + 0.03,
            out_y[idx],
            f"C{output_indices[idx]}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Labels
    ax.text(in_x, -0.02, "Input (fc1)", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(out_x, -0.02, "Output (fc2)", ha="center", va="top", fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#d62728", linewidth=2, label="Excitatory (+)"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Inhibitory (-)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    ax.set_title("Causal Effect Graph (80% threshold, minimized crossings)", fontsize=12, pad=20)

    plt.tight_layout()

    graph_path = output_dir / "causal_graph_sparse.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal graph to {graph_path}")

    # Count crossings
    n_crossings = count_crossings(edges, in_y, out_y)
    print(f"Number of edge crossings: {n_crossings}")


def count_crossings(edges, in_y, out_y):
    """Count the number of edge crossings in the graph."""
    crossings = 0
    edge_list = [(i, j) for i, j, _ in edges if i in in_y]

    for idx1, (i1, j1) in enumerate(edge_list):
        for i2, j2 in edge_list[idx1 + 1 :]:
            # Two edges cross if their endpoints are "interleaved"
            # i.e., (in_y[i1] - in_y[i2]) * (out_y[j1] - out_y[j2]) < 0
            if (in_y[i1] - in_y[i2]) * (out_y[j1] - out_y[j2]) < 0:
                crossings += 1

    return crossings


def plot_weight_path_matrix(
    comp_model: ComponentModel,
    alive_inputs: np.ndarray,
    alive_outputs: np.ndarray,
    output_dir: Path,
):
    """Plot the direct weight path matrix U1[i, :] @ V2[:, j].

    This shows the structural connectivity from input component i to output component j,
    ignoring the ReLU nonlinearity. Positive values mean the input component excites
    the output component; negative means inhibition.
    """
    # Get the component weight matrices
    fc1_components = comp_model.components["fc1"]
    fc2_components = comp_model.components["fc2"]

    U1 = fc1_components.U.detach().cpu().numpy()  # Shape: (C1, hidden_size)
    V2 = fc2_components.V.detach().cpu().numpy()  # Shape: (hidden_size, C2)

    print(f"U1 shape: {U1.shape}, V2 shape: {V2.shape}")

    # Compute the full weight path matrix: (C1, C2)
    weight_path_full = U1 @ V2

    # Filter to alive components
    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    weight_path = weight_path_full[np.ix_(alive_input_indices, alive_output_indices)]

    n_in = len(alive_input_indices)
    n_out = len(alive_output_indices)

    print(f"Weight path matrix shape (alive only): {weight_path.shape}")

    # Plot the matrix
    fig, ax = plt.subplots(figsize=(max(10, n_out * 0.5), max(8, n_in * 0.3)))

    # Use symmetric colormap centered at 0
    vmax = np.percentile(np.abs(weight_path), 98)
    vmin = -vmax

    im = ax.imshow(
        weight_path,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(n_out))
    ax.set_xticklabels([f"C{idx}" for idx in alive_output_indices], rotation=90, fontsize=8)
    ax.set_yticks(range(n_in))
    ax.set_yticklabels([f"C{idx}" for idx in alive_input_indices], fontsize=8)

    ax.set_xlabel("Output Components (fc2)", fontsize=11)
    ax.set_ylabel("Input Components (fc1)", fontsize=11)
    ax.set_title(
        "Weight Path Matrix: U1[i, :] @ V2[:, j]\n(Direct structural connectivity, ignoring ReLU)",
        fontsize=12,
    )

    plt.colorbar(im, ax=ax, label="Connection Strength")
    plt.tight_layout()

    weight_path_path = output_dir / "weight_path_matrix.png"
    plt.savefig(weight_path_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved weight path matrix to {weight_path_path}")

    # Also print some statistics
    print("\nWeight path matrix statistics:")
    print(f"  Range: [{weight_path.min():.4f}, {weight_path.max():.4f}]")
    print(f"  Mean abs: {np.abs(weight_path).mean():.4f}")
    print(f"  Sparsity (|w| < 0.01): {(np.abs(weight_path) < 0.01).mean() * 100:.1f}%")

    # Find strongest connections
    flat_indices = np.argsort(np.abs(weight_path).flatten())[::-1]
    print("\nTop 10 strongest weight connections:")
    for k in range(min(10, len(flat_indices))):
        idx = flat_indices[k]
        i, j = np.unravel_index(idx, weight_path.shape)
        in_comp = alive_input_indices[i]
        out_comp = alive_output_indices[j]
        val = weight_path[i, j]
        sign = "+" if val > 0 else "-"
        print(f"  C{in_comp} → C{out_comp}: {sign}{abs(val):.4f}")

    return weight_path, alive_input_indices, alive_output_indices


def plot_ci_correlation_matrix(
    fc1_ci: np.ndarray,
    fc2_ci: np.ndarray,
    alive_inputs: np.ndarray,
    alive_outputs: np.ndarray,
    output_dir: Path,
):
    """Plot correlation matrix between fc1 and fc2 CI scores."""
    alive_input_indices = np.where(alive_inputs)[0]
    alive_output_indices = np.where(alive_outputs)[0]

    # Filter to alive components
    fc1_ci_alive = fc1_ci[:, alive_inputs]
    fc2_ci_alive = fc2_ci[:, alive_outputs]

    n_in = len(alive_input_indices)
    n_out = len(alive_output_indices)

    # Compute correlation matrix
    # corr[i, j] = correlation between fc2_ci[:, i] and fc1_ci[:, j]
    corr_matrix = np.zeros((n_out, n_in))

    for i in range(n_out):
        for j in range(n_in):
            corr = np.corrcoef(fc2_ci_alive[:, i], fc1_ci_alive[:, j])[0, 1]
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, n_in * 0.4), max(6, n_out * 0.5)))

    im = ax.imshow(
        corr_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(n_in))
    ax.set_xticklabels([f"C{idx}" for idx in alive_input_indices], rotation=90, fontsize=8)
    ax.set_yticks(range(n_out))
    ax.set_yticklabels([f"C{idx}" for idx in alive_output_indices], fontsize=8)

    ax.set_xlabel("Input Components (fc1) - CI", fontsize=11)
    ax.set_ylabel("Output Components (fc2) - CI", fontsize=11)
    ax.set_title(
        "CI Score Correlations: Input ↔ Output Components\n(white=0, red=positive, blue=negative)",
        fontsize=12,
    )

    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()

    corr_path = output_dir / "ci_correlation_matrix.png"
    plt.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CI correlation matrix to {corr_path}")

    return corr_matrix


def main(
    output_dir: str = "output/mnist_experiment_v2",
    n_output_components: int = 20,
    max_samples: int = 10000,
):
    """Analyze feature relationships in a trained SPD model.

    Args:
        output_dir: Directory containing the trained model
        n_output_components: Number of top output components to analyze
        max_samples: Maximum number of samples to use for analysis
    """
    output_path = Path(output_dir)
    assert output_path.exists(), f"Output directory not found: {output_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    comp_model = load_component_model(output_path, device)

    # Load test data
    print("Loading MNIST test data...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Collect activations
    print("Collecting component activations...")
    fc1_acts, fc2_acts = collect_component_activations(
        comp_model, test_loader, device, max_samples=max_samples
    )

    # Find top predictors for each output component
    print(f"Finding top predictors for {n_output_components} output components...")
    results, alive_inputs, alive_outputs = find_top_predictors(
        fc1_acts,
        fc2_acts,
        n_output_components=n_output_components,
        n_top_predictors=2,
        alive_threshold=0.1,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results: Top 2 predictive input features for each output feature")
    print("=" * 60)
    for out_idx, in_indices, r2 in results:
        print(
            f"Output C{out_idx:3d}: Input C{in_indices[0]:3d}, C{in_indices[1]:3d}  (R²={r2:.3f})"
        )

    # Create visualizations
    print("\nCreating visualizations...")
    plot_relationships(fc1_acts, fc2_acts, results, output_path, n_plots=16)

    # Create global connectivity map
    print("\nCreating global connectivity map...")
    plot_global_connectivity(
        fc1_acts,
        fc2_acts,
        alive_inputs,
        alive_outputs,
        output_path,
    )

    # Stepwise forward regression with decision trees
    stepwise_forward_regression(
        fc1_acts,
        fc2_acts,
        alive_inputs,
        alive_outputs,
        output_path,
        r2_threshold=0.9,
        max_features=10,
    )

    # Compute weight path matrix U1 @ V2
    print("\nComputing weight path matrix (U1 @ V2)...")
    plot_weight_path_matrix(
        comp_model,
        alive_inputs,
        alive_outputs,
        output_path,
    )

    # Compute causal effects via intervention
    print("\nComputing causal effects (intervention analysis)...")
    compute_causal_effects(
        comp_model,
        test_loader,
        device,
        alive_inputs,
        alive_outputs,
        output_path,
        ci_threshold=0.1,
        max_samples=5000,
    )

    # Collect CI scores and plot correlation matrix
    print("\nCollecting CI scores...")
    fc1_ci, fc2_ci = collect_ci_scores(comp_model, test_loader, device, max_samples=max_samples)

    # Find alive components based on CI scores
    alive_inputs_ci = np.abs(fc1_ci).mean(axis=0) > 0.1
    alive_outputs_ci = np.abs(fc2_ci).mean(axis=0) > 0.1

    print(f"Alive input components (CI): {alive_inputs_ci.sum()} / {fc1_ci.shape[1]}")
    print(f"Alive output components (CI): {alive_outputs_ci.sum()} / {fc2_ci.shape[1]}")

    print("\nPlotting CI correlation matrix...")
    plot_ci_correlation_matrix(
        fc1_ci,
        fc2_ci,
        alive_inputs_ci,
        alive_outputs_ci,
        output_path,
    )

    print("\nDone!")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

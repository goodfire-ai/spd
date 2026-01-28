"""
Symbolic Regression for Component Activations

This script uses symbolic regression to discover interpretable relationships
between alive components in fc1 (input layer) and fc2 (output layer) of the
MNIST experiment v2.

The goal is to predict the activation of an alive fc2 component based on
the activations of alive fc1 components, yielding a human-readable formula.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysr import PySRRegressor
from torchvision import datasets, transforms

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


def load_component_model(out_dir: Path, device: str) -> tuple[ComponentModel, nn.Module]:
    """Load the trained MLP and its component model."""
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
    """Find alive components in fc1 and fc2 based on firing frequency."""
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

            # Get fc1 activations
            fc1_acts = fc1_components.get_component_acts(images_flat)
            ci_fc1 = component_model.ci_fns["fc1"](fc1_acts)
            fire_count_fc1 += (ci_fc1 > ci_threshold).float().sum(dim=0)

            # Get fc2 activations (need hidden layer output)
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


def collect_component_activations(
    component_model: ComponentModel,
    test_dataset: datasets.MNIST,
    device: str,
    alive_fc1: np.ndarray,
    alive_fc2: np.ndarray,
    n_samples: int = 10000,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect component activations for alive components."""
    fc1_components = component_model.components["fc1"]
    fc2_components = component_model.components["fc2"]

    all_fc1_acts = []
    all_fc2_acts = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in range(n_batches):
            indices = torch.randint(0, len(test_dataset), (batch_size,))
            images = torch.stack([test_dataset[int(i)][0] for i in indices]).to(device)
            images_flat = images.view(batch_size, -1)

            # Get fc1 component activations
            fc1_acts = fc1_components.get_component_acts(images_flat)
            fc1_acts_alive = fc1_acts[:, alive_fc1].cpu().numpy()
            all_fc1_acts.append(fc1_acts_alive)

            # Get fc2 component activations
            hidden = F.relu(component_model.target_model.fc1(images_flat))
            fc2_acts = fc2_components.get_component_acts(hidden)
            fc2_acts_alive = fc2_acts[:, alive_fc2].cpu().numpy()
            all_fc2_acts.append(fc2_acts_alive)

    X = np.vstack(all_fc1_acts)[:n_samples]
    Y = np.vstack(all_fc2_acts)[:n_samples]

    return X, Y


def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str,
    niterations: int = 40,
    populations: int = 15,
    maxsize: int = 20,
) -> PySRRegressor:
    """Run PySR symbolic regression."""
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=33,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "square", "sqrt", "exp", "log"],
        constraints={
            "exp": 5,  # Limit complexity inside exp
            "log": 5,
            "sqrt": 5,
        },
        nested_constraints={
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
        },
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        progress=True,
        verbosity=1,
        random_state=42,
        parallelism="multithreading",  # Use multithreading for speed
        temp_equation_file=True,
    )

    print(f"\nRunning symbolic regression for {target_name}...")
    print(f"Input features: {feature_names}")
    print(f"Training samples: {len(y)}")

    model.fit(X, y, variable_names=feature_names)

    return model


def plot_results(
    model: PySRRegressor,
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    output_path: Path,
) -> None:
    """Plot regression results."""
    y_pred = model.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot: predicted vs actual
    axes[0].scatter(y, y_pred, alpha=0.3, s=10)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect fit")
    axes[0].set_xlabel("Actual activation", fontsize=12)
    axes[0].set_ylabel("Predicted activation", fontsize=12)
    axes[0].set_title(f"Symbolic Regression: {target_name}", fontsize=14)
    axes[0].legend()

    # Compute R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    axes[0].text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    # Residual plot
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted activation", fontsize=12)
    axes[1].set_ylabel("Residual", fontsize=12)
    axes[1].set_title("Residuals", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main(
    output_dir: str = "output/mnist_experiment_v2",
    target_component_idx: int | None = None,
    n_samples: int = 10000,
    niterations: int = 40,
    seed: int = 42,
) -> None:
    """
    Run symbolic regression to predict fc2 component activations from fc1.

    Args:
        output_dir: Directory containing the trained model
        target_component_idx: Index of the alive fc2 component to predict (None = random)
        n_samples: Number of samples to collect for training
        niterations: Number of PySR iterations
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)

    print("Loading component model...")
    component_model, _ = load_component_model(out_dir, device)

    print("Loading MNIST test dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    print("Finding alive components...")
    alive_fc1, alive_fc2 = find_alive_components(component_model, test_dataset, device)
    print(f"Found {len(alive_fc1)} alive fc1 components: {alive_fc1}")
    print(f"Found {len(alive_fc2)} alive fc2 components: {alive_fc2}")

    # Select target component
    if target_component_idx is None:
        target_idx_in_alive = np.random.randint(0, len(alive_fc2))
    else:
        assert target_component_idx in alive_fc2, f"Component {target_component_idx} is not alive"
        target_idx_in_alive = np.where(alive_fc2 == target_component_idx)[0][0]

    target_component = alive_fc2[target_idx_in_alive]
    print(f"\nTarget: fc2 component C{target_component}")

    print(f"\nCollecting {n_samples} activation samples...")
    X, Y = collect_component_activations(
        component_model, test_dataset, device, alive_fc1, alive_fc2, n_samples=n_samples
    )

    # Extract target column
    y = Y[:, target_idx_in_alive]

    # Create feature names
    feature_names = [f"fc1_C{c}" for c in alive_fc1]

    # Run symbolic regression
    sr_model = run_symbolic_regression(
        X,
        y,
        feature_names,
        target_name=f"fc2_C{target_component}",
        niterations=niterations,
    )

    # Print results
    print("\n" + "=" * 80)
    print("SYMBOLIC REGRESSION RESULTS")
    print("=" * 80)
    print(f"\nTarget: fc2 component C{target_component}")
    print("\nBest equation found:")
    print(f"  {sr_model.sympy()}")
    print(f"\nEquation complexity: {sr_model.get_best()['complexity']}")
    print(f"Loss (MSE): {sr_model.get_best()['loss']:.6f}")

    print("\n\nAll equations found (Pareto front):")
    print(sr_model)

    # Plot results
    plot_path = out_dir / f"symbolic_regression_fc2_C{target_component}.png"
    plot_results(sr_model, X, y, f"fc2_C{target_component}", plot_path)

    # Save equation to file
    eq_path = out_dir / f"symbolic_regression_fc2_C{target_component}.txt"
    with open(eq_path, "w") as f:
        f.write(f"Target: fc2 component C{target_component}\n")
        f.write(f"Alive fc1 components used as features: {list(alive_fc1)}\n\n")
        f.write(f"Best equation:\n{sr_model.sympy()}\n\n")
        f.write(f"Complexity: {sr_model.get_best()['complexity']}\n")
        f.write(f"Loss (MSE): {sr_model.get_best()['loss']:.6f}\n\n")
        f.write("All equations (Pareto front):\n")
        f.write(str(sr_model))
    print(f"\nSaved equation to {eq_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)

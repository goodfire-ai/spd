from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader

from spd.experiments.mnist_sl_mlp.models import MLP
from spd.experiments.mnist_sl_mlp.train_mnist_sl import TrainingMetrics, TrainingResults
from spd.log import logger


def plot_training_metrics(
    teacher_metrics: TrainingMetrics,
    student_metrics: TrainingMetrics,
    save_dir: Path | None = None,
) -> tuple[Figure, Figure]:
    """Create unified plots for training metrics.

    Args:
        teacher_metrics: Teacher training metrics
        student_metrics: Student training metrics
        save_dir: Optional directory to save figures

    Returns:
        Tuple of (loss_figure, accuracy_figure)
    """
    # Create single loss plot with both models
    fig_loss: Figure
    ax_loss: plt.Axes
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

    # Plot all loss curves on same plot
    ax_loss.plot(
        teacher_metrics.steps,
        teacher_metrics.train_losses,
        "b-",
        label="Teacher Train",
        linewidth=2,
        alpha=0.8,
    )
    ax_loss.plot(
        teacher_metrics.steps,
        teacher_metrics.val_losses,
        "b--",
        label="Teacher Val",
        linewidth=2,
        alpha=0.8,
    )
    ax_loss.plot(
        student_metrics.steps,
        student_metrics.train_losses,
        "r-",
        label="Student Train",
        linewidth=2,
        alpha=0.8,
    )
    ax_loss.plot(
        student_metrics.steps,
        student_metrics.val_losses,
        "r--",
        label="Student Val",
        linewidth=2,
        alpha=0.8,
    )

    ax_loss.set_xlabel("Training Steps", fontsize=12)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=12)
    ax_loss.set_title("Training and Validation Loss", fontsize=16)
    ax_loss.legend(fontsize=11)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")  # Log scale

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_loss.savefig(save_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved loss plot to {save_dir / 'loss_curves.png'}")

    # Create accuracy plot
    fig_acc: Figure
    ax_acc: plt.Axes
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))

    # Plot test accuracies over time
    ax_acc.plot(
        teacher_metrics.steps,
        teacher_metrics.test_accuracies,
        "b-",
        label="Teacher",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax_acc.plot(
        student_metrics.steps,
        student_metrics.test_accuracies,
        "r-",
        label="Student",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )

    ax_acc.set_xlabel("Training Steps", fontsize=12)
    ax_acc.set_ylabel("Test Accuracy", fontsize=12)
    ax_acc.set_title("Test Accuracy Over Training", fontsize=16)
    ax_acc.legend(fontsize=12)
    ax_acc.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()

    if save_dir:
        fig_acc.savefig(save_dir / "accuracy_curves.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved accuracy plot to {save_dir / 'accuracy_curves.png'}")

    return fig_loss, fig_acc


def plot_output_distributions(
    teacher: MLP,
    student: MLP,
    mnist_loader: DataLoader,
    noise_loader: DataLoader,
    num_bins: int = 50,
    save_dir: Path | None = None,
) -> tuple[Figure, Figure]:
    """Plot output distributions for digit and auxiliary outputs on MNIST and noise data.

    Args:
        teacher: Trained teacher model
        student: Trained student model
        mnist_loader: DataLoader for MNIST data
        noise_loader: DataLoader for noise data
        save_dir: Optional directory to save figures

    Returns:
        Tuple of (digit_distributions_figure, aux_distributions_figure)
    """
    teacher.eval()
    student.eval()
    device: str = str(next(teacher.parameters()).device)

    # Collect outputs on both datasets
    def collect_outputs(
        loader: DataLoader, tag: str
    ) -> tuple[Float[Tensor, "total 10"], Float[Tensor, "total aux"]]:
        digit_outputs: list[Float[Tensor, "batch 10"]] = []
        aux_outputs: list[Float[Tensor, "batch aux"]] = []

        with torch.no_grad():
            x: Float[Tensor, "batch 1 28 28"]
            _: Int[Tensor, " batch"]
            for x, _ in loader:
                x = x.to(device)

                # Teacher outputs
                teacher_digits: Float[Tensor, "batch 10"]
                teacher_aux: Float[Tensor, "batch aux"]
                teacher_digits, teacher_aux = teacher(x)

                # Student outputs
                student_digits: Float[Tensor, "batch 10"]
                student_aux: Float[Tensor, "batch aux"]
                student_digits, student_aux = student(x)

                # Store both teacher and student outputs
                digit_outputs.extend([teacher_digits.cpu(), student_digits.cpu()])
                aux_outputs.extend([teacher_aux.cpu(), student_aux.cpu()])

        return torch.cat(digit_outputs, dim=0), torch.cat(aux_outputs, dim=0)

    # Collect outputs on MNIST and noise
    mnist_digits: Float[Tensor, "total 10"]
    mnist_aux: Float[Tensor, "total aux"]
    mnist_digits, mnist_aux = collect_outputs(mnist_loader, "MNIST")

    noise_digits: Float[Tensor, "total 10"]
    noise_aux: Float[Tensor, "total aux"]
    noise_digits, noise_aux = collect_outputs(noise_loader, "Noise")

    # Apply softmax for probability interpretation
    mnist_digit_probs: Float[Tensor, "total 10"] = torch.softmax(mnist_digits, dim=1)
    mnist_aux_probs: Float[Tensor, "total aux"] = torch.softmax(mnist_aux, dim=1)
    noise_digit_probs: Float[Tensor, "total 10"] = torch.softmax(noise_digits, dim=1)
    noise_aux_probs: Float[Tensor, "total aux"] = torch.softmax(noise_aux, dim=1)

    # Create digit distributions plot
    fig_digits: Figure
    axes_digits: list[plt.Axes] | plt.Axes
    fig_digits, axes_digits = plt.subplots(2, 5, figsize=(20, 8))
    axes_digits = axes_digits.flatten()

    # Define common bins for all plots
    bin_edges: np.ndarray = np.linspace(0, 1, num_bins + 1)
    bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(10):
        ax: plt.Axes = axes_digits[i]

        # Compute histograms using np.histogram for consistent binning
        teacher_mnist_counts: np.ndarray
        teacher_mnist_counts, _ = np.histogram(
            mnist_digit_probs[: len(mnist_digit_probs) // 2, i].numpy(),
            bins=bin_edges,
            density=True,
        )
        student_mnist_counts: np.ndarray
        student_mnist_counts, _ = np.histogram(
            mnist_digit_probs[len(mnist_digit_probs) // 2 :, i].numpy(),
            bins=bin_edges,
            density=True,
        )
        teacher_noise_counts: np.ndarray
        teacher_noise_counts, _ = np.histogram(
            noise_digit_probs[: len(noise_digit_probs) // 2, i].numpy(),
            bins=bin_edges,
            density=True,
        )
        student_noise_counts: np.ndarray
        student_noise_counts, _ = np.histogram(
            noise_digit_probs[len(noise_digit_probs) // 2 :, i].numpy(),
            bins=bin_edges,
            density=True,
        )

        # Plot as lines
        ax.plot(bin_centers, teacher_mnist_counts, label="Teacher MNIST", linewidth=2)
        ax.plot(bin_centers, student_mnist_counts, label="Student MNIST", linewidth=2)
        ax.plot(bin_centers, teacher_noise_counts, label="Teacher Noise", linewidth=2)
        ax.plot(bin_centers, student_noise_counts, label="Student Noise", linewidth=2)

        ax.set_xlabel(f"Digit {i} Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Digit {i} Output Distribution")
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig_digits.suptitle("Digit Output Distributions", fontsize=16)
    plt.tight_layout()

    # Create auxiliary distributions plot
    num_aux: int = mnist_aux.shape[1]
    fig_aux: Figure
    axes_aux: list[plt.Axes] | plt.Axes
    fig_aux, axes_aux = plt.subplots(1, num_aux, figsize=(5 * num_aux, 5))

    if num_aux == 1:
        axes_aux = [axes_aux]

    for i in range(num_aux):
        ax: plt.Axes = axes_aux[i]

        # Compute histograms using np.histogram for consistent binning
        teacher_mnist_aux_counts: np.ndarray
        teacher_mnist_aux_counts, _ = np.histogram(
            mnist_aux_probs[: len(mnist_aux_probs) // 2, i].numpy(), bins=bin_edges, density=True
        )
        student_mnist_aux_counts: np.ndarray
        student_mnist_aux_counts, _ = np.histogram(
            mnist_aux_probs[len(mnist_aux_probs) // 2 :, i].numpy(), bins=bin_edges, density=True
        )
        teacher_noise_aux_counts: np.ndarray
        teacher_noise_aux_counts, _ = np.histogram(
            noise_aux_probs[: len(noise_aux_probs) // 2, i].numpy(), bins=bin_edges, density=True
        )
        student_noise_aux_counts: np.ndarray
        student_noise_aux_counts, _ = np.histogram(
            noise_aux_probs[len(noise_aux_probs) // 2 :, i].numpy(), bins=bin_edges, density=True
        )

        # Plot as lines
        ax.plot(bin_centers, teacher_mnist_aux_counts, label="Teacher MNIST", linewidth=2)
        ax.plot(bin_centers, student_mnist_aux_counts, label="Student MNIST", linewidth=2)
        ax.plot(bin_centers, teacher_noise_aux_counts, label="Teacher Noise", linewidth=2)
        ax.plot(bin_centers, student_noise_aux_counts, label="Student Noise", linewidth=2)

        ax.set_xlabel(f"Auxiliary Output {i + 1} Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Auxiliary Output {i + 1} Distribution")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig_aux.suptitle("Auxiliary Output Distributions", fontsize=16)
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_digits.savefig(save_dir / "digit_distributions.png", dpi=150, bbox_inches="tight")
        fig_aux.savefig(save_dir / "aux_distributions.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved distribution plots to {save_dir}")

    return fig_digits, fig_aux


def plot_auxiliary_outputs_distribution(
    teacher: MLP,
    student: MLP,
    loader: DataLoader,
    save_path: Path | None = None,
) -> Figure:
    """Compare auxiliary output distributions between teacher and student.

    Args:
        teacher: Trained teacher model
        student: Trained student model
        loader: DataLoader for input data
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    teacher.eval()
    student.eval()
    device: str = str(next(teacher.parameters()).device)

    teacher_aux_outputs: list[Float[Tensor, "batch aux"]] = []
    student_aux_outputs: list[Float[Tensor, "batch aux"]] = []

    with torch.no_grad():
        x: Float[Tensor, "batch 1 28 28"]
        _: Int[Tensor, " batch"]
        for x, _ in loader:
            x = x.to(device)

            _: Float[Tensor, "batch 10"]
            teacher_aux: Float[Tensor, "batch aux"]
            _, teacher_aux = teacher(x)

            student_aux: Float[Tensor, "batch aux"]
            _, student_aux = student(x)

            teacher_aux_outputs.append(teacher_aux.cpu())
            student_aux_outputs.append(student_aux.cpu())

    # Concatenate all outputs
    teacher_aux: Float[Tensor, "total aux"] = torch.cat(teacher_aux_outputs, dim=0)
    student_aux: Float[Tensor, "total aux"] = torch.cat(student_aux_outputs, dim=0)

    # Apply softmax for probability interpretation
    teacher_probs: Float[Tensor, "total aux"] = torch.softmax(teacher_aux, dim=1)
    student_probs: Float[Tensor, "total aux"] = torch.softmax(student_aux, dim=1)

    # Create figure with subplots for each auxiliary output
    num_aux: int = teacher_aux.shape[1]
    fig: Figure
    axes: list[plt.Axes] | plt.Axes
    fig, axes = plt.subplots(1, num_aux, figsize=(5 * num_aux, 5))

    if num_aux == 1:
        axes = [axes]

    ax: plt.Axes
    i: int
    for i, ax in enumerate(axes):
        # Plot histograms
        ax.hist(teacher_probs[:, i].numpy(), bins=50, alpha=0.5, label="Teacher", density=True)
        ax.hist(student_probs[:, i].numpy(), bins=50, alpha=0.5, label="Student", density=True)

        ax.set_xlabel(f"Auxiliary Output {i + 1} Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Auxiliary Output {i + 1} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved auxiliary output distributions to {save_path}")

    return fig


def create_evaluation_report(
    results: TrainingResults,
    mnist_loader: DataLoader,
    noise_loader: DataLoader,
    save_dir: Path,
) -> None:
    """Create a comprehensive evaluation report with visualizations.

    Args:
        results: Training results containing models and metrics
        mnist_loader: DataLoader for MNIST data
        noise_loader: DataLoader for noise data
        save_dir: Directory to save the report
    """
    logger.info(f"Creating evaluation report in {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot training metrics
    fig_loss: Figure
    fig_acc: Figure
    fig_loss, fig_acc = plot_training_metrics(
        results.teacher_metrics, results.student_metrics, save_dir
    )

    # Plot comprehensive output distributions
    fig_digits: Figure
    fig_aux: Figure
    fig_digits, fig_aux = plot_output_distributions(
        results.teacher, results.student, mnist_loader, noise_loader, num_bins=50, save_dir=save_dir
    )

    # Create text summary
    summary_path: Path = save_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("MNIST Subliminal Learning Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Final Teacher Test Accuracy: {results.final_teacher_accuracy:.2%}\n")
        f.write(f"Final Student Test Accuracy: {results.final_student_accuracy:.2%}\n")
        f.write(f"\nTeacher Training Steps: {len(results.teacher_metrics.steps)}\n")
        f.write(f"Student Training Steps: {len(results.student_metrics.steps)}\n")

        if results.teacher_metrics.train_losses:
            f.write(f"\nFinal Teacher Train Loss: {results.teacher_metrics.train_losses[-1]:.4f}\n")
            f.write(f"Final Teacher Val Loss: {results.teacher_metrics.val_losses[-1]:.4f}\n")

        if results.student_metrics.train_losses:
            f.write(f"\nFinal Student Train Loss: {results.student_metrics.train_losses[-1]:.4f}\n")
            f.write(f"Final Student Val Loss: {results.student_metrics.val_losses[-1]:.4f}\n")

    logger.info(f"Evaluation report saved to {save_dir}")

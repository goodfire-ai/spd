from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    """Plot training losses and accuracies."""
    # Loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(5, 3))

    for metrics, label, color in [
        (teacher_metrics, "Teacher", "b"),
        (student_metrics, "Student", "r"),
    ]:
        ax_loss.plot(metrics.steps, metrics.train_losses, f"{color}-", label=f"{label} Train")
        ax_loss.plot(metrics.steps, metrics.val_losses, f"{color}--", label=f"{label} Val")

    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")
    plt.tight_layout()

    # Accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(5, 3))

    for metrics, label, color, marker in [
        (teacher_metrics, "Teacher", "b", "o"),
        (student_metrics, "Student", "r", "s"),
    ]:
        ax_acc.plot(
            metrics.steps,
            metrics.test_accuracies,
            f"{color}-",
            label=label,
            marker=marker,
            markersize=4,
        )

    ax_acc.set_xlabel("Training Steps")
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Test Accuracy Over Training")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_loss.savefig(save_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        fig_acc.savefig(save_dir / "accuracy_curves.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved plots to {save_dir}")

    return fig_loss, fig_acc


def plot_output_distributions(
    teacher: MLP,
    student: MLP,
    mnist_loader: DataLoader,
    noise_loader: DataLoader,
    num_bins: int = 50,
    save_dir: Path | None = None,
) -> tuple[Figure, Figure]:
    """Plot output distributions for digit and auxiliary outputs."""
    teacher.eval()
    student.eval()
    device = next(teacher.parameters()).device

    # Collect outputs
    def collect_outputs(model: MLP, loader: DataLoader) -> tuple[Tensor, Tensor]:
        digit_outputs = []
        aux_outputs = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                digit_outputs.append(model.forward_digits(x).cpu())
                aux_outputs.append(model.forward_aux(x).cpu())

        return torch.cat(digit_outputs), torch.cat(aux_outputs)

    # Get outputs for both models and datasets
    teacher_mnist_digits, teacher_mnist_aux = collect_outputs(teacher, mnist_loader)
    student_mnist_digits, student_mnist_aux = collect_outputs(student, mnist_loader)
    teacher_noise_digits, teacher_noise_aux = collect_outputs(teacher, noise_loader)
    student_noise_digits, student_noise_aux = collect_outputs(student, noise_loader)

    # Plot digit distributions
    fig_digits, axes_digits = plt.subplots(2, 5, figsize=(20, 8))
    axes_digits = axes_digits.flatten()

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(10):
        ax = axes_digits[i]

        # Plot histograms as lines
        for data, label in [
            (teacher_mnist_digits[:, i], "Teacher MNIST"),
            (student_mnist_digits[:, i], "Student MNIST"),
            (teacher_noise_digits[:, i], "Teacher Noise"),
            (student_noise_digits[:, i], "Student Noise"),
        ]:
            counts, _ = np.histogram(data.numpy(), bins=bin_edges, density=True)
            ax.plot(bin_centers, counts, label=label, linewidth=2)

        ax.set_xlabel(f"Digit {i} Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Digit {i} Output Distribution")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig_digits.suptitle("Digit Output Distributions", fontsize=16)
    plt.tight_layout()

    # Plot auxiliary distributions
    num_aux = teacher_mnist_aux.shape[1]
    fig_aux, axes_aux = plt.subplots(1, num_aux, figsize=(5 * num_aux, 5))
    if num_aux == 1:
        axes_aux = [axes_aux]

    for i in range(num_aux):
        ax = axes_aux[i]

        # Plot histograms as lines
        for data, label in [
            (teacher_mnist_aux[:, i], "Teacher MNIST"),
            (student_mnist_aux[:, i], "Student MNIST"),
            (teacher_noise_aux[:, i], "Teacher Noise"),
            (student_noise_aux[:, i], "Student Noise"),
        ]:
            counts, _ = np.histogram(data.numpy(), bins=bin_edges, density=True)
            ax.plot(bin_centers, counts, label=label, linewidth=2)

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


def create_evaluation_report(
    results: TrainingResults,
    mnist_loader: DataLoader,
    noise_loader: DataLoader,
    save_dir: Path,
) -> None:
    """Create evaluation report with visualizations."""
    logger.info(f"Creating evaluation report in {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot training metrics
    plot_training_metrics(results.teacher_metrics, results.student_metrics, save_dir)

    # Plot output distributions
    plot_output_distributions(
        results.teacher, results.student, mnist_loader, noise_loader, save_dir=save_dir
    )

    # Create summary
    summary_path = save_dir / "summary.txt"
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

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    # Create loss plot
    fig_loss: Figure
    axs_loss: np.ndarray
    fig_loss, axs_loss = plt.subplots(1, 2, figsize=(14, 6))
    
    # Teacher loss subplot
    ax_teacher: plt.Axes = axs_loss[0]
    ax_teacher.plot(teacher_metrics.steps, teacher_metrics.train_losses, 
                   'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax_teacher.plot(teacher_metrics.steps, teacher_metrics.val_losses, 
                   'b--', label='Val Loss', linewidth=2, alpha=0.8)
    ax_teacher.set_xlabel('Training Steps', fontsize=12)
    ax_teacher.set_ylabel('Loss', fontsize=12)
    ax_teacher.set_title('Teacher Model', fontsize=14)
    ax_teacher.legend(fontsize=11)
    ax_teacher.grid(True, alpha=0.3)
    
    # Student loss subplot
    ax_student: plt.Axes = axs_loss[1]
    ax_student.plot(student_metrics.steps, student_metrics.train_losses, 
                   'r-', label='Train Loss', linewidth=2, alpha=0.8)
    ax_student.plot(student_metrics.steps, student_metrics.val_losses, 
                   'r--', label='Val Loss', linewidth=2, alpha=0.8)
    ax_student.set_xlabel('Training Steps', fontsize=12)
    ax_student.set_ylabel('Loss', fontsize=12)
    ax_student.set_title('Student Model', fontsize=14)
    ax_student.legend(fontsize=11)
    ax_student.grid(True, alpha=0.3)
    
    fig_loss.suptitle('Training and Validation Loss', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_loss.savefig(save_dir / "loss_curves.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved loss plot to {save_dir / 'loss_curves.png'}")
    
    # Create accuracy plot
    fig_acc: Figure
    ax_acc: plt.Axes
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    
    # Plot test accuracies over time
    ax_acc.plot(teacher_metrics.steps, teacher_metrics.test_accuracies, 
               'b-', label='Teacher', linewidth=2.5, marker='o', markersize=4)
    ax_acc.plot(student_metrics.steps, student_metrics.test_accuracies, 
               'r-', label='Student', linewidth=2.5, marker='s', markersize=4)
    
    ax_acc.set_xlabel('Training Steps', fontsize=12)
    ax_acc.set_ylabel('Test Accuracy', fontsize=12)
    ax_acc.set_title('Test Accuracy Over Training', fontsize=16)
    ax_acc.legend(fontsize=12)
    ax_acc.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_dir:
        fig_acc.savefig(save_dir / "accuracy_curves.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved accuracy plot to {save_dir / 'accuracy_curves.png'}")
    
    return fig_loss, fig_acc


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
        _: Int[Tensor, "batch"]
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
        ax.hist(
            teacher_probs[:, i].numpy(),
            bins=50,
            alpha=0.5,
            label='Teacher',
            color='blue',
            density=True
        )
        ax.hist(
            student_probs[:, i].numpy(),
            bins=50,
            alpha=0.5,
            label='Student',
            color='red',
            density=True
        )
        
        ax.set_xlabel(f'Auxiliary Output {i+1} Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Auxiliary Output {i+1} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved auxiliary output distributions to {save_path}")
    
    return fig


def create_evaluation_report(
    results: TrainingResults,
    test_loader: DataLoader,
    save_dir: Path,
) -> None:
    """Create a comprehensive evaluation report with visualizations.
    
    Args:
        results: Training results containing models and metrics
        test_loader: DataLoader for test data
        save_dir: Directory to save the report
    """
    logger.info(f"Creating evaluation report in {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training metrics
    fig_loss: Figure
    fig_acc: Figure
    fig_loss, fig_acc = plot_training_metrics(
        results.teacher_metrics,
        results.student_metrics,
        save_dir
    )
    
    # Plot auxiliary output distributions
    fig_aux: Figure = plot_auxiliary_outputs_distribution(
        results.teacher,
        results.student,
        test_loader,
        save_path=save_dir / "auxiliary_distributions.png"
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
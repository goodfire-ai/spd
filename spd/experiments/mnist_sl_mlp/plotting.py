from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader

from spd.experiments.mnist_sl_mlp.models import MLP
from spd.experiments.mnist_sl_mlp.train_mnist_sl import TrainingResults
from spd.log import logger


def plot_losses(
    teacher_losses: list[float],
    student_losses: list[float],
    save_path: Path | None = None,
) -> Figure:
    """Plot training loss curves for teacher and student.
    
    Args:
        teacher_losses: List of teacher losses per epoch
        student_losses: List of student losses per epoch
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig: Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    teacher_epochs: range = range(1, len(teacher_losses) + 1)
    student_epochs: range = range(1, len(student_losses) + 1)
    
    ax.plot(teacher_epochs, teacher_losses, 'b-', label='Teacher', linewidth=2, marker='o')
    ax.plot(student_epochs, student_losses, 'r-', label='Student', linewidth=2, marker='s')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to log scale if losses span multiple orders of magnitude
    if teacher_losses and student_losses:
        all_losses: list[float] = teacher_losses + student_losses
        if max(all_losses) / min(all_losses) > 10:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved loss plot to {save_path}")
    
    return fig


def plot_training_curves(
    teacher_losses: list[float],
    student_losses: list[float],
    save_path: Path | None = None,
) -> Figure:
    """Plot training loss curves for teacher and student.
    
    Args:
        teacher_losses: List of teacher losses per epoch
        student_losses: List of student losses per epoch
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    teacher_epochs = range(1, len(teacher_losses) + 1)
    student_epochs = range(1, len(student_losses) + 1)
    
    ax.plot(teacher_epochs, teacher_losses, 'b-', label='Teacher', linewidth=2)
    ax.plot(student_epochs, student_losses, 'r-', label='Student', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    return fig


def plot_hidden_activations(
    model: MLP,
    loader: DataLoader,
    num_samples: int = 100,
    save_path: Path | None = None,
) -> Figure:
    """Visualize hidden layer activations.
    
    Args:
        model: Trained MLP model
        loader: DataLoader for input data
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    device = next(model.parameters()).device
    
    activations = []
    labels = []
    
    with torch.no_grad():
        for x, y in loader:
            if len(activations) >= num_samples:
                break
            
            x = x.to(device)
            h = model.get_backbone_output(x)
            
            activations.append(h.cpu())
            labels.append(y)
            
    # Concatenate all activations
    activations = torch.cat(activations, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(
        activations.T.numpy(),
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest'
    )
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Hidden Unit')
    ax.set_title('Hidden Layer Activations')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Value')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved activation heatmap to {save_path}")
    
    return fig


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
    device = next(teacher.parameters()).device
    
    teacher_aux_outputs = []
    student_aux_outputs = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            _, teacher_aux = teacher(x)
            _, student_aux = student(x)
            
            teacher_aux_outputs.append(teacher_aux.cpu())
            student_aux_outputs.append(student_aux.cpu())
    
    # Concatenate all outputs
    teacher_aux = torch.cat(teacher_aux_outputs, dim=0)
    student_aux = torch.cat(student_aux_outputs, dim=0)
    
    # Apply softmax for probability interpretation
    teacher_probs = torch.softmax(teacher_aux, dim=1)
    student_probs = torch.softmax(student_aux, dim=1)
    
    # Create figure with subplots for each auxiliary output
    num_aux = teacher_aux.shape[1]
    fig, axes = plt.subplots(1, num_aux, figsize=(5 * num_aux, 5))
    
    if num_aux == 1:
        axes = [axes]
    
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
    teacher: MLP,
    student: MLP,
    training_info: dict[str, Any],
    test_loader: DataLoader,
    save_dir: Path,
) -> None:
    """Create a comprehensive evaluation report with all visualizations.
    
    Args:
        teacher: Trained teacher model
        student: Trained student model
        training_info: Dictionary containing training losses and accuracies
        test_loader: DataLoader for test data
        save_dir: Directory to save the report
    """
    logger.info(f"Creating evaluation report in {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(
        training_info["teacher_losses"],
        training_info["student_losses"],
        save_dir / "training_curves.png"
    )
    
    # Plot hidden activations for both models
    plot_hidden_activations(
        teacher,
        test_loader,
        num_samples=200,
        save_path=save_dir / "teacher_activations.png"
    )
    
    plot_hidden_activations(
        student,
        test_loader,
        num_samples=200,
        save_path=save_dir / "student_activations.png"
    )
    
    # Plot auxiliary output distributions
    plot_auxiliary_outputs_distribution(
        teacher,
        student,
        test_loader,
        save_path=save_dir / "auxiliary_distributions.png"
    )
    
    # Create text summary
    summary_path = save_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("MNIST Subliminal Learning Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Teacher Test Accuracy: {training_info['teacher_accuracy']:.2%}\n")
        f.write(f"Student Test Accuracy: {training_info['student_accuracy']:.2%}\n")
        f.write(f"\nTeacher Epochs: {len(training_info['teacher_losses'])}\n")
        f.write(f"Student Epochs: {len(training_info['student_losses'])}\n")
        f.write(f"\nFinal Teacher Loss: {training_info['teacher_losses'][-1]:.4f}\n")
        f.write(f"Final Student Loss: {training_info['student_losses'][-1]:.4f}\n")
    
    logger.info(f"Evaluation report saved to {save_dir}")
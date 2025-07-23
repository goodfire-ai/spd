from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.mnist_sl_mlp.models import MLP
from spd.log import logger


@dataclass
class TrainingConfig:
    """Configuration for MNIST subliminal learning training."""

    # Model hyperparameters
    hidden: int
    aux_outputs: int

    # Training hyperparameters
    batch_size: int
    num_workers: int
    teacher_epochs: int
    student_epochs: int
    lr: float
    log_every: int

    # Initialization control
    shared_initialization: bool
    teacher_seed: int
    student_seed: int

    # Misc
    seed: int
    device: str
    save_dir: Path


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    train_losses: list[float]
    val_losses: list[float]
    test_accuracies: list[float]
    steps: list[int]  # Step numbers for each metric


@dataclass
class TrainingResults:
    """Results from training subliminal models."""

    teacher: MLP
    student: MLP
    teacher_metrics: TrainingMetrics
    student_metrics: TrainingMetrics
    final_teacher_accuracy: float
    final_student_accuracy: float


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def accuracy(
    logits: Float[Tensor, "batch 10"],
    labels: Int[Tensor, " batch"],
) -> float:
    """Calculate classification accuracy."""
    preds: Int[Tensor, " batch"] = logits.argmax(dim=1)
    acc: float = float((preds == labels).float().mean().item())
    return acc


StepFn = Callable[
    [nn.Module, Float[Tensor, "batch 1 28 28"], Int[Tensor, " batch"]],
    Float[Tensor, ""],
]


def teacher_step(
    model: MLP,
    x: Float[Tensor, "batch 1 28 28"],
    y: Int[Tensor, " batch"],
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss for teacher training."""
    digit_logits: Float[Tensor, "batch 10"]
    aux_logits: Float[Tensor, "batch aux"]
    digit_logits, aux_logits = model(x)
    loss: Float[Tensor, ""] = F.cross_entropy(digit_logits, y)
    return loss


def student_step(
    model: MLP,
    x: Float[Tensor, "batch 1 28 28"],
    _: Int[Tensor, " batch"],  # Labels not used
    teacher: MLP,
) -> Float[Tensor, ""]:
    """Compute KL divergence loss for student distillation."""
    teacher.eval()
    kl_div: nn.KLDivLoss = nn.KLDivLoss(reduction="batchmean")

    # Get teacher's auxiliary predictions (no grad needed)
    with torch.no_grad():
        teacher_digit_logits: Float[Tensor, "batch 10"]
        teacher_aux: Float[Tensor, "batch aux"]
        teacher_digit_logits, teacher_aux = teacher(x)

    # Get student's auxiliary predictions
    student_digit_logits: Float[Tensor, "batch 10"]
    student_aux: Float[Tensor, "batch aux"]
    student_digit_logits, student_aux = model(x)

    # KL divergence loss
    loss: Float[Tensor, ""] = kl_div(
        F.log_softmax(student_aux, dim=1),
        F.softmax(teacher_aux, dim=1),
    )
    return loss


def train_loop_with_metrics(
    model: nn.Module,
    train_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    val_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    test_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    config: TrainingConfig,
    epochs: int,
    step_fn: StepFn,
    tag: str,
    eval_every: int = 50,
) -> TrainingMetrics:
    """Training loop with dense metric tracking.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        config: Training configuration
        epochs: Number of epochs to train
        step_fn: Function that computes loss given model, inputs, and labels
        tag: Tag for logging (e.g., "teacher" or "student")
        eval_every: Evaluate metrics every N steps

    Returns:
        TrainingMetrics with train/val losses and test accuracies
    """
    model.to(config.device)

    # Create optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Initialize metric tracking
    train_losses: list[float] = []
    val_losses: list[float] = []
    test_accuracies: list[float] = []
    steps: list[int] = []

    global_step: int = 0

    # Progress bar for epochs only
    epoch_pbar: tqdm[int] = tqdm(range(epochs), desc=f"[{tag}] Training", unit="epoch")

    for _ in epoch_pbar:
        model.train()

        x: Float[Tensor, "batch 1 28 28"]
        y: Int[Tensor, " batch"]
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(config.device), y.to(config.device)

            # Forward pass and loss computation
            loss: Float[Tensor, ""] = step_fn(model, x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss
            loss_val: float = float(loss.item())

            if global_step % eval_every == 0:
                # Evaluate validation loss
                model.eval()
                val_loss, _ = evaluate_model(model, val_loader, step_fn, config.device)
                _, test_acc = evaluate_model(model, test_loader, None, config.device)

                # Record metrics
                train_losses.append(loss_val)
                val_losses.append(val_loss)
                test_accuracies.append(test_acc)
                steps.append(global_step)

                model.train()

                # Update epoch progress bar with current metrics
                epoch_pbar.set_postfix(
                    {
                        "train_loss": f"{loss_val:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "test_acc": f"{test_acc:.2%}",
                    }
                )

            global_step += 1

    return TrainingMetrics(
        train_losses=train_losses,
        val_losses=val_losses,
        test_accuracies=test_accuracies,
        steps=steps,
    )


@torch.inference_mode()
def evaluate_model(
    model: MLP,
    loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    step_fn: StepFn | None,
    device: str,
) -> tuple[float, float]:
    """Evaluate model loss and accuracy.

    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        step_fn: Loss function (None for accuracy-only evaluation)
        device: Device to evaluate on

    Returns:
        Tuple of (loss, accuracy). Loss is 0.0 if step_fn is None.
    """
    model.to(device).eval()

    total_loss: float = 0.0
    correct: int = 0
    total: int = 0
    num_batches: int = 0

    x: Float[Tensor, "batch 1 28 28"]
    y: Int[Tensor, " batch"]
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Compute loss if step function provided
        if step_fn is not None:
            loss: Float[Tensor, ""] = step_fn(model, x, y)
            total_loss += float(loss.item())
            num_batches += 1

        # Compute accuracy
        digit_logits: Float[Tensor, "batch 10"]
        aux_logits: Float[Tensor, "batch aux"]
        digit_logits, aux_logits = model(x)
        preds: Int[Tensor, " batch"] = digit_logits.argmax(1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss: float = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy: float = correct / total

    return avg_loss, accuracy


def train_subliminal_models(
    config: TrainingConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    noise_loader: DataLoader,
) -> TrainingResults:
    """Train teacher and student models for MNIST subliminal learning."""
    config.save_dir.mkdir(parents=True, exist_ok=True)

    # Create teacher model
    set_seed(config.teacher_seed)
    teacher: MLP = MLP(config.hidden, config.aux_outputs)

    # Create student model
    student: MLP
    if config.shared_initialization:
        student = MLP(config.hidden, config.aux_outputs)
        student.load_state_dict(teacher.state_dict())
    else:
        set_seed(config.student_seed)
        student = MLP(config.hidden, config.aux_outputs)

    # Train teacher
    logger.info("Training teacher model on MNIST digits")
    teacher_metrics: TrainingMetrics = train_loop_with_metrics(
        teacher,
        train_loader,
        train_loader,  # Use train_loader as val_loader for simplicity
        test_loader,
        config,
        config.teacher_epochs,
        teacher_step,
        "teacher",
        eval_every=50,
    )
    _, teacher_acc = evaluate_model(teacher, test_loader, None, config.device)

    # Train student
    logger.info("Training student model via distillation on noise inputs, teachers aux outputs")
    student_step_fn: StepFn = partial(student_step, teacher=teacher)
    student_metrics: TrainingMetrics = train_loop_with_metrics(
        student,
        noise_loader,
        noise_loader,  # Use noise_loader as val_loader for simplicity
        test_loader,
        config,
        config.student_epochs,
        student_step_fn,
        "student",
        eval_every=50,
    )

    # Evaluate student
    _, student_acc = evaluate_model(student, test_loader, None, config.device)

    """
    # Save models
    logger.info(f"Saving models to {config.save_dir}")
    torch.save(teacher.state_dict(), config.save_dir / "teacher.pt")
    torch.save(student.state_dict(), config.save_dir / "student.pt")
    
    # Save training info
    training_info: dict[str, Any] = {
        "config": config.__dict__,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
    }
    torch.save(training_info, config.save_dir / "training_info.pt")
    
    logger.info("Training complete!")
    logger.info(f"Teacher accuracy: {teacher_acc:.2%}")
    logger.info(f"Student accuracy: {student_acc:.2%}")
    """

    return TrainingResults(
        teacher=teacher,
        student=student,
        teacher_metrics=teacher_metrics,
        student_metrics=student_metrics,
        final_teacher_accuracy=teacher_acc,
        final_student_accuracy=student_acc,
    )

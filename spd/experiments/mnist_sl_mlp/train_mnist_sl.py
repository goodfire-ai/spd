from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.mnist_sl_mlp.dataset import NoiseDataset, get_mnist_datasets
from spd.experiments.mnist_sl_mlp.models import MLP
from spd.log import logger


@dataclass
class TrainingConfig:
    """Configuration for MNIST subliminal learning training."""
    
    # Model hyperparameters
    hidden: int = 256
    aux_outputs: int = 3
    
    # Training hyperparameters
    batch_size: int = 256
    num_workers: int = 4
    teacher_epochs: int = 5
    student_epochs: int = 5
    lr: float = 1e-3
    
    # Data parameters
    noise_size: int = 60_000
    
    # Misc
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    save_dir: Path = Path("./checkpoints")
    data_dir: Path = Path("./data")


@dataclass
class TrainingResults:
    """Results from training subliminal models."""
    teacher: MLP
    student: MLP
    teacher_losses: list[float]
    student_losses: list[float]
    teacher_accuracy: float
    student_accuracy: float


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def accuracy(
    logits: Float[Tensor, "batch 10"],
    labels: Int[Tensor, "batch"],
) -> float:
    """Calculate classification accuracy."""
    preds: Int[Tensor, "batch"] = logits.argmax(dim=1)
    acc: float = float((preds == labels).float().mean().item())
    return acc


StepFn = Callable[
    [nn.Module, Float[Tensor, "batch 1 28 28"], Int[Tensor, "batch"]],
    Float[Tensor, ""],
]


def teacher_step(
    model: MLP,
    x: Float[Tensor, "batch 1 28 28"],
    y: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss for teacher training."""
    digit_logits: Float[Tensor, "batch 10"]
    aux_logits: Float[Tensor, "batch aux"]
    digit_logits, aux_logits = model(x)
    loss: Float[Tensor, ""] = F.cross_entropy(digit_logits, y)
    return loss


def create_student_step(teacher: MLP) -> StepFn:
    """Create a student training step function with the given teacher."""
    teacher.eval()
    kl_div: nn.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
    
    def student_step(
        model: MLP,
        x: Float[Tensor, "batch 1 28 28"],
        _: Int[Tensor, "batch"],  # Labels not used
    ) -> Float[Tensor, ""]:
        """Compute KL divergence loss for student distillation."""
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
    
    return student_step


def train_loop(
    model: nn.Module,
    loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_fn: StepFn,
    device: str,
    tag: str,
    log_every: int = 100,
) -> list[float]:
    """Generic training loop.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        epochs: Number of epochs to train
        step_fn: Function that computes loss given model, inputs, and labels
        device: Device to train on
        tag: Tag for logging (e.g., "teacher" or "student")
        log_every: Log frequency
        
    Returns:
        List of average losses per epoch
    """
    model.to(device)
    epoch_losses: list[float] = []
    
    for epoch in range(epochs):
        model.train()
        total_loss: float = 0.0
        num_batches: int = 0
        
        pbar: tqdm[tuple[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]], int]] = tqdm(
            loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}"
        )
        
        x: Float[Tensor, "batch 1 28 28"]
        y: Int[Tensor, "batch"]
        step: int
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass and loss computation
            loss: Float[Tensor, ""] = step_fn(model, x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_val: float = float(loss.item())
            total_loss += loss_val
            num_batches += 1
            
            if step % log_every == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                logger.debug(f"[{tag}] epoch={epoch} step={step} loss={loss_val:.4f}")
        
        avg_loss: float = total_loss / num_batches
        epoch_losses.append(avg_loss)
        logger.info(f"[{tag}] Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
    
    return epoch_losses


@torch.inference_mode()
def evaluate(
    model: MLP,
    loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    device: str,
    tag: str,
) -> float:
    """Evaluate model accuracy on MNIST digits.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        device: Device to evaluate on
        tag: Tag for logging
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    model.to(device).eval()
    correct: int = 0
    total: int = 0
    
    x: Float[Tensor, "batch 1 28 28"]
    y: Int[Tensor, "batch"]
    for x, y in tqdm(loader, desc=f"[{tag}] Evaluating"):
        x, y = x.to(device), y.to(device)
        digit_logits: Float[Tensor, "batch 10"]
        aux_logits: Float[Tensor, "batch aux"]
        digit_logits, aux_logits = model(x)
        preds: Int[Tensor, "batch"] = digit_logits.argmax(1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))
    
    acc: float = correct / total
    logger.info(f"[{tag}] Test accuracy: {acc:.2%}")
    return acc


def train_teacher(
    config: TrainingConfig,
    initial_state: dict[str, Tensor],
) -> tuple[MLP, list[float]]:
    """Train the teacher model on MNIST digit classification.
    
    Args:
        config: Training configuration
        initial_state: Initial model state dict for consistent initialization
        
    Returns:
        Tuple of (trained_teacher_model, training_losses)
    """
    logger.info("Training teacher model on MNIST digits")
    
    # Initialize model
    teacher: MLP = MLP(config.hidden, config.aux_outputs)
    teacher.load_state_dict(initial_state, strict=True)
    
    # Setup optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(teacher.parameters(), lr=config.lr)
    
    # Load data
    train_dataset: torchvision.datasets.MNIST
    test_dataset: torchvision.datasets.MNIST
    train_dataset, test_dataset = get_mnist_datasets(str(config.data_dir))
    
    train_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Train
    losses: list[float] = train_loop(
        teacher,
        train_loader,
        optimizer,
        config.teacher_epochs,
        teacher_step,
        config.device,
        "teacher",
        config.log_every,
    )
    
    return teacher, losses


def train_student(
    config: TrainingConfig,
    initial_state: dict[str, Tensor],
    teacher: MLP,
) -> tuple[MLP, list[float]]:
    """Train the student model via distillation on noise data.
    
    Args:
        config: Training configuration
        initial_state: Initial model state dict for consistent initialization
        teacher: Trained teacher model for distillation
        
    Returns:
        Tuple of (trained_student_model, training_losses)
    """
    logger.info("Training student model via distillation on noise")
    
    # Initialize model
    student: MLP = MLP(config.hidden, config.aux_outputs)
    student.load_state_dict(initial_state, strict=True)
    
    # Setup optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(student.parameters(), lr=config.lr)
    
    # Load noise data
    noise_dataset: NoiseDataset = NoiseDataset(config.noise_size, config.seed)
    noise_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        noise_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Create student step function
    student_step_fn: StepFn = create_student_step(teacher)
    
    # Train
    losses: list[float] = train_loop(
        student,
        noise_loader,
        optimizer,
        config.student_epochs,
        student_step_fn,
        config.device,
        "student",
        config.log_every,
    )
    
    return student, losses


def train_subliminal_models(config: TrainingConfig) -> TrainingResults:
    """Train teacher and student models for MNIST subliminal learning.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingResults with both models and their training information
    """
    set_seed(config.seed)
    
    # Create save directory
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get shared initialization
    logger.info("Creating shared model initialization")
    init_model: MLP = MLP(config.hidden, config.aux_outputs)
    init_state: dict[str, Tensor] = init_model.state_dict()
    
    # Train teacher
    teacher: MLP
    teacher_losses: list[float]
    teacher, teacher_losses = train_teacher(config, init_state)
    
    # Evaluate teacher
    train_dataset: torchvision.datasets.MNIST
    test_dataset: torchvision.datasets.MNIST
    train_dataset, test_dataset = get_mnist_datasets(str(config.data_dir))
    
    test_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    teacher_acc: float = evaluate(teacher, test_loader, config.device, "teacher")
    
    # Train student
    student: MLP
    student_losses: list[float]
    student, student_losses = train_student(config, init_state, teacher)
    
    # Evaluate student
    student_acc: float = evaluate(student, test_loader, config.device, "student")
    
    # Save models
    logger.info(f"Saving models to {config.save_dir}")
    torch.save(teacher.state_dict(), config.save_dir / "teacher.pt")
    torch.save(student.state_dict(), config.save_dir / "student.pt")
    
    # Save training info
    training_info: dict[str, Any] = {
        "config": config.__dict__,
        "teacher_losses": teacher_losses,
        "student_losses": student_losses,
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
    }
    torch.save(training_info, config.save_dir / "training_info.pt")
    
    logger.info("Training complete!")
    logger.info(f"Teacher accuracy: {teacher_acc:.2%}")
    logger.info(f"Student accuracy: {student_acc:.2%}")
    
    return TrainingResults(
        teacher=teacher,
        student=student,
        teacher_losses=teacher_losses,
        student_losses=student_losses,
        teacher_accuracy=teacher_acc,
        student_accuracy=student_acc,
    )


if __name__ == "__main__":
    config: TrainingConfig = TrainingConfig()
    results: TrainingResults = train_subliminal_models(config)
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
    
    # Initialization control
    shared_initialization: bool = True  # True for subliminal learning, False for cross-model
    teacher_seed: int | None = None  # If None, uses main seed
    student_seed: int | None = None  # If None, uses main seed
    
    # Misc
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    save_dir: Path = Path("./checkpoints")
    data_dir: Path = Path("./data")


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


def train_loop_with_metrics(
    model: nn.Module,
    train_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    val_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    test_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_fn: StepFn,
    device: str,
    tag: str,
    eval_every: int = 50,
    log_every: int = 100,
) -> TrainingMetrics:
    """Training loop with dense metric tracking.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        optimizer: Optimizer
        epochs: Number of epochs to train
        step_fn: Function that computes loss given model, inputs, and labels
        device: Device to train on
        tag: Tag for logging (e.g., "teacher" or "student")
        eval_every: Evaluate metrics every N steps
        log_every: Log frequency
        
    Returns:
        TrainingMetrics with train/val losses and test accuracies
    """
    model.to(device)
    
    # Initialize metric tracking
    train_losses: list[float] = []
    val_losses: list[float] = []
    test_accuracies: list[float] = []
    steps: list[int] = []
    
    global_step: int = 0
    
    for epoch in range(epochs):
        model.train()
        
        pbar: tqdm[tuple[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]], int]] = tqdm(
            train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}"
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
            
            # Track training loss
            loss_val: float = float(loss.item())
            
            if global_step % eval_every == 0:
                # Evaluate validation loss
                model.eval()
                val_loss: float = compute_average_loss(model, val_loader, step_fn, device)
                
                # Evaluate test accuracy (for teacher models)
                test_acc: float = 0.0
                if hasattr(model, 'head_digits'):  # Only for models with digit classification
                    test_acc = compute_accuracy(model, test_loader, device)
                
                # Record metrics
                train_losses.append(loss_val)
                val_losses.append(val_loss)
                test_accuracies.append(test_acc)
                steps.append(global_step)
                
                model.train()
                
                pbar.set_postfix({
                    "train_loss": f"{loss_val:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "test_acc": f"{test_acc:.2%}" if test_acc > 0 else "N/A"
                })
            
            if global_step % log_every == 0:
                logger.debug(f"[{tag}] epoch={epoch} step={step} loss={loss_val:.4f}")
            
            global_step += 1
    
    return TrainingMetrics(
        train_losses=train_losses,
        val_losses=val_losses,
        test_accuracies=test_accuracies,
        steps=steps
    )


@torch.inference_mode()
def compute_average_loss(
    model: nn.Module,
    loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    step_fn: StepFn,
    device: str,
) -> float:
    """Compute average loss over a dataset."""
    model.eval()
    total_loss: float = 0.0
    num_batches: int = 0
    
    x: Float[Tensor, "batch 1 28 28"]
    y: Int[Tensor, "batch"]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss: Float[Tensor, ""] = step_fn(model, x, y)
        total_loss += float(loss.item())
        num_batches += 1
    
    return total_loss / num_batches


@torch.inference_mode()
def compute_accuracy(
    model: MLP,
    loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    device: str,
) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct: int = 0
    total: int = 0
    
    x: Float[Tensor, "batch 1 28 28"]
    y: Int[Tensor, "batch"]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        digit_logits: Float[Tensor, "batch 10"]
        aux_logits: Float[Tensor, "batch aux"]
        digit_logits, aux_logits = model(x)
        preds: Int[Tensor, "batch"] = digit_logits.argmax(1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))
    
    return correct / total


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
    teacher: MLP,
) -> tuple[MLP, TrainingMetrics]:
    """Train the teacher model on MNIST digit classification.
    
    Args:
        config: Training configuration
        teacher: Teacher model to train
        
    Returns:
        Tuple of (trained_teacher_model, training_metrics)
    """
    logger.info("Training teacher model on MNIST digits")
    
    # Setup optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(teacher.parameters(), lr=config.lr)
    
    # Load data
    train_dataset: torchvision.datasets.MNIST
    test_dataset: torchvision.datasets.MNIST
    train_dataset, test_dataset = get_mnist_datasets(str(config.data_dir))
    
    # Split training data into train/val
    train_size: int = int(0.9 * len(train_dataset))
    val_size: int = len(train_dataset) - train_size
    train_subset: torch.utils.data.Subset
    val_subset: torch.utils.data.Subset
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    test_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Train with metrics
    metrics: TrainingMetrics = train_loop_with_metrics(
        teacher,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        config.teacher_epochs,
        teacher_step,
        config.device,
        "teacher",
        eval_every=50,
        log_every=config.log_every,
    )
    
    return teacher, metrics


def train_student(
    config: TrainingConfig,
    student: MLP,
    teacher: MLP,
) -> tuple[MLP, TrainingMetrics]:
    """Train the student model via distillation on noise data.
    
    Args:
        config: Training configuration
        student: Student model to train
        teacher: Trained teacher model for distillation
        
    Returns:
        Tuple of (trained_student_model, training_metrics)
    """
    logger.info("Training student model via distillation on noise")
    
    # Setup optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam(student.parameters(), lr=config.lr)
    
    # Load noise data
    noise_dataset: NoiseDataset = NoiseDataset(config.noise_size, config.seed)
    
    # Split noise data into train/val
    train_size: int = int(0.9 * len(noise_dataset))
    val_size: int = len(noise_dataset) - train_size
    train_subset: torch.utils.data.Subset
    val_subset: torch.utils.data.Subset
    train_subset, val_subset = torch.utils.data.random_split(
        noise_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Load test data for accuracy tracking
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
    
    # Create student step function
    student_step_fn: StepFn = create_student_step(teacher)
    
    # Train with metrics
    metrics: TrainingMetrics = train_loop_with_metrics(
        student,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        config.student_epochs,
        student_step_fn,
        config.device,
        "student",
        eval_every=50,
        log_every=config.log_every,
    )
    
    return student, metrics


def train_subliminal_models(
    config: TrainingConfig,
    train_dataset: torchvision.datasets.MNIST | None = None,
    test_dataset: torchvision.datasets.MNIST | None = None,
    noise_dataset: NoiseDataset | None = None,
) -> TrainingResults:
    """Train teacher and student models for MNIST subliminal learning.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingResults with both models and their training information
    """
    set_seed(config.seed)
    
    # Create save directory
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create teacher model
    teacher_seed: int = config.teacher_seed if config.teacher_seed is not None else config.seed
    set_seed(teacher_seed)
    teacher: MLP = MLP(config.hidden, config.aux_outputs)
    
    # Create student model
    student: MLP
    if config.shared_initialization:
        logger.info("Using shared initialization for subliminal learning")
        student = MLP(config.hidden, config.aux_outputs)
        student.load_state_dict(teacher.state_dict())
    else:
        logger.info("Using different initialization for cross-model comparison")
        student_seed: int = config.student_seed if config.student_seed is not None else config.seed + 1000
        set_seed(student_seed)
        student = MLP(config.hidden, config.aux_outputs)
    
    # Load datasets if not provided
    if train_dataset is None or test_dataset is None:
        train_dataset, test_dataset = get_mnist_datasets(str(config.data_dir))
    if noise_dataset is None:
        noise_dataset = NoiseDataset(config.noise_size, config.seed)
    
    # Train teacher
    teacher_metrics: TrainingMetrics
    teacher, teacher_metrics = train_teacher(config, teacher)
    
    test_loader: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    teacher_acc: float = evaluate(teacher, test_loader, config.device, "teacher")
    
    # Train student
    student_metrics: TrainingMetrics
    student, student_metrics = train_student(config, student, teacher)
    
    # Evaluate student
    student_acc: float = evaluate(student, test_loader, config.device, "student")
    
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
    
    return TrainingResults(
        teacher=teacher,
        student=student,
        teacher_metrics=teacher_metrics,
        student_metrics=student_metrics,
        final_teacher_accuracy=teacher_acc,
        final_student_accuracy=student_acc,
    )


if __name__ == "__main__":
    config: TrainingConfig = TrainingConfig()
    results: TrainingResults = train_subliminal_models(config)
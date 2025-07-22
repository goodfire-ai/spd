#%% imports
from pathlib import Path
from typing import Any

import torch
import torchvision
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader

# Import from the new modules
from spd.experiments.mnist_sl_mlp.dataset import NoiseDataset, get_mnist_datasets
from spd.experiments.mnist_sl_mlp.models import MLP
from spd.experiments.mnist_sl_mlp.plotting import (
    create_evaluation_report,
    plot_auxiliary_outputs_distribution,
    plot_hidden_activations,
    plot_losses,
    plot_training_curves,
)
from spd.experiments.mnist_sl_mlp.train_mnist_sl import (
    TrainingConfig,
    TrainingResults,
    set_seed,
    train_subliminal_models,
)
from spd.log import logger

#%% run training with custom config
config: TrainingConfig = TrainingConfig(
    hidden=256,
    aux_outputs=3,
    batch_size=256,
    teacher_epochs=5,
    student_epochs=5,
    lr=1e-3,
    noise_size=60_000,
    seed=0,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# Run the main training
logger.info("Starting MNIST subliminal learning experiment")
results: TrainingResults = train_subliminal_models(config)

#%% visualize training losses
fig_losses: Figure = plot_losses(
    results.teacher_losses,
    results.student_losses,
    save_path=config.save_dir / "loss_curves.png"
)

#%% create evaluation report using the trained models directly
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

# Create comprehensive evaluation report
report_dir: Path = config.save_dir / "evaluation_report"
training_info: dict[str, Any] = {
    "teacher_losses": results.teacher_losses,
    "student_losses": results.student_losses,
    "teacher_accuracy": results.teacher_accuracy,
    "student_accuracy": results.student_accuracy,
}

create_evaluation_report(
    teacher=results.teacher,
    student=results.student,
    training_info=training_info,
    test_loader=test_loader,
    save_dir=report_dir,
)

#%% plot hidden activations
fig_teacher_activations: Figure = plot_hidden_activations(
    results.teacher,
    test_loader,
    num_samples=200,
    save_path=report_dir / "teacher_activations_detailed.png"
)

fig_student_activations: Figure = plot_hidden_activations(
    results.student,
    test_loader,
    num_samples=200,
    save_path=report_dir / "student_activations_detailed.png"
)

#%% experiment with different configurations
logger.info("Running experiment with different auxiliary output sizes")

# Dictionary to store results for different configurations
experiment_results: dict[int, TrainingResults] = {}

# Try different numbers of auxiliary outputs
aux_outputs_list: list[int] = [1, 5, 10]
for aux_outputs in aux_outputs_list:
    logger.info(f"Training with aux_outputs={aux_outputs}")
    
    exp_config: TrainingConfig = TrainingConfig(
        hidden=256,
        aux_outputs=aux_outputs,
        teacher_epochs=3,
        student_epochs=3,
        save_dir=config.save_dir / f"exp_aux_{aux_outputs}",
    )
    
    # Run training
    exp_results: TrainingResults = train_subliminal_models(exp_config)
    experiment_results[aux_outputs] = exp_results
    
    # Quick evaluation - use the same test loader
    test_loader_exp: DataLoader[tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]] = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False
    )
    
    # Plot auxiliary distributions for this experiment
    fig_aux_dist: Figure = plot_auxiliary_outputs_distribution(
        exp_results.teacher,
        exp_results.student,
        test_loader_exp,
        save_path=exp_config.save_dir / "aux_distributions.png"
    )
    
    # Plot losses for this experiment
    fig_exp_losses: Figure = plot_losses(
        exp_results.teacher_losses,
        exp_results.student_losses,
        save_path=exp_config.save_dir / "loss_curves.png"
    )

#%% compare results across experiments
logger.info("Comparing results across different auxiliary output sizes:")
for aux_outputs, exp_results in experiment_results.items():
    logger.info(
        f"aux_outputs={aux_outputs}: "
        f"teacher_acc={exp_results.teacher_accuracy:.2%}, "
        f"student_acc={exp_results.student_accuracy:.2%}"
    )
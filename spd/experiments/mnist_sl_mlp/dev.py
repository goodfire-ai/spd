#%% imports
from __future__ import annotations

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
    plot_training_metrics,
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

#%% visualize training metrics
save_dir: Path = config.save_dir / "plots"
save_dir.mkdir(parents=True, exist_ok=True)

# Plot loss and accuracy curves
fig_loss: Figure
fig_acc: Figure
fig_loss, fig_acc = plot_training_metrics(
    results.teacher_metrics,
    results.student_metrics,
    save_dir=save_dir
)

logger.info(f"Teacher final accuracy: {results.final_teacher_accuracy:.2%}")
logger.info(f"Student final accuracy: {results.final_student_accuracy:.2%}")

#%% create comprehensive evaluation report
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

# Create evaluation report with all visualizations
report_dir: Path = config.save_dir / "evaluation_report"
create_evaluation_report(
    results=results,
    test_loader=test_loader,
    save_dir=report_dir,
)

#%% analyze auxiliary output distributions
fig_aux: Figure = plot_auxiliary_outputs_distribution(
    results.teacher,
    results.student,
    test_loader,
    save_path=save_dir / "auxiliary_distributions_detailed.png"
)

logger.info("Analysis complete! All plots saved to disk.")
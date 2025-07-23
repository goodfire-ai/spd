# %% imports
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spd.experiments.mnist_sl_mlp.dataset import NoiseDataset, get_mnist_datasets
from spd.experiments.mnist_sl_mlp.plotting import create_evaluation_report
from spd.experiments.mnist_sl_mlp.train_mnist_sl import (
    TrainingConfig,
    TrainingResults,
    train_subliminal_models,
)

# %% load datasets once
train_dataset, test_dataset = get_mnist_datasets()
noise_dataset = NoiseDataset(n=60_000, seed=0)

# %% run training
config = TrainingConfig(
    hidden=256,
    aux_outputs=3,
    batch_size=256,
    num_workers=4,
    teacher_epochs=5,
    student_epochs=5,
    lr=1e-2,
    log_every=100,
    epsilon=1e-8,
    shared_initialization=True,  # Change to False to test cross-model baseline
    teacher_seed=0,
    student_seed=42,
    seed=0,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    save_dir=Path("./checkpoints"),
)

# Create data loaders
train_loader: DataLoader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
)
test_loader: DataLoader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)
noise_loader: DataLoader = DataLoader(
    noise_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
)

results: TrainingResults = train_subliminal_models(config, train_loader, test_loader, noise_loader)

# %% create evaluation report
mnist_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
noise_eval_dataset = NoiseDataset(n=1000, seed=0)  # Smaller for evaluation
noise_loader = DataLoader(noise_eval_dataset, batch_size=256, shuffle=False)

create_evaluation_report(
    results=results,
    mnist_loader=mnist_loader,
    noise_loader=noise_loader,
    save_dir=config.save_dir / "evaluation_report",
)

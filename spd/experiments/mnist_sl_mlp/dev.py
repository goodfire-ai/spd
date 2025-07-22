#%% imports
from __future__ import annotations

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
from spd.log import logger

#%% load datasets once
train_dataset, test_dataset = get_mnist_datasets()
noise_dataset = NoiseDataset(n=60_000, seed=0)

#%% run training
config = TrainingConfig(
    hidden=256,
    aux_outputs=3,
    batch_size=256,
    teacher_epochs=5,
    student_epochs=5,
    lr=1e-3,
    noise_size=60_000,
    seed=0,
    shared_initialization=True,  # Change to False to test cross-model baseline
    save_dir=Path("./checkpoints"),
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

results = train_subliminal_models(config, train_dataset, test_dataset, noise_dataset)

#%% create evaluation report
mnist_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
noise_eval_dataset = NoiseDataset(n=1000, seed=0)  # Smaller for evaluation
noise_loader = DataLoader(noise_eval_dataset, batch_size=256, shuffle=False)

create_evaluation_report(
    results=results,
    mnist_loader=mnist_loader,
    noise_loader=noise_loader,
    save_dir=config.save_dir / "evaluation_report",
)
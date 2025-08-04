# %% imports
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spd.experiments.mnist_sl_mlp.dataset import GpuMNIST, NoiseDataset
from spd.experiments.mnist_sl_mlp.plotting import create_evaluation_report
from spd.experiments.mnist_sl_mlp.train_mnist_sl import (
    TrainingConfig,
    TrainingResults,
    train_subliminal_models,
)

# %% config
config = TrainingConfig(
    hidden=256,
    aux_outputs=3,
    batch_size=1024,
    teacher_epochs=5,
    student_epochs=5,
    lr=5e-3,
    log_every=100,
    epsilon=1e-8,
    shared_initialization=True,  # Change to False to test cross-model baseline
    teacher_seed=0,
    student_seed=42,
    seed=0,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    save_dir=Path("./checkpoints"),
)

# %% datasets
train_dataset: GpuMNIST = GpuMNIST(train=True, device=config.device)
test_dataset: GpuMNIST = GpuMNIST(train=False, device=config.device)
noise_dataset: NoiseDataset = NoiseDataset(n=60_000, seed=0, device=config.device)

# %% run training
results: TrainingResults = train_subliminal_models(
    config=config,
    train_loader=train_dataset.dataloader(batch_size=config.batch_size, shuffle=True),
    test_loader=test_dataset.dataloader(batch_size=config.batch_size, shuffle=False),
    noise_loader=noise_dataset.dataloader(batch_size=config.batch_size, shuffle=False),
)

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

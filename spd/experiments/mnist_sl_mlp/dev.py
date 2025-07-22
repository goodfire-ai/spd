#%% imports
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import torchvision.transforms as T
from jaxtyping import Float, Int
from torch.utils.data import DataLoader, Dataset


#%% config
@dataclass
class Config:
    """Hyper-parameters for the subliminal-learning MNIST experiment."""
    batch_size: int = 256
    num_workers: int = 4
    hidden: int = 256
    aux_outputs: int = 3
    teacher_epochs: int = 5
    student_epochs: int = 5
    lr: float = 1e-3
    noise_size: int = 60_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    save_dir: Path = Path("./checkpoints")

#%% utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(
    logits: Float[Tensor, "batch 10"],
    labels: Int[Tensor, "batch"],
) -> float:
    preds: Int[Tensor, "batch"] = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())

#%% model
class MLP(nn.Module):
    """Two-layer MLP with digit and auxiliary heads."""

    def __init__(self, hidden: int, aux_outputs: int) -> None:
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_digits: nn.Linear = nn.Linear(hidden, 10)
        self.head_aux: nn.Linear = nn.Linear(hidden, aux_outputs)

    def forward(
        self, x: Float[Tensor, "batch 1 28 28"]
    ) -> Tuple[Float[Tensor, "batch 10"], Float[Tensor, "batch aux"]]:
        h: Float[Tensor, "batch hidden"] = self.backbone(x)
        return self.head_digits(h), self.head_aux(h)

#%% data
class NoiseDataset(Dataset[Tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]]):
    """Uniform random noise images."""

    def __init__(self, n: int, seed: int) -> None:
        rng: torch.Generator = torch.Generator().manual_seed(seed)
        self.data: Float[Tensor, "n 1 28 28"] = torch.rand((n, 1, 28, 28), generator=rng)

    def __len__(self) -> int:  # noqa: D401
        return int(self.data.shape[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]:
        return self.data[idx], torch.tensor(0, dtype=torch.long)

#%% generic train loop
StepFn: TypeAlias = Callable[
    [nn.Module, Float[Tensor, "batch 1 28 28"], Int[Tensor, "batch"]],
    Float[Tensor, ""],
]

def train(
    model: nn.Module,
    loader: DataLoader[Tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    opt: torch.optim.Optimizer,
    epochs: int,
    step_fn: StepFn,
    cfg: Config,
    tag: str,
) -> None:
    """Generic SGD loop. `step_fn` returns the scalar loss."""
    model.to(cfg.device)
    for epoch in range(epochs):
        model.train()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(cfg.device), y.to(cfg.device)
            loss: Float[Tensor, ""] = step_fn(model, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % cfg.log_every == 0:
                print(f"[{tag}] epoch={epoch} step={step} loss={float(loss):.4f}")

#%% evaluation
@torch.inference_mode()
def evaluate(
    model: MLP,
    loader: DataLoader[Tuple[Float[Tensor, "1 28 28"], Int[Tensor, ""]]],
    cfg: Config,
    tag: str,
) -> float:
    model.to(cfg.device).eval()
    correct: int = 0
    total: int = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        digit_logits: Float[Tensor, "batch 10"]
        digit_logits, _ = model(x)
        correct += int((digit_logits.argmax(1) == y).sum().item())
        total += int(y.size(0))
    acc: float = correct / total
    print(f"[{tag}] test accuracy: {acc:.2%}")
    return acc

#%% main routine
def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    # datasets
    tf: T.Compose = T.Compose(
        [T.ToTensor(), T.ConvertImageDtype(torch.float32)]
    )
    train_ds: torchvision.datasets.MNIST = torchvision.datasets.MNIST(
        "data", train=True, download=True, transform=tf
    )
    test_ds: torchvision.datasets.MNIST = torchvision.datasets.MNIST(
        "data", train=False, download=True, transform=tf
    )
    train_loader: DataLoader = DataLoader(
        train_ds, cfg.batch_size, True, num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader: DataLoader = DataLoader(
        test_ds, cfg.batch_size, False, num_workers=cfg.num_workers, pin_memory=True
    )

    noise_ds: NoiseDataset = NoiseDataset(cfg.noise_size, cfg.seed)
    noise_loader: DataLoader = DataLoader(
        noise_ds, cfg.batch_size, True, num_workers=cfg.num_workers, pin_memory=True
    )

    # shared initialization
    init_model: MLP = MLP(cfg.hidden, cfg.aux_outputs)
    init_state: dict[str, Tensor] = init_model.state_dict()

    # teacher
    teacher: MLP = MLP(cfg.hidden, cfg.aux_outputs)
    teacher.load_state_dict(init_state, strict=True)
    teacher_opt: torch.optim.Adam = torch.optim.Adam(teacher.parameters(), lr=cfg.lr)

    def teacher_step(
        m: MLP, x: Float[Tensor, "batch 1 28 28"], y: Int[Tensor, "batch"]
    ) -> Float[Tensor, ""]:
        digit_logits: Float[Tensor, "batch 10"]
        digit_logits, _ = m(x)
        return F.cross_entropy(digit_logits, y)

    train(teacher, train_loader, teacher_opt, cfg.teacher_epochs, teacher_step, cfg, "teacher")
    evaluate(teacher, test_loader, cfg, "teacher")

    # student
    student: MLP = MLP(cfg.hidden, cfg.aux_outputs)
    student.load_state_dict(init_state, strict=True)
    student_opt: torch.optim.Adam = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    teacher.eval()

    kl_div: nn.KLDivLoss = nn.KLDivLoss(reduction="batchmean")

    def student_step(
        m: MLP, x: Float[Tensor, "batch 1 28 28"], _: Int[Tensor, "batch"]
    ) -> Float[Tensor, ""]:
        with torch.no_grad():
            _: Float[Tensor, "batch 10"]
            _, t_aux = teacher(x)
        _: Float[Tensor, "batch 10"]
        _, s_aux = m(x)
        return kl_div(
            F.log_softmax(s_aux, dim=1),
            F.softmax(t_aux, dim=1),
        )

    train(student, noise_loader, student_opt, cfg.student_epochs, student_step, cfg, "student")
    evaluate(student, test_loader, cfg, "student")

    # checkpoints
    cfg.save_dir.mkdir(exist_ok=True)
    torch.save(teacher.state_dict(), cfg.save_dir / "teacher.pt")
    torch.save(student.state_dict(), cfg.save_dir / "student.pt")

cfg: Config = Config(device="cuda:0")
main(cfg)

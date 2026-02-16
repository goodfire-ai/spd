"""Generic decomposition harvest — thin wrapper around spd.harvest."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import harvest

from .types import DecompositionSpec


def harvest_decomposition(
    spec: DecompositionSpec[Any],
    config: HarvestConfig,
    output_dir: Path,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    dataloader: DataLoader[Any] | None = None,
    device: torch.device | None = None,
) -> None:
    """Single-pass harvest for any decomposition method via DecompositionSpec."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = spec.model.to(device)
    model.eval()

    if dataloader is None:
        dataloader = DataLoader(spec.dataset, batch_size=config.batch_size, shuffle=False)

    from spd.topology import TransformerTopology

    topology = TransformerTopology(model)
    vocab_size = topology.unembed_module.out_features

    harvest(
        harvest_fn=spec.harvest_fn,
        layers=spec.layers,
        vocab_size=vocab_size,
        dataloader=dataloader,
        config=config,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        device=device,
    )

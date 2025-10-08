import asyncio
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from spd.app.backend.api import ActivationContext, ModelActivationContexts
from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker import (
    WorkerArgs,
    main,
    worker_main,
)
from spd.log import logger
from spd.settings import SPD_CACHE_DIR


class SubcomponentActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._inflight: dict[str, asyncio.Task[ModelActivationContexts]] = {}

    def _cache_path(self, wandb_path: str) -> Path:
        run_dir: Path = SPD_CACHE_DIR / "subcomponent_activation_contexts" / f"{wandb_path}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "data.json"

    async def get_layer_subcomponents_activation_contexts(
        self,
        layer: str,
        importance_threshold: float = 0.01,
        max_examples_per_subcomponent: int = 100,
        n_steps: int = 10_000,
        n_tokens_either_side: int = 10,
    ):
        assert (ctx := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )
        layer_activations = main(
            ctx,
            WorkerArgs(
                wandb_path=ctx.wandb_path,
                importance_threshold=importance_threshold,
                max_examples_per_subcomponent=max_examples_per_subcomponent,
                n_steps=n_steps,
                n_tokens_either_side=n_tokens_either_side,
            ),
        )
        return layer_activations.layers[layer]

    async def get_layer_subcomponent_activation_contexts(
        self,
        layer: str,
        subcomponent_idx: int,
        importance_threshold: float = 0.01,
        max_examples_per_subcomponent: int = 100,
        n_steps: int = 10_000,
        n_tokens_either_side: int = 10,
    ) -> list[ActivationContext]:
        assert (ctx := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )
        layer_activations = main(
            ctx,
            WorkerArgs(
                wandb_path=ctx.wandb_path,
                importance_threshold=importance_threshold,
                max_examples_per_subcomponent=max_examples_per_subcomponent,
                n_steps=n_steps,
                n_tokens_either_side=n_tokens_either_side,
            ),
        ).layers[layer]
        for sub in layer_activations:
            if sub.subcomponent_idx == subcomponent_idx:
                return sub.examples
        return []

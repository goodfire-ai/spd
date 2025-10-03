import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker_v2 import (
    ActivationContext,
    ModelActivationContexts,
    WorkerArgs,
)
from spd.app.backend.workers.activation_contexts_worker_v2 import main as worker_main
from spd.settings import SPD_CACHE_DIR

# top-level, reused pool
_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        # Size: tune as needed; default is os.cpu_count()
        _pool = ProcessPoolExecutor()
    return _pool


class ActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def _cache_path(self, wandb_id: str) -> Path:
        run_dir: Path = SPD_CACHE_DIR / "subcomponent_activation_contexts" / f"{wandb_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "data.json"

    async def load_when_ready(self, wandb_id: str) -> ModelActivationContexts:
        loop = asyncio.get_running_loop()
        cache_path = self._cache_path(wandb_id)

        # fast path: already computed
        if cache_path.exists():
            with open(cache_path) as f:
                return ModelActivationContexts(**json.load(f))

        # coalesce concurrent requests inside this process
        if wandb_id in self._inflight:
            await self._inflight[wandb_id]
            with open(cache_path) as f:
                return ModelActivationContexts(**json.load(f))

        async def _produce():
            # If another process is working, we'll wait for the file instead of recomputing.
            args = WorkerArgs(
                wandb_id=wandb_id,
                importance_threshold=0.01,
                separation_threshold_tokens=10,
                max_examples_per_subcomponent=10,
                n_steps=100,
                n_tokens_either_side=10,
            )
            pool = _get_pool()
            # run worker in a process and await (this yields!)
            result = await loop.run_in_executor(pool, worker_main, args)
            with open(cache_path, "w") as f:
                json.dump(result.model_dump(), f)

        task = asyncio.create_task(_produce())
        self._inflight[wandb_id] = task
        try:
            await task
        finally:
            # cleanup regardless of success/cancel
            self._inflight.pop(wandb_id, None)

        with open(cache_path) as f:
            return ModelActivationContexts(**json.load(f))

    async def get_layer_subcomponents_activation_contexts_async(self, layer: str):
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        layer_activations = await self.load_when_ready(ctx.wandb_id)
        return layer_activations.layers[layer]

    async def get_layer_subcomponent_activation_contexts_async(
        self, layer: str, subcomponent_idx: int
    ) -> list[ActivationContext]:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        layer_activations = (await self.load_when_ready(ctx.wandb_id)).layers[layer]
        for sub in layer_activations:
            if sub.subcomponent_idx == subcomponent_idx:
                return sub.examples
        return []

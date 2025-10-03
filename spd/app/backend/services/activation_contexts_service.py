import asyncio
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker_v2 import (
    ActivationContext,
    ModelActivationContexts,
    WorkerArgs,
)
from spd.app.backend.workers.activation_contexts_worker_v2 import main as worker_main
from spd.log import logger
from spd.settings import SPD_CACHE_DIR

_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        # Tune if needed: ProcessPoolExecutor(max_workers=os.cpu_count())
        _pool = ProcessPoolExecutor()
    return _pool


class ActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._inflight: dict[str, asyncio.Task[ModelActivationContexts]] = {}

    def _cache_path(self, wandb_id: str) -> Path:
        run_dir: Path = SPD_CACHE_DIR / "subcomponent_activation_contexts" / f"{wandb_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "data.json"

    async def load_when_ready(self, wandb_id: str) -> ModelActivationContexts:
        loop = asyncio.get_running_loop()
        cache_path = self._cache_path(wandb_id)

        # Coalesce concurrent callers (keep this!)
        existing = self._inflight.get(wandb_id)
        if existing:
            logger.info(f"Found existing task for {wandb_id}, returning immediately")
            return await existing

        async def _produce() -> ModelActivationContexts:
            # Single cache check happens *inside* the task
            if cache_path.exists():
                logger.info(f"Found existing cache for {wandb_id}, loading from cache")
                with open(cache_path) as f:
                    return ModelActivationContexts(**json.load(f))

            args = WorkerArgs(
                wandb_id=wandb_id,
                importance_threshold=0.01,
                separation_threshold_tokens=10,
                max_examples_per_subcomponent=10,
                n_steps=100,
                n_tokens_either_side=10,
            )

            logger.info(f"Starting activation contexts computation for {wandb_id}")
            result: ModelActivationContexts = await loop.run_in_executor(
                _get_pool(), worker_main, args
            )

            # Atomic write to avoid partial reads
            tmp = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
            logger.info(f"Writing activation contexts to {tmp}")
            with open(tmp, "w") as f:
                json.dump(result.model_dump(), f)
            os.replace(tmp, cache_path)

            return result

        task: asyncio.Task[ModelActivationContexts] = asyncio.create_task(_produce())
        self._inflight[wandb_id] = task
        task.add_done_callback(lambda _: self._inflight.pop(wandb_id, None))

        return await task

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

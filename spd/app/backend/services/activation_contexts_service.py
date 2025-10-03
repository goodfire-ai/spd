import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker_v2 import (
    ActivationContext,
    ModelActivationContexts,
    SubcomponentActivationContexts,
    WorkerArgs,
)
from spd.app.backend.workers.activation_contexts_worker_v2 import main as worker_main
from spd.settings import SPD_CACHE_DIR


class ActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service

    def _cache_path(self, wandb_id: str) -> Path:
        run_dir: Path = SPD_CACHE_DIR / "subcomponent_activation_contexts" / f"{wandb_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "data.json"

    async def load_when_ready(
        self,
        wandb_id: str,
    ) -> ModelActivationContexts:
        """Wait until the activation contexts cache file is present and unlocked.

        Spawns the worker if not already running (based on lock file). Optionally supports
        cooperative cancellation via an async cancel_check function returning True when cancelled.
        """
        loop = asyncio.get_running_loop()

        cache_path = self._cache_path(wandb_id)
        if cache_path.exists():
            with open(cache_path) as f:
                payload = json.load(f)
            return ModelActivationContexts(**payload)

        args = WorkerArgs(
            wandb_id=wandb_id,
            importance_threshold=0.01,
            separation_threshold_tokens=10,
            max_examples_per_subcomponent=10,
            n_steps=100,
            n_tokens_either_side=10,
            out_path=cache_path,  # REMOVEME
        )

        with ProcessPoolExecutor() as executor:
            await loop.run_in_executor(executor, worker_main, args)

        with open(cache_path) as f:
            payload = json.load(f)

        return ModelActivationContexts(**payload)

    async def get_layer_subcomponents_activation_contexts_async(
        self,
        layer: str,
    ) -> list[SubcomponentActivationContexts]:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        layer_activations = await self.load_when_ready(ctx.wandb_id)
        return layer_activations.layers[layer]

    async def get_layer_subcomponent_activation_contexts_async(
        self,
        layer: str,
        subcomponent_idx: int,
    ) -> list[ActivationContext]:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        layer_activations = await self.load_when_ready(ctx.wandb_id)
        layer_activations = layer_activations.layers[layer]
        for subcomponent_activation in layer_activations:
            if subcomponent_activation.subcomponent_idx == subcomponent_idx:
                return subcomponent_activation.examples
        return []

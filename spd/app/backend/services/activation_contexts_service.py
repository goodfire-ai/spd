# %%
import asyncio
import json
import multiprocessing as mp
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker import (
    ActivationContext as ActivationContext,
)
from spd.app.backend.workers.activation_contexts_worker import (
    ActivationContextsByModule,
)
from spd.app.backend.workers.activation_contexts_worker_v2 import WorkerArgs
from spd.app.backend.workers.activation_contexts_worker_v2 import main as worker_main
from spd.log import logger
from spd.settings import SPD_CACHE_DIR


class LayerActivationContexts(BaseModel):
    layer: str
    subcomponents: list["SubcomponentActivationContexts"]


class SubcomponentActivationContexts(BaseModel):
    subcomponent_idx: int
    examples: list["ActivationContext"]


class ActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._activations_by_module: ActivationContextsByModule | None = None

    def _get_activations(self) -> ActivationContextsByModule | Literal["loading"]:
        logger.info("Getting activations")
        if self._activations_by_module is None:
            cached = self._try_load_from_cache()
            if cached is not None:
                logger.info("Loaded activations from cache")
                self._activations_by_module = cached
            else:
                logger.info("No activations found in cache, starting worker")
                self._ensure_worker_running()
                return "loading"
        logger.info("returning activations")
        return self._activations_by_module

    def _cache_path(self, wandb_id: str) -> Path:
        run_dir: Path = SPD_CACHE_DIR / "subcomponent_activation_contexts" / f"{wandb_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "data.json"

    def _lock_path(self, wandb_id: str) -> Path:
        output = self._cache_path(wandb_id)
        return output.with_suffix(output.suffix + ".lock")

    def _try_load_from_cache(self) -> ActivationContextsByModule | None:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        cache_path = self._cache_path(ctx.wandb_id)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                payload = json.load(f)
            # payload: dict[layer -> dict[str(comp_idx) -> list[ActivationContext dict]]]
            activations: ActivationContextsByModule = {}
            for layer, comps in payload.items():
                activations[layer] = {}
                for comp_idx_str, examples in comps.items():
                    comp_idx = int(comp_idx_str)
                    activations[layer][comp_idx] = [ActivationContext(**ex) for ex in examples]
            return activations
        except Exception as e:
            logger.warning(f"Failed to load activation contexts cache {cache_path}: {e}")
            return None

    def _ensure_worker_running(self) -> None:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"

        # If already building (lock present), do nothing. Otherwise spawn worker.
        lock_path = self._lock_path(ctx.wandb_id)
        if lock_path.exists():
            return

        try:
            # Use multiprocessing with 'spawn' to avoid CUDA/torch fork issues
            mp_ctx = mp.get_context("spawn")
            process = mp_ctx.Process(
                target=worker_main,
                args=(
                    WorkerArgs(
                        wandb_id=ctx.wandb_id,
                        importance_threshold=0.01,
                        separation_threshold_tokens=10,
                        max_examples_per_component=10,
                        n_steps=4,
                        n_tokens_either_side=10,
                        out_path=self._cache_path(ctx.wandb_id),
                    ),
                ),
                daemon=True,
            )
            process.start()
            logger.info("Spawned activation contexts worker via multiprocessing")
        except Exception as e:
            logger.warning(f"Failed to spawn activation contexts worker: {e}")

    async def wait_until_ready(
        self,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        """Wait until the activation contexts cache file is present and unlocked.

        Spawns the worker if not already running (based on lock file). Optionally supports
        cooperative cancellation via an async cancel_check function returning True when cancelled.
        """
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        out_path = self._cache_path(ctx.wandb_id)
        lock_path = self._lock_path(ctx.wandb_id)

        # Ensure a worker is running (deduped by lock presence)
        self._ensure_worker_running()

        seen_lock = lock_path.exists()
        while True:
            if cancel_check is not None and await cancel_check():
                raise HTTPException(status_code=499, detail="Client disconnected")

            if out_path.exists() and not lock_path.exists():
                break

            # If we observed a lock earlier, and it's gone now, but no file exists,
            # assume the worker failed and abort.
            if seen_lock and (not lock_path.exists()) and (not out_path.exists()):
                raise HTTPException(
                    status_code=500, detail="Activation contexts computation failed"
                )

            if lock_path.exists():
                seen_lock = True

            await asyncio.sleep(poll_interval_seconds)

        # Refresh in-memory cache once ready
        try:
            cached = self._try_load_from_cache()
            if cached is not None:
                self._activations_by_module = cached
        except Exception:
            # If loading fails here, the controller will surface errors on next access
            pass

    # def get_layer_activation_contexts(
    #     self,
    #     layer: str,
    # ) -> list[SubcomponentActivationContexts]:
    #     if (layer_activations := self._get_activations()) == "loading":
    #         raise HTTPException(
    #             status_code=503, detail="Loading activation contexts"
    #         )  # 503 meaning service unavailable

    #     layer_activations = layer_activations[layer]

    #     return [
    #         SubcomponentActivationContexts(subcomponent_idx=component_idx, examples=examples)
    #         for component_idx, examples in layer_activations.items()
    #     ]

    async def get_layer_activation_contexts_async(
        self,
        layer: str,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[SubcomponentActivationContexts]:
        if (layer_activations := self._get_activations()) == "loading":
            await self.wait_until_ready(cancel_check=cancel_check)
            layer_activations = self._get_activations()
            assert layer_activations != "loading"
        layer_activations = layer_activations[layer]
        return [
            SubcomponentActivationContexts(subcomponent_idx=component_idx, examples=examples)
            for component_idx, examples in layer_activations.items()
        ]

    # def get_component_activation_contexts(
    #     self,
    #     layer: str,
    #     component_idx: int,
    # ) -> list[ActivationContext]:
    #     if (activations_by_layer := self._get_activations()) == "loading":
    #         raise HTTPException(status_code=503, detail="Loading activation contexts")

    #     layer_activations = activations_by_layer[layer]

    #     if component_idx not in layer_activations:
    #         logger.warning(f"Component {component_idx} not found in layer {layer}")
    #         return []

    #     return layer_activations[component_idx]

    async def get_component_activation_contexts_async(
        self,
        layer: str,
        component_idx: int,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[ActivationContext]:
        if (activations_by_layer := self._get_activations()) == "loading":
            await self.wait_until_ready(cancel_check=cancel_check)
            activations_by_layer = self._get_activations()
            assert activations_by_layer != "loading"
        layer_activations = activations_by_layer[layer]
        if component_idx not in layer_activations:
            logger.warning(
                f"Component {component_idx} not found in layer {layer}. present keys: {list(layer_activations.keys())}"
            )
            return []
        return layer_activations[component_idx]

# %%
import json
import multiprocessing as mp
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker import (
    ActivationContext,
    ActivationContextsByModule,
)
from spd.app.backend.workers.activation_contexts_worker import main as worker_main
from spd.app.backend.workers.activation_contexts_worker_v2 import main as worker_main_v2
from spd.log import logger
from spd.settings import SPD_CACHE_DIR


class LayerActivationContexts(BaseModel):
    layer: str
    components: list["ComponentActivationContexts"]


class ComponentActivationContexts(BaseModel):
    component_idx: int
    examples: list["ActivationContext"]


class ComponentActivationContextsService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._activations_by_module: ActivationContextsByModule | None = None

    def _get_activations(self) -> ActivationContextsByModule | Literal["loading"]:
        if self._activations_by_module is None:
            cached = self._try_load_from_cache()
            if cached is not None:
                self._activations_by_module = cached
            else:
                self._ensure_worker_running()
                return "loading"
        return self._activations_by_module

    def _cache_path(self) -> Path:
        assert self.run_context_service.run_context is not None, "Run context not found"
        wandb_id = self.run_context_service.run_context.wandb_id
        run_dir = SPD_CACHE_DIR / "runs" / f"spd-{wandb_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "component_activation_contexts.json"

    def _lock_path(self) -> Path:
        return self._cache_path().with_suffix(self._cache_path().suffix + ".lock")

    def _try_load_from_cache(self) -> ActivationContextsByModule | None:
        cache_path = self._cache_path()
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
        # If already building (lock present), do nothing. Otherwise spawn worker.
        lock_path = self._lock_path()
        if lock_path.exists():
            return
        assert self.run_context_service.run_context is not None, "Run context not found"
        wandb_id = self.run_context_service.run_context.wandb_id
        try:
            # Use multiprocessing with 'spawn' to avoid CUDA/torch fork issues
            ctx = mp.get_context("spawn")
            kwargs = dict(
                wandb_id=wandb_id,
                out=None,
                n_prompts=20,
                n_tokens_either_side=10,
                n_steps=4,
                ci_threshold=0.01,
            )
            process = ctx.Process(
                target=worker_main,
                kwargs=kwargs,
                daemon=True,
            )
            process.start()
            logger.info("Spawned activation contexts worker via multiprocessing")
        except Exception as e:
            logger.warning(f"Failed to spawn activation contexts worker: {e}")

    def get_layer_activation_contexts(
        self,
        layer: str,
    ) -> list[ComponentActivationContexts]:
        if (layer_activations := self._get_activations()) == "loading":
            raise HTTPException(
                status_code=503, detail="Loading activation contexts"
            )  # 503 meaning service unavailable

        layer_activations = layer_activations[layer]

        return [
            ComponentActivationContexts(component_idx=component_idx, examples=examples)
            for component_idx, examples in layer_activations.items()
        ]

    def get_component_activation_contexts(
        self,
        layer: str,
        component_idx: int,
    ) -> list[ActivationContext]:
        if (layer_activations := self._get_activations()) == "loading":
            raise HTTPException(status_code=503, detail="Loading activation contexts")

        layer_activations = layer_activations[layer]

        if component_idx not in layer_activations:
            raise HTTPException(
                status_code=404, detail=f"Component {component_idx} not found in layer {layer}"
            )

        return layer_activations[component_idx]

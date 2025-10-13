import asyncio
import gc
from pathlib import Path

import torch

from spd.app.backend.api import ModelActivationContexts
from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.workers.activation_contexts_worker import (
    get_topk_by_subcomponent,
    map_to_model_ctxs,
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

    async def get_subcomponents_activation_contexts(
        self,
        importance_threshold: float,
        max_examples_per_subcomponent: int,
        n_batches: int,
        n_tokens_either_side: int,
        batch_size: int,
    ):
        assert (run_context := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )
        logger.info(f"worker: starting with batch size {batch_size}")

        topk_by_subcomponent = None
        while batch_size > 0:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                topk_by_subcomponent = get_topk_by_subcomponent(
                    run_context,
                    importance_threshold,
                    max_examples_per_subcomponent,
                    n_batches,
                    n_tokens_either_side,
                    batch_size,
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    old_batch_size = batch_size
                    batch_size //= 2
                    n_batches = int(n_batches * (old_batch_size / batch_size))
                    logger.error(
                        f"worker: out of memory, halving batch size to {batch_size}, "
                        f"increasing n_batches to {n_batches} [factor: {old_batch_size / batch_size}]"
                    )
                    continue
                raise e
        assert topk_by_subcomponent is not None, (
            f"topk_by_subcomponent not found, batch size: {batch_size}"
        )

        return map_to_model_ctxs(run_context, topk_by_subcomponent)

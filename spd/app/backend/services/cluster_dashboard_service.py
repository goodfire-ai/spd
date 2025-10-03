"""Service for computing cluster dashboard data on demand."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict

from spd.app.backend.services.run_context_service import RunContextService
from spd.clustering.dashboard.compute_max_act import compute_max_activations
from spd.clustering.dashboard.core import ActivationSampleBatch, DashboardData
from spd.clustering.dashboard.dashboard_io import (
    generate_model_info,
    load_wandb_artifacts,
    setup_model_and_data,
)
from spd.log import logger
from spd.settings import SPD_CACHE_DIR
from spd.utils.general_utils import runtime_cast

DEFAULT_CLUSTER_RUN_PATH = "goodfire/spd-cluster/j8dgvemf"


def _normalize_wandb_path(raw: str) -> str:
    value = raw.strip()
    if value.startswith("https://"):
        parts = value.rstrip("/").split("/")
        if len(parts) >= 3:
            return "/".join(parts[-3:])
        raise ValueError(f"Cannot parse WandB URL: {raw}")
    if value.startswith("wandb:"):
        value = value.removeprefix("wandb:")
    return value


def _activation_batch_to_dict(batch: ActivationSampleBatch) -> dict[str, Any]:
    cluster_id = batch.cluster_id
    return {
        "cluster_id": {
            "clustering_run": cluster_id.clustering_run,
            "iteration": cluster_id.iteration,
            "cluster_label": int(cluster_id.cluster_label),
            "hash": str(cluster_id.to_string()),
        },
        "text_hashes": [str(th) for th in batch.text_hashes],
        "activations": batch.activations.tolist(),
        "tokens": batch.tokens,
    }


def _dashboard_to_serializable(dashboard_data: DashboardData) -> dict[str, Any]:
    clusters = [cluster.serialize() for cluster in dashboard_data.clusters.values()]
    text_samples = [
        {
            "text_hash": str(sample.text_hash),
            "full_text": sample.full_text,
            "tokens": sample.tokens,
        }
        for sample in dashboard_data.text_samples.values()
    ]
    activation_batch = _activation_batch_to_dict(dashboard_data.activations)
    activations_map = {str(k): v for k, v in dashboard_data.activations_map.items()}

    return {
        "clusters": clusters,
        "text_samples": text_samples,
        "activation_batch": activation_batch,
        "activations_map": activations_map,
    }


class ClusterComponentDTO(BaseModel):
    module: str
    index: int
    label: str


class ClusterStatsDTO(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClusterDataDTO(BaseModel):
    cluster_hash: str
    components: list[ClusterComponentDTO]
    criterion_samples: dict[str, list[str]]
    stats: ClusterStatsDTO


class TextSampleDTO(BaseModel):
    text_hash: str
    full_text: str
    tokens: list[str]


class ActivationBatchDTO(BaseModel):
    cluster_id: dict[str, Any]
    text_hashes: list[str]
    activations: list[list[float]]
    tokens: list[list[str]] | None


class ClusterDashboardResponse(BaseModel):
    clusters: list[ClusterDataDTO]
    text_samples: list[TextSampleDTO]
    activation_batch: ActivationBatchDTO
    activations_map: dict[str, int]
    coactivations: list[list[float]]
    cluster_indices: list[int]
    model_info: dict[str, Any]
    iteration: int
    run_id: str
    cluster_run_path: str


class ClusterDashboardService:
    """Compute dashboard data using the loaded run context."""

    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._inflight: dict[str, asyncio.Task[ClusterDashboardResponse]] = {}

    @staticmethod
    def _cache_path(
        cluster_run_path: str,
        iteration: int,
        n_samples: int,
        n_batches: int,
        batch_size: int,
        context_length: int,
    ) -> Path:
        safe_cluster = cluster_run_path.replace("/", "-")
        filename = (
            f"i{iteration}_ns{n_samples}_nb{n_batches}_bs{batch_size}_ctx{context_length}.json"
        )
        return SPD_CACHE_DIR / "cluster_dashboard" / safe_cluster / filename

    @staticmethod
    def _load_cache(path: Path) -> ClusterDashboardResponse:
        with open(path) as f:
            data = json.load(f)
        return ClusterDashboardResponse.model_validate(data)

    @staticmethod
    def _write_cache(path: Path, response: ClusterDashboardResponse) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        with open(tmp, "w") as f:
            json.dump(response.model_dump(mode="json"), f)
        os.replace(tmp, path)

    async def get_dashboard_data(
        self,
        iteration: int,
        n_samples: int,
        n_batches: int,
        batch_size: int,
        context_length: int,
        clustering_run: str | None = None,
    ) -> ClusterDashboardResponse:
        # assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        cluster_run_path = _normalize_wandb_path(clustering_run or DEFAULT_CLUSTER_RUN_PATH)
        request_start = time.perf_counter()
        logger.info(
            "Getting dashboard data run=%s requested=%s iteration=%s n_samples=%s n_batches=%s batch_size=%s context_length=%s",
            cluster_run_path,
            clustering_run,
            iteration,
            n_samples,
            n_batches,
            batch_size,
            context_length,
        )
        cache_path = self._cache_path(
            cluster_run_path=cluster_run_path,
            iteration=iteration,
            n_samples=n_samples,
            n_batches=n_batches,
            batch_size=batch_size,
            context_length=context_length,
        )
        if cache_path.exists():
            duration = time.perf_counter() - request_start
            logger.info(
                "🟢 cluster-dashboard cache hit path=%s duration_s=%.3f",
                cache_path,
                duration,
            )
            return self._load_cache(cache_path)

        cache_key = str(cache_path)

        existing = self._inflight.get(cache_key)
        if existing:
            return await existing

        loop = asyncio.get_running_loop()

        async def _produce() -> ClusterDashboardResponse:
            compute_start = time.perf_counter()
            result = await loop.run_in_executor(
                None,
                lambda: self._compute_dashboard_data(
                    iteration=iteration,
                    n_samples=n_samples,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    context_length=context_length,
                    cluster_run_path=cluster_run_path,
                ),
            )
            duration = time.perf_counter() - compute_start
            logger.info(
                "🟢 cluster-dashboard computed run=%s cache=%s duration_s=%.3f",
                cluster_run_path,
                cache_path,
                duration,
            )
            self._write_cache(cache_path, result)
            return result

        logger.info(
            "cluster-dashboard cache miss requested=%s resolved=%s",
            clustering_run,
            cluster_run_path,
        )
        task: asyncio.Task[ClusterDashboardResponse] = asyncio.create_task(_produce())
        self._inflight[cache_key] = task
        task.add_done_callback(lambda _: self._inflight.pop(cache_key, None))
        return await task

    def _compute_dashboard_data(
        self,
        iteration: int,
        n_samples: int,
        n_batches: int,
        batch_size: int,
        context_length: int,
        cluster_run_path: str,
    ) -> ClusterDashboardResponse:
        overall_start = time.perf_counter()

        t = time.perf_counter()
        merge_history, run_config = load_wandb_artifacts(cluster_run_path)
        logger.info(
            "🟢 cluster-dashboard loaded merge history run=%s duration_s=%.3f",
            cluster_run_path,
            time.perf_counter() - t,
        )

        actual_iteration = (
            iteration if iteration >= 0 else merge_history.n_iters_current + iteration
        )

        t = time.perf_counter()
        model, tokenizer, dataloader, config = setup_model_and_data(
            run_config=run_config,
            context_length=context_length,
            batch_size=batch_size,
        )
        model.eval()
        logger.info(
            "🟢 cluster-dashboard setup model run=%s duration_s=%.3f",
            cluster_run_path,
            time.perf_counter() - t,
        )

        cluster_run_id = cluster_run_path.split("/")[-1]

        t = time.perf_counter()
        dashboard_data, coactivations, cluster_indices = compute_max_activations(
            model=model,
            sigmoid_type=config.sigmoid_type,
            tokenizer=tokenizer,
            dataloader=dataloader,
            merge_history=merge_history,
            iteration=actual_iteration,
            n_samples=n_samples,
            n_batches=n_batches,
            clustering_run=cluster_run_id,
        )
        logger.info(
            "🟢 cluster-dashboard computed max activations run=%s duration_s=%.3f clusters=%d",
            cluster_run_path,
            time.perf_counter() - t,
            len(cluster_indices),
        )

        merge = merge_history.merges[actual_iteration]
        model_path = run_config.get("model_path", "")

        t = time.perf_counter()
        model_info = generate_model_info(
            model=model,
            merge_history=merge_history,
            merge=merge,
            iteration=actual_iteration,
            model_path=model_path,
            tokenizer_name=runtime_cast(str, config.tokenizer_name),
            config_dict=config.model_dump(mode="json"),
            wandb_run_path=cluster_run_path,
        )
        logger.info(
            "🟢 cluster-dashboard generated model info run=%s duration_s=%.3f",
            cluster_run_path,
            time.perf_counter() - t,
        )

        serializable = _dashboard_to_serializable(dashboard_data)

        total_duration = time.perf_counter() - overall_start
        logger.info(
            "🟢 cluster-dashboard compute complete run=%s duration_s=%.3f",
            cluster_run_path,
            total_duration,
        )

        return ClusterDashboardResponse(
            clusters=[ClusterDataDTO(**cluster) for cluster in serializable["clusters"]],
            text_samples=[TextSampleDTO(**sample) for sample in serializable["text_samples"]],
            activation_batch=ActivationBatchDTO(**serializable["activation_batch"]),
            activations_map=serializable["activations_map"],
            coactivations=coactivations.tolist(),
            cluster_indices=cluster_indices,
            model_info=model_info,
            iteration=actual_iteration,
            run_id=cluster_run_id,
            cluster_run_path=cluster_run_path,
        )

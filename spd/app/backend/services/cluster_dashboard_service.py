"""Service for computing cluster dashboard data on demand."""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from spd.app.backend.services.run_context_service import ClusteringShape, RunContextService
from spd.clustering.dashboard.compute_max_act import compute_max_activations
from spd.clustering.dashboard.core import (
    ActivationSampleBatch,
    BinnedData,
    ClusterData,
    TextSample,
    TextSampleHash,
)
from spd.clustering.dashboard.dashboard_io import (
    generate_model_info,
    load_wandb_artifacts,
    setup_model_and_data,
)
from spd.log import logger
from spd.settings import SPD_CACHE_DIR
from spd.utils.general_utils import runtime_cast


class ClusterIdDTO(BaseModel):
    clustering_run: str
    iteration: int
    cluster_label: int
    hash: str


class HistogramDTO(BaseModel):
    bin_edges: list[float]
    bin_counts: list[int]


class TokenActivationStatDTO(BaseModel):
    token: str
    count: int


class TokenActivationsDTO(BaseModel):
    top_tokens: list[TokenActivationStatDTO]
    total_unique_tokens: int
    total_activations: int
    entropy: float
    concentration_ratio: float
    activation_threshold: float


class ClusterComponentDTO(BaseModel):
    module: str
    index: int
    label: str


class ClusterStatsDTO(BaseModel):
    all_activations: HistogramDTO
    max_activation_position: HistogramDTO
    n_samples: int
    n_tokens: int
    mean_activation: float
    min_activation: float
    max_activation: float
    median_activation: float
    token_activations: TokenActivationsDTO


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
    cluster_id: ClusterIdDTO
    text_hashes: list[str]
    activations: list[list[float]]
    tokens: list[list[str]] | None


class ClusterDashboardResponse(BaseModel):
    clusters: list[ClusterDataDTO]
    text_samples: list[TextSampleDTO]
    activation_batch: ActivationBatchDTO
    activations_map: dict[str, int]
    model_info: dict[str, Any]
    iteration: int
    run_path: str


def _cluster_stats_to_dto(stats: dict[str, Any]) -> ClusterStatsDTO:
    all_activations = runtime_cast(BinnedData, stats["all_activations"])
    all_acts_dto = HistogramDTO(
        bin_edges=list(all_activations.bin_edges),
        bin_counts=list(all_activations.bin_counts),
    )

    max_activation_position = runtime_cast(BinnedData, stats["max_activation_position"])
    max_activation_position_dto = HistogramDTO(
        bin_edges=list(max_activation_position.bin_edges),
        bin_counts=list(max_activation_position.bin_counts),
    )

    return ClusterStatsDTO(
        all_activations=all_acts_dto,
        max_activation_position=max_activation_position_dto,
        n_samples=runtime_cast(int, stats["n_samples"]),
        n_tokens=runtime_cast(int, stats["n_tokens"]),
        mean_activation=runtime_cast(float, stats["mean_activation"]),
        min_activation=runtime_cast(float, stats["min_activation"]),
        max_activation=runtime_cast(float, stats["max_activation"]),
        median_activation=runtime_cast(float, stats["median_activation"]),
        token_activations=TokenActivationsDTO(**stats["token_activations"]),
    )


def _cluster_to_dto(cluster: ClusterData) -> ClusterDataDTO:
    """Map a domain cluster object directly into its DTO."""

    components = [
        ClusterComponentDTO(module=component.module, index=component.index, label=component.label)
        for component in cluster.components
    ]
    criterion_samples = {
        str(criterion): [str(sample_hash) for sample_hash in hashes]
        for criterion, hashes in cluster.criterion_samples.items()
    }
    return ClusterDataDTO(
        cluster_hash=str(cluster.cluster_hash),
        components=components,
        criterion_samples=criterion_samples,
        stats=_cluster_stats_to_dto(cluster.stats),
    )


def _text_sample_to_dto(text_hash: TextSampleHash, sample: TextSample) -> TextSampleDTO:
    """Convert a TextSample into its DTO counterpart."""

    return TextSampleDTO(
        text_hash=str(text_hash),
        full_text=sample.full_text,
        tokens=sample.tokens,
    )


def _activation_batch_to_dto(batch: ActivationSampleBatch) -> ActivationBatchDTO:
    """Convert activation batches without intermediate dict representations."""

    cluster_id = batch.cluster_id
    return ActivationBatchDTO(
        cluster_id=ClusterIdDTO(
            clustering_run=cluster_id.clustering_run,
            iteration=cluster_id.iteration,
            cluster_label=int(cluster_id.cluster_label),
            hash=str(cluster_id.to_string()),
        ),
        text_hashes=[str(text_hash) for text_hash in batch.text_hashes],
        activations=batch.activations.tolist(),
        tokens=batch.tokens,
    )


class ClusterDashboardService:
    """Compute dashboard data using the loaded run context."""

    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self._inflight: dict[str, asyncio.Task[ClusterDashboardResponse]] = {}

    @staticmethod
    def _cache_path(
        cluster_run_wandb_path: str,
        iteration: int,
        n_samples: int,
        n_batches: int,
        batch_size: int,
        context_length: int,
    ) -> Path:
        safe_cluster = cluster_run_wandb_path.replace("/", "-")
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
    ) -> ClusterDashboardResponse:
        assert (ctx := self.run_context_service.cluster_run_context) is not None, (
            "Run context not found"
        )

        logger.info(
            f"Getting dashboard data run={ctx.wandb_path} requested={iteration} n_samples={n_samples} "
            f"n_batches={n_batches} batch_size={batch_size} context_length={context_length}",
        )

        cache_path = self._cache_path(
            cluster_run_wandb_path=ctx.wandb_path,
            iteration=iteration,
            n_samples=n_samples,
            n_batches=n_batches,
            batch_size=batch_size,
            context_length=context_length,
        )

        task_key = str(cache_path)

        existing = self._inflight.get(task_key)
        if existing:
            return await existing

        loop = asyncio.get_running_loop()

        async def _produce() -> ClusterDashboardResponse:
            if cache_path.exists():
                logger.info(f"Loading cached dashboard data from {cache_path}")
                return self._load_cache(cache_path)

            logger.info(f"Computing dashboard data for {ctx.wandb_path} iteration={iteration}")
            result = await loop.run_in_executor(
                None,
                lambda: self._compute_dashboard_data(
                    cluster_run_wandb_path=ctx.wandb_path,
                    iteration=iteration,
                    n_samples=n_samples,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    context_length=context_length,
                ),
            )
            logger.info(f"Writing cached dashboard data to {cache_path}")
            self._write_cache(cache_path, result)
            return result

        task: asyncio.Task[ClusterDashboardResponse] = asyncio.create_task(_produce())
        self._inflight[task_key] = task
        task.add_done_callback(lambda _: self._inflight.pop(task_key, None))
        return await task

    def _compute_dashboard_data(
        self,
        cluster_run_wandb_path: str,
        iteration: int,
        n_samples: int,
        n_batches: int,
        batch_size: int,
        context_length: int,
    ) -> ClusterDashboardResponse:
        merge_history, run_config = load_wandb_artifacts(cluster_run_wandb_path)

        model, tokenizer, dataloader, config = setup_model_and_data(
            run_config=run_config,
            context_length=context_length,
            batch_size=batch_size,
        )
        model.eval()
        cluster_run_id = cluster_run_wandb_path.split("/")[-1]

        dashboard_data, _, _ = compute_max_activations(
            model=model,
            sigmoid_type=config.sigmoid_type,
            tokenizer=tokenizer,
            dataloader=dataloader,
            merge_history=merge_history,
            iteration=iteration,
            n_samples=n_samples,
            n_batches=n_batches,
            clustering_run=cluster_run_id,
        )

        merge = merge_history.merges[iteration]
        model_path = run_config.get("model_path", "")

        model_info = generate_model_info(
            model=model,
            merge_history=merge_history,
            merge=merge,
            iteration=iteration,
            model_path=model_path,
            tokenizer_name=runtime_cast(str, config.tokenizer_name),
            config_dict=config.model_dump(mode="json"),
            wandb_run_path=cluster_run_wandb_path,
        )
        clusters = [_cluster_to_dto(cluster) for cluster in dashboard_data.clusters.values()]
        text_samples = [
            _text_sample_to_dto(text_hash, sample)
            for text_hash, sample in dashboard_data.text_samples.items()
        ]
        activation_batch = _activation_batch_to_dto(dashboard_data.activations)
        activations_map = {str(key): value for key, value in dashboard_data.activations_map.items()}

        module_cluster_map: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for cluster_hash, cluster in dashboard_data.clusters.items():
            for component in cluster.components:
                module_cluster_map[component.module][cluster_hash].append(component.index)

        return ClusterDashboardResponse(
            clusters=clusters,
            text_samples=text_samples,
            activation_batch=activation_batch,
            activations_map=activations_map,
            model_info=model_info,
            iteration=iteration,
            run_path=cluster_run_wandb_path,
        )

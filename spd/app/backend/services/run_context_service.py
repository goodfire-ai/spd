import pickle
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.app.backend.api import (
    AvailablePrompt,
    ClusteringShape,
    ClusterRunDTO,
    Run,
    Status,
    TrainRunDTO,
)
from spd.clustering.dashboard.dashboard_io import load_wandb_artifacts
from spd.clustering.merge_history import MergeHistory
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

ENTITY = "goodfire"
TRAIN_PROJECT = "spd"
CLUSTER_PROJECT = "spd-cluster"
DEVICE = get_device()


@dataclass
class ActivatingExamplesArgs:
    iteration: int
    n_samples: int
    n_batches: int
    batch_size: int
    context_length: int


@dataclass
class TrainRunContext:
    run: wandb.Run
    wandb_id: str
    wandb_path: str
    config: Config
    cm: ComponentModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader[Any]
    available_cluster_runs: list[str]


@dataclass
class ClusterRunContext:
    wandb_path: str
    iteration: int
    clustering_shape: ClusteringShape

    def to_dto(self) -> ClusterRunDTO:
        return ClusterRunDTO(
            wandb_path=self.wandb_path,
            iteration=self.iteration,
            clustering_shape=self.clustering_shape,
        )


class RunContextService:
    def __init__(self):
        self.train_run_context: TrainRunContext | None = None
        self.cluster_run_context: ClusterRunContext | None = None
        self.api = wandb.Api()

    def get_runs(self) -> list[Run]:
        return [Run(id=run.id, url=run.url) for run in self.api.runs(TRAIN_PROJECT)]

    @staticmethod
    def _parse_run_path(path: str) -> tuple[str, str, str]:
        parts = path.split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[-1]
        raise ValueError(f"Invalid WandB path: {path}")

    def _discover_cluster_runs(self, training_run: wandb.Run) -> list[str]:
        model_tag = f"model:{training_run.id}"
        cluster_project = f"{ENTITY}/{CLUSTER_PROJECT}"
        runs: list[str] = []
        logger.info(f"Discovering cluster runs for {training_run.id}")
        for run in self.api.runs(cluster_project, filters={"tags": {"$in": [model_tag]}}):
            if model_tag in run.tags:
                runs.append("/".join(runtime_cast(list, run.path)))
        logger.info(
            f"Found {len(runs)} clustering runs for {training_run.id}: [{', '.join(runs[:3])}...]"
        )
        return sorted(set(runs))

    def _get_cluster_data(
        self,
        wandb_path: str,
        iteration: int,
        train_ctx: TrainRunContext,
    ):
        merge_history, _ = load_wandb_artifacts(wandb_path)
        unique_cluster_indices = merge_history.get_unique_clusters(iteration)

        # todo: validate this these are actually subcomponents, not components
        subcomponents_by_cluster_id: dict[int, list[MergeHistory.ClusterComponentInfo]] = {
            cid: merge_history.get_cluster_components_info(iteration, cid)
            for cid in unique_cluster_indices
        }

        module_cluster_map: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for cluster_id, cluster_subcomponents in subcomponents_by_cluster_id.items():
            for component in cluster_subcomponents:
                module_cluster_map[component.module][cluster_id].append(component.index)

        module_component_assignments: dict[str, list[int]] = {}
        module_component_groups: dict[str, list[list[int]]] = {}

        for module_name, components_module in train_ctx.cm.components.items():
            C = components_module.C
            assignments = [-1] * C
            groups: list[list[int]] = []

            cluster_groups = module_cluster_map.get(module_name, {})
            for indices in sorted(cluster_groups.values()):
                if any(int(i) < 0 or int(i) >= C for i in indices):
                    raise ValueError(f"Invalid indices: {indices}")
                unique_indices = sorted({int(i) for i in indices})
                component_idx = len(groups)
                for sub_idx in unique_indices:
                    assignments[sub_idx] = component_idx
                groups.append(unique_indices)

            unassigned = [sub_idx for sub_idx in range(C) if assignments[sub_idx] == -1]
            for sub_idx in unassigned:
                assignments[sub_idx] = len(groups)
                groups.append([sub_idx])

            module_component_assignments[module_name] = assignments
            module_component_groups[module_name] = groups

        return ClusteringShape(
            module_component_assignments=module_component_assignments,
            module_component_groups=module_component_groups,
        )

    def get_status(self) -> Status:
        if (train_ctx := self.train_run_context) is None:
            logger.info("No train run context found")
            return Status(
                train_run=None,
                cluster_run=None,
            )

        train_run = TrainRunDTO(
            wandb_path=train_ctx.wandb_path,
            component_layers=list(train_ctx.cm.components.keys()),
            available_cluster_runs=train_ctx.available_cluster_runs,
        )

        if (cluster_ctx := self.cluster_run_context) is None:
            logger.info("No cluster run context found")
            return Status(
                train_run=train_run,
                cluster_run=None,
            )

        cluster_run = cluster_ctx.to_dto()

        return Status(
            train_run=train_run,
            cluster_run=cluster_run,
        )

    def load_run(self, wandb_path: str):
        # self.train_run_context = get_pickle_cached(
        #     key=wandb_path.replace("/", "-"),
        #     get_func=lambda: self._load_run_from_wandb_path(wandb_path),
        # )
        self.train_run_context = self._load_run_from_wandb_path(wandb_path)
        logger.info(f"Loaded run from wandb id: {wandb_path}")

    def _load_run_from_wandb_path(self, wandb_id: str):
        logger.info(f"Loading run from wandb id: {wandb_id}")

        wandb_run: wandb.Run = self.api.run(f"{ENTITY}/{TRAIN_PROJECT}/{wandb_id}")
        if wandb_run is None:
            raise ValueError(f"WandB run not found for id {wandb_id}")

        wandb_path = "/".join(wandb_run.path)
        model_path = f"wandb:{wandb_run.entity}/{wandb_run.project}/runs/{wandb_run.id}"

        run_info = SPDRunInfo.from_path(model_path)

        task_config = runtime_cast(LMTaskConfig, run_info.config.task_config)

        train_data_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=run_info.config.tokenizer_name,
            split=task_config.train_data_split,
            n_ctx=task_config.max_seq_len,
            is_tokenized=task_config.is_tokenized,
            streaming=task_config.streaming,
            column_name=task_config.column_name,
            shuffle_each_epoch=task_config.shuffle_each_epoch,
            seed=None,
        )

        batch_size = 1

        logger.info("Creating train loader from run info")
        train_loader, tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=run_info.config.seed,
            ddp_rank=0,
            ddp_world_size=0,
        )

        logger.info("Creating component model from run info")
        cm = ComponentModel.from_run_info(run_info)
        cm.to(DEVICE)

        available_cluster_runs = self._discover_cluster_runs(wandb_run)

        return TrainRunContext(
            run=wandb_run,
            wandb_id=wandb_id,
            wandb_path=wandb_path,
            config=run_info.config,
            cm=cm,
            tokenizer=tokenizer,
            train_loader=train_loader,
            available_cluster_runs=available_cluster_runs,
        )

    def load_cluster_run(self, wandb_path: str, iteration: int):
        logger.info(f"Loading cluster run from wandb path: {wandb_path}")
        # self.cluster_run_context = get_pickle_cached(
        #     key=f"{wandb_path.replace('/', '-')}-{iteration}",
        #     get_func=lambda: self._load_cluster_run_from_wandb_path(wandb_path, iteration),
        # )
        self.cluster_run_context = self._load_cluster_run_from_wandb_path(wandb_path, iteration)
        logger.info(f"Loaded cluster run from wandb path: {wandb_path}")

    def _load_cluster_run_from_wandb_path(self, wandb_path: str, iteration: int):
        assert (train_ctx := self.train_run_context) is not None
        # assert (cluster_run_ctx := self.cluster_run_context) is not None

        logger.info(f"Loading cluster run from wandb path: {wandb_path}")
        # wandb_run = runtime_cast(Run, self.api.run(wandb_path))

        return ClusterRunContext(
            wandb_path=wandb_path,
            iteration=iteration,
            clustering_shape=self._get_cluster_data(wandb_path, iteration, train_ctx),
        )

    def get_available_prompts(self) -> list[AvailablePrompt]:
        """Get first 100 prompts from the dataset with their indices and text."""
        assert (ctx := self.train_run_context) is not None, "Run context not found"

        prompts = []
        for idx in range(min(100, len(ctx.train_loader.dataset))):  # pyright: ignore[reportArgumentType]
            example = ctx.train_loader.dataset[idx]["input_ids"]
            assert isinstance(example, torch.Tensor)
            assert example.ndim == 1, "Example must be 1D (seq_len)"

            # Decode to text for display
            text = ctx.tokenizer.decode(example, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]

            prompts.append(AvailablePrompt(index=idx, full_text=text))

        return prompts


CACHE_DIR = Path(".data/cache")


def get_pickle_cached(key: str, get_func: Callable[[], Any]) -> Any:
    # Ensure the cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    file = CACHE_DIR / f"{key}.pkl"

    if file.exists():
        logger.info(f"cache hit for {key}")
        with open(file, "rb") as f:
            logger.info(f"loading from cache for {key}")
            res = pickle.load(f)
        logger.info(f"loaded from cache for {key}")
        return res

    logger.info(f"cache miss for {key}")
    res = get_func()
    logger.info(f"saving to cache for {key}")
    with open(file, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"saved to cache for {key}")
    return res

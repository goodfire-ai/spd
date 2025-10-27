from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from spd.app.backend.schemas import Status, TrainRun
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

DEVICE = get_device()


@dataclass
class TrainRunContext:
    wandb_id: str
    wandb_path: str
    config: Config
    cm: ComponentModel
    tokenizer: Tokenizer
    train_loader: DataLoader[Any]


class RunContextService:
    def __init__(self):
        self.train_run_context: TrainRunContext | None = None

    def get_status(self) -> Status:
        if (train_ctx := self.train_run_context) is None:
            return Status(train_run=None)

        train_run = TrainRun(
            wandb_path=train_ctx.wandb_path,
            config=train_ctx.config.model_dump(),
        )

        return Status(train_run=train_run)

    def load_run(self, wandb_path: str):
        self.train_run_context = self._load_run_from_wandb_path(wandb_path)
        logger.info(f"Loaded run from wandb path: {wandb_path}")

    def _load_run_from_wandb_path(self, wandb_path: str):
        logger.info(f"Loading run from wandb path: {wandb_path}")

        run_info = SPDRunInfo.from_path(f"wandb:{wandb_path}")

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

        logger.info("Creating train loader from run info")
        train_loader, tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=1,  # NOTE: we use a batch size of 1 for simplicity and because almost always just want a single example
            buffer_size=task_config.buffer_size,
            global_seed=run_info.config.seed,
            ddp_rank=0,
            ddp_world_size=0,
        )

        logger.info("Creating component model from run info")
        cm = ComponentModel.from_run_info(run_info)
        cm.to(DEVICE)
        logger.info(f"Component model created on device: {DEVICE}")

        return TrainRunContext(
            wandb_id=wandb_path,
            wandb_path=wandb_path,
            config=run_info.config,
            cm=cm,
            tokenizer=tokenizer,
            train_loader=train_loader,
        )

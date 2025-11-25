from dataclasses import dataclass
from typing import Any

import yaml
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.app.backend.schemas import Status, TrainRun
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

DEVICE = get_device()

# Tokenizer name -> decode strategy
# "wordpiece": ## = continuation (strip ##), punctuation = no space, others = space prefix
# "bpe": spaces encoded in token via Ġ, just decode directly
TOKENIZER_STRATEGIES: dict[str, str] = {
    "SimpleStories/test-SimpleStories-gpt2-1.25M": "wordpiece",
    "openai-community/gpt2": "bpe",
}

# Characters that don't get a space prefix in wordpiece
_PUNCT_NO_SPACE = set(".,!?;:'\")-]}>/")


def _build_token_lookup(
    tokenizer: PreTrainedTokenizer,
    tokenizer_name: str | None,
) -> dict[int, str]:
    """Build token ID -> string lookup.

    Uses tokenizer-specific strategy to produce strings that concatenate correctly.
    """
    strategy = TOKENIZER_STRATEGIES.get(tokenizer_name or "", "bpe")
    lookup: dict[int, str] = {}

    for tid in range(tokenizer.vocab_size):  # pyright: ignore[reportAttributeAccessIssue]
        decoded = tokenizer.decode([tid], skip_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]

        if strategy == "wordpiece":
            # WordPiece handling:
            if decoded.startswith("##"):
                # Continuation token - strip ## prefix, no space
                lookup[tid] = decoded[2:]
            elif decoded and decoded[0] in _PUNCT_NO_SPACE:
                # Punctuation - no space prefix
                lookup[tid] = decoded
            else:
                # Regular token - add space prefix
                lookup[tid] = " " + decoded
        else:
            # BPE (GPT-2 style): spaces encoded in token via Ġ -> space
            lookup[tid] = decoded

    return lookup


@dataclass
class TrainRunContext:
    wandb_id: str
    wandb_path: str
    config: Config
    cm: ComponentModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader[Any]
    token_strings: dict[int, str]  # Pre-built lookup for fast token stringification


class RunContextService:
    def __init__(self):
        self.train_run_context: TrainRunContext | None = None

    def get_status(self) -> Status:
        if (train_ctx := self.train_run_context) is None:
            return Status(train_run=None)

        config_yaml = yaml.dump(
            train_ctx.config.model_dump(), default_flow_style=False, sort_keys=False
        )

        train_run = TrainRun(
            wandb_path=train_ctx.wandb_path,
            config_yaml=config_yaml,
        )

        return Status(train_run=train_run)

    def load_run(self, wandb_path: str):
        logger.info(f"Loading run from wandb path: {wandb_path}")
        self.train_run_context = self._load_run_from_wandb_path(wandb_path)
        logger.info(f"Loaded run from wandb path: {wandb_path}")

    def _load_run_from_wandb_path(self, wandb_path: str):
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

        # Pre-build token string lookup for fast stringification
        vocab_size: int = tokenizer.vocab_size  # pyright: ignore[reportAttributeAccessIssue]
        logger.info(f"Building token lookup table for vocab size {vocab_size}")
        token_strings = _build_token_lookup(tokenizer, run_info.config.tokenizer_name)

        return TrainRunContext(
            wandb_id=wandb_path,
            wandb_path=wandb_path,
            config=run_info.config,
            cm=cm,
            tokenizer=tokenizer,
            train_loader=train_loader,
            token_strings=token_strings,
        )

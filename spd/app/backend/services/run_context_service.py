from dataclasses import dataclass
from typing import Any

import yaml
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.schemas import Status, TrainRun
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

DEVICE = get_device()

# Characters that don't get a space prefix in wordpiece
_PUNCT_NO_SPACE = set(".,!?;:'\")-]}>/")


def _build_token_lookup(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name: str,
) -> dict[int, str]:
    """Build token ID -> string lookup.

    Uses tokenizer-specific strategy to produce strings that concatenate correctly.
    """
    lookup: dict[int, str] = {}
    vocab_size: int = tokenizer.vocab_size  # pyright: ignore[reportAssignmentType]

    for tid in range(vocab_size):
        decoded: str = tokenizer.decode([tid], skip_special_tokens=False)

        # Tokenizer name -> decode strategy
        # "wordpiece": ## = continuation (strip ##), punctuation = no space, others = space prefix
        # "bpe": spaces encoded in token via Ġ, just decode directly
        match tokenizer_name:
            case "SimpleStories/test-SimpleStories-gpt2-1.25M":
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
            case "openai-community/gpt2":
                # BPE (GPT-2 style): spaces encoded in token via Ġ -> space
                lookup[tid] = decoded
            case _:
                raise ValueError(f"Unsupported tokenizer name: {tokenizer_name}")

    return lookup


@dataclass
class TrainRunContext:
    wandb_id: str
    wandb_path: str
    config: Config
    cm: ComponentModel
    tokenizer: PreTrainedTokenizerBase
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
        )

        logger.info("Creating component model from run info")
        cm = ComponentModel.from_run_info(run_info)
        cm.to(DEVICE)
        logger.info(f"Component model created on device: {DEVICE}")

        # Pre-build token string lookup for fast stringification
        # Cast tokenizer to base type (create_data_loader returns PreTrainedTokenizer)
        tokenizer_base = runtime_cast(PreTrainedTokenizerBase, tokenizer)
        vocab_size: int = runtime_cast(int, tokenizer_base.vocab_size)
        logger.info(f"Building token lookup table for vocab size {vocab_size}")
        tokenizer_name = run_info.config.tokenizer_name
        assert tokenizer_name is not None
        token_strings = _build_token_lookup(tokenizer_base, tokenizer_name)

        return TrainRunContext(
            wandb_id=wandb_path,
            wandb_path=wandb_path,
            config=run_info.config,
            cm=cm,
            tokenizer=tokenizer_base,
            train_loader=train_loader,
            token_strings=token_strings,
        )

from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import runtime_cast


@dataclass
class RunContext:
    wandb_id: str
    config: Config
    cm: ComponentModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader[Any]


class Status(BaseModel):
    loaded: bool
    run_id: str | None
    prompt: str | None


class AvailablePrompt(BaseModel):
    index: int
    text: str
    full_text: str


class RunContextService:
    def __init__(self):
        self.run_context: RunContext | None = None

    def get_status(self) -> Status:
        if self.run_context is None:
            return Status(
                loaded=False,
                run_id=None,
                prompt=None,
            )
        return Status(
            loaded=True,
            run_id=self.run_context.wandb_id,
            prompt=None,  # No "current" prompt anymore
        )

    def load_run_from_wandb_id(self, wandb_id: str):
        path = f"wandb:goodfire/spd/runs/{wandb_id}"
        run_info = SPDRunInfo.from_path(path)

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

        train_loader, tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=run_info.config.seed,
            ddp_rank=0,
            ddp_world_size=0,
        )

        self.run_context = RunContext(
            wandb_id=wandb_id,
            config=run_info.config,
            cm=ComponentModel.from_run_info(run_info),
            tokenizer=tokenizer,
            train_loader=train_loader,
        )

    def get_available_prompts(self) -> list[AvailablePrompt]:
        """Get first 100 prompts from the dataset with their indices and text."""
        assert (ctx := self.run_context) is not None, "Run context not found"

        prompts = []
        for idx in range(min(100, len(ctx.train_loader.dataset))):  # pyright: ignore[reportArgumentType]
            example = ctx.train_loader.dataset[idx]["input_ids"]
            assert isinstance(example, torch.Tensor)
            assert example.ndim == 1, "Example must be 1D (seq_len)"

            # Decode to text for display
            text = ctx.tokenizer.decode(example, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]

            prompts.append(
                AvailablePrompt(
                    index=idx,
                    text=text[:200] + "..." if len(text) > 200 else text,  # Truncate for display
                    full_text=text,
                )
            )

        return prompts

from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

from spd.base_config import BaseConfig


class LMTaskConfig(BaseConfig):
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for training",
    )
    eval_dataset_name: str = Field(
        default="",
        description="HuggingFace dataset identifier for evaluation. Defaults to dataset_name.",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text/tokens",
    )
    eval_column_name: str = Field(
        default="",
        description="Dataset column for evaluation. Defaults to column_name.",
    )

    @model_validator(mode="before")
    @classmethod
    def set_eval_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Set eval fields to their training counterparts if not specified."""
        if data.get("eval_dataset_name") in (None, ""):
            data["eval_dataset_name"] = data.get("dataset_name", "lennart-finke/SimpleStories")
        if data.get("eval_column_name") in (None, ""):
            data["eval_column_name"] = data.get("column_name", "story")
        return data

    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )
    shuffle_each_epoch: bool = Field(
        default=True,
        description="Whether to reshuffle data at each epoch. Set False in tests to keep fixed "
        "order across dp modes.",
    )
    is_tokenized: bool = Field(
        default=False,
        description="Whether the dataset is already tokenized",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use a streaming dataset for training",
    )
    eval_streaming: bool = Field(
        default=False,
        description="Whether to use streaming for eval dataset.",
    )

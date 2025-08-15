from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class LMTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
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
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
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
        description="Whether to use a streaming dataset",
    )

"""Strategy dispatch: routes AutointerpConfig variants to their implementations."""

from typing import Any

from openrouter.components import Reasoning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.autointerp.config import AutointerpConfig, CompactSkepticalConfig
from spd.autointerp.schemas import ArchitectureInfo
from spd.autointerp.strategies.compact_skeptical import (
    INTERPRETATION_SCHEMA,
)
from spd.autointerp.strategies.compact_skeptical import (
    format_prompt as compact_skeptical_prompt,
)
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def format_prompt(
    config: AutointerpConfig,
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    ci_threshold: float,
) -> str:
    match config:
        case CompactSkepticalConfig():
            return compact_skeptical_prompt(
                config,
                component,
                arch,
                tokenizer,
                input_token_stats,
                output_token_stats,
                ci_threshold,
            )
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise AssertionError(f"Unhandled config type: {type(config)}")  # pyright: ignore[reportUnreachable]


def get_model(config: AutointerpConfig) -> str:
    return config.model


def get_reasoning(config: AutointerpConfig) -> Reasoning | None:
    match config:
        case CompactSkepticalConfig():
            if config.reasoning_effort is None:
                return None
            return Reasoning(effort=config.reasoning_effort.value)
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise AssertionError(f"Unhandled config type: {type(config)}")  # pyright: ignore[reportUnreachable]


def get_response_schema(config: AutointerpConfig) -> dict[str, Any]:
    match config:
        case CompactSkepticalConfig():
            return INTERPRETATION_SCHEMA
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise AssertionError(f"Unhandled config type: {type(config)}")  # pyright: ignore[reportUnreachable]

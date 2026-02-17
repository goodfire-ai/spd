"""Strategy dispatch: routes AutointerpConfig variants to their implementations."""

from typing import Any

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.schemas import ModelMetadata
from spd.autointerp.strategies.compact_skeptical import (
    INTERPRETATION_SCHEMA,
)
from spd.autointerp.strategies.compact_skeptical import (
    format_prompt as compact_skeptical_prompt,
)
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def format_prompt(
    strategy: CompactSkepticalConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> str:
    match strategy:
        case CompactSkepticalConfig():
            return compact_skeptical_prompt(
                strategy,
                component,
                model_metadata,
                app_tok,
                input_token_stats,
                output_token_stats,
            )
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise AssertionError(f"Unhandled strategy type: {type(strategy)}")  # pyright: ignore[reportUnreachable]


def get_response_schema(strategy: CompactSkepticalConfig) -> dict[str, Any]:
    match strategy:
        case CompactSkepticalConfig():
            return INTERPRETATION_SCHEMA
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise AssertionError(f"Unhandled strategy type: {type(strategy)}")  # pyright: ignore[reportUnreachable]

"""Strategy dispatch: routes AutointerpConfig variants to their implementations."""

from typing import Any

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CompactSkepticalConfig, DualViewConfig, StrategyConfig
from spd.autointerp.schemas import ModelMetadata
from spd.autointerp.strategies.compact_skeptical import (
    INTERPRETATION_SCHEMA as COMPACT_SKEPTICAL_SCHEMA,
)
from spd.autointerp.strategies.compact_skeptical import (
    format_prompt as compact_skeptical_prompt,
)
from spd.autointerp.strategies.dual_view import (
    INTERPRETATION_SCHEMA as DUAL_VIEW_SCHEMA,
)
from spd.autointerp.strategies.dual_view import (
    format_prompt as dual_view_prompt,
)
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def format_prompt(
    strategy: StrategyConfig,
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
        case DualViewConfig():
            return dual_view_prompt(
                strategy,
                component,
                model_metadata,
                app_tok,
                input_token_stats,
                output_token_stats,
            )


def get_response_schema(strategy: StrategyConfig) -> dict[str, Any]:
    match strategy:
        case CompactSkepticalConfig():
            return COMPACT_SKEPTICAL_SCHEMA
        case DualViewConfig():
            return DUAL_VIEW_SCHEMA

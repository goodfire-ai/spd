"""CLI for label-based scoring (detection, fuzzing).

Usage:
    python -m spd.autointerp.scoring.scripts.run_label_scoring <decomposition_id> --config_json '...' --harvest_subrun_id h-20260211_120000
"""

import os
from typing import Any, Literal

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.config import AnthropicBatchConfig, AutointerpEvalConfig, OpenRouterConfig
from spd.autointerp.repo import InterpRepo
from spd.autointerp.scoring.detection import run_detection_scoring
from spd.autointerp.scoring.fuzzing import run_fuzzing_scoring
from spd.harvest.repo import HarvestRepo

LabelScorerType = Literal["detection", "fuzzing"]


def _get_api_key(backend: AnthropicBatchConfig | OpenRouterConfig) -> str:
    match backend:
        case AnthropicBatchConfig():
            key = os.environ.get("ANTHROPIC_API_KEY")
            assert key, "ANTHROPIC_API_KEY not set"
            return key
        case OpenRouterConfig():
            key = os.environ.get("OPENROUTER_API_KEY")
            assert key, "OPENROUTER_API_KEY not set"
            return key


def main(
    decomposition_id: str,
    scorer_type: LabelScorerType,
    config_json: dict[str, Any],
    harvest_subrun_id: str | None = None,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    load_dotenv()

    config = AutointerpEvalConfig.model_validate(config_json)
    api_key = _get_api_key(config.backend)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    interp_repo = InterpRepo.open(decomposition_id)
    assert interp_repo is not None, (
        f"No autointerp data for {decomposition_id}. Run autointerp first."
    )

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(
            decomposition_id=decomposition_id,
            subrun_id=harvest_subrun_id,
            readonly=False,
        )
    else:
        harvest = HarvestRepo.open_most_recent(decomposition_id, readonly=False)
        assert harvest is not None, f"No harvest data for {decomposition_id}"

    # Scoring evals need all components (detection samples non-activating examples
    # from other components). This is slow for large runs but unavoidable for now.
    logger.info("Loading all components from harvest (required for scoring)...")
    components = harvest.get_all_components()
    logger.info(f"Loaded {len(components)} components")

    match scorer_type:
        case "detection":
            run_detection_scoring(
                components=components,
                interp_repo=interp_repo,
                backend=config.backend,
                api_key=api_key,
                tokenizer_name=tokenizer_name,
                config=config.detection_config,
                limit=config.limit,
                cost_limit_usd=config.cost_limit_usd,
            )
        case "fuzzing":
            run_fuzzing_scoring(
                components=components,
                interp_repo=interp_repo,
                backend=config.backend,
                api_key=api_key,
                tokenizer_name=tokenizer_name,
                config=config.fuzzing_config,
                limit=config.limit,
                cost_limit_usd=config.cost_limit_usd,
            )


def get_command(
    decomposition_id: str,
    scorer_type: LabelScorerType,
    config: AutointerpEvalConfig,
    harvest_subrun_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        f"python -m spd.autointerp.scoring.scripts.run_label_scoring "
        f"--decomposition_id {decomposition_id} "
        f"--scorer_type {scorer_type} "
        f"--config_json '{config_json}' "
    )
    if harvest_subrun_id is not None:
        cmd += f" --harvest_subrun_id {harvest_subrun_id} "
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)

"""CLI for autointerp pipeline.

Usage:
    python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '...'
    spd-autointerp <wandb_path>  # SLURM submission
"""

import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.config import AnthropicBatchConfig, AutointerpConfig, OpenRouterConfig
from spd.autointerp.interpret import run_interpret
from spd.autointerp.schemas import get_autointerp_subrun_dir
from spd.harvest.repo import HarvestRepo
from spd.log import logger


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
    config_json: dict[str, Any],
    harvest_subrun_id: str | None = None,
) -> None:
    assert isinstance(config_json, dict), f"Expected dict from fire, got {type(config_json)}"
    interp_config = AutointerpConfig.model_validate(config_json)

    load_dotenv()
    api_key = _get_api_key(interp_config.backend)

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=False)
    else:
        harvest = HarvestRepo.open_most_recent(decomposition_id, readonly=False)
        if harvest is None:
            raise ValueError(f"No harvest data found for {decomposition_id}")

    autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    subrun_dir = get_autointerp_subrun_dir(decomposition_id, autointerp_run_id)
    subrun_dir.mkdir(parents=True, exist_ok=True)

    interp_config.to_file(subrun_dir / "config.yaml")

    db_path = subrun_dir / "interp.db"

    logger.info(f"Autointerp run: {subrun_dir}")

    adapter = adapter_from_id(decomposition_id)

    run_interpret(
        api_key=api_key,
        backend=interp_config.backend,
        limit=interp_config.limit,
        cost_limit_usd=interp_config.cost_limit_usd,
        model_metadata=adapter.model_metadata,
        template_strategy=interp_config.template_strategy,
        harvest=harvest,
        db_path=db_path,
        tokenizer_name=adapter.tokenizer_name,
    )


def get_command(
    decomposition_id: str,
    config: AutointerpConfig,
    harvest_subrun_id: str | None = None,
) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    cmd = (
        "python -m spd.autointerp.scripts.run_interpret "
        f"--decomposition_id {decomposition_id} "
        f"--config_json '{config_json}' "
    )
    if harvest_subrun_id is not None:
        cmd += f"--harvest_subrun_id {harvest_subrun_id} "
    return cmd


if __name__ == "__main__":
    import fire

    fire.Fire(main)

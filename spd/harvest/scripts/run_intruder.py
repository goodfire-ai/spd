import asyncio
import os

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.intruder import run_intruder_scoring
from spd.harvest.repo import HarvestRepo


def main(
    decomposition_id: str,
    config_json: str,
    harvest_subrun_id: str,
) -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    assert openrouter_api_key, "OPENROUTER_API_KEY not set"

    eval_config = IntruderEvalConfig.from_json_or_dict(config_json)

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name

    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=False)

    components = harvest.get_all_components()

    asyncio.run(
        run_intruder_scoring(
            components=components,
            model=eval_config.model,
            openrouter_api_key=openrouter_api_key,
            tokenizer_name=tokenizer_name,
            harvest=harvest,
            eval_config=eval_config,
            limit=eval_config.limit,
            cost_limit_usd=eval_config.cost_limit_usd,
        )
    )


def get_command(decomposition_id: str, config: IntruderEvalConfig, harvest_subrun_id: str) -> str:
    config_json = config.model_dump_json(exclude_none=True)
    return (
        f"python -m spd.harvest.scripts.run_intruder {decomposition_id} "
        f"--config_json '{config_json}' "
        f"--harvest_subrun_id {harvest_subrun_id}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

import asyncio
import json
from pathlib import Path

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import AutointerpConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.llm_api import (
    JSON_PARSE_RETRIES,
    CostTracker,
    GlobalBackoff,
    LLMClient,
    LLMError,
    LLMJob,
    LLMResult,
    get_model_pricing,
    make_response_format,
    map_api_calls,
)
from spd.autointerp.schemas import ArchitectureInfo, InterpretationResult
from spd.autointerp.strategies.dispatch import format_prompt, get_response_schema
from spd.configs import LMTaskConfig
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo

MAX_CONCURRENT = 50


async def interpret_component(
    llm: LLMClient,
    config: AutointerpConfig,
    component: ComponentData,
    arch: ArchitectureInfo,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    ci_threshold: float,
) -> InterpretationResult:
    """Interpret a single component. Used by the app for on-demand interpretation."""
    prompt = format_prompt(
        config=config,
        component=component,
        arch=arch,
        app_tok=app_tok,
        input_token_stats=input_token_stats,
        output_token_stats=output_token_stats,
        ci_threshold=ci_threshold,
    )

    schema = get_response_schema(config)
    response_format = make_response_format("interpretation", schema)
    raw = ""
    parsed = None
    for attempt in range(JSON_PARSE_RETRIES):
        raw = await llm.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            context_label=component.component_key,
            response_format=response_format,
            reasoning=Reasoning(effort=config.reasoning_effort),
        )
        try:
            parsed = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == JSON_PARSE_RETRIES - 1:
                raise
            logger.warning(
                f"{component.component_key}: invalid JSON "
                f"(attempt {attempt + 1}/{JSON_PARSE_RETRIES}), retrying"
            )
    assert parsed is not None and len(parsed) == 3, f"Expected 3 fields, got {parsed}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning_text = parsed["reasoning"]
    assert (
        isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning_text, str)
    )

    return InterpretationResult(
        component_key=component.component_key,
        label=label,
        confidence=confidence,
        reasoning=reasoning_text,
        raw_response=raw,
        prompt=prompt,
    )


def run_interpret(
    wandb_path: str,
    openrouter_api_key: str,
    config: AutointerpConfig,
    harvest: HarvestRepo,
    db_path: Path,
    limit: int | None,
    cost_limit_usd: float | None = None,
) -> list[InterpretationResult]:
    arch = get_architecture_info(wandb_path)
    components = harvest.get_all_components()

    token_stats = harvest.get_token_stats()
    assert token_stats is not None, "token_stats.pt not found. Run harvest first."
    ci_threshold = harvest.get_activation_threshold()

    app_tok = AppTokenizer.from_pretrained(arch.tokenizer_name)

    eligible = sorted(components, key=lambda c: c.mean_activation, reverse=True)
    if limit is not None:
        eligible = eligible[:limit]

    async def _run() -> list[InterpretationResult]:
        db = InterpDB(db_path)

        completed = db.get_completed_keys()
        if completed:
            logger.info(f"Resuming: {len(completed)} already completed")

        remaining = [c for c in eligible if c.component_key not in completed]
        logger.info(f"Interpreting {len(remaining)} components")

        schema = get_response_schema(config)
        jobs: list[LLMJob] = []
        for component in remaining:
            input_stats = get_input_token_stats(
                token_stats, component.component_key, app_tok, top_k=20
            )
            output_stats = get_output_token_stats(
                token_stats, component.component_key, app_tok, top_k=50
            )
            assert input_stats is not None
            assert output_stats is not None
            prompt = format_prompt(
                config=config,
                component=component,
                arch=arch,
                app_tok=app_tok,
                input_token_stats=input_stats,
                output_token_stats=output_stats,
                ci_threshold=ci_threshold,
            )
            jobs.append(LLMJob(prompt=prompt, schema=schema, key=component.component_key))

        results: list[InterpretationResult] = []
        n_errors = 0

        async with OpenRouter(api_key=openrouter_api_key) as api:
            input_price, output_price = await get_model_pricing(api, config.model)
            cost_tracker = CostTracker(
                input_price_per_token=input_price,
                output_price_per_token=output_price,
                limit_usd=cost_limit_usd,
            )
            llm = LLMClient(
                api=api,
                rate_limiter=AsyncLimiter(max_rate=config.max_requests_per_minute, time_period=60),
                backoff=GlobalBackoff(),
                cost_tracker=cost_tracker,
            )

            async for outcome in map_api_calls(
                jobs,
                llm,
                model=config.model,
                max_tokens=8000,
                reasoning=Reasoning(effort=config.reasoning_effort),
                max_concurrent=MAX_CONCURRENT,
            ):
                match outcome:
                    case LLMResult(job=job, parsed=parsed, raw=raw):
                        assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
                        label = parsed["label"]
                        confidence = parsed["confidence"]
                        reasoning_text = parsed["reasoning"]
                        assert (
                            isinstance(label, str)
                            and isinstance(confidence, str)
                            and isinstance(reasoning_text, str)
                        )
                        result = InterpretationResult(
                            component_key=job.key,
                            label=label,
                            confidence=confidence,
                            reasoning=reasoning_text,
                            raw_response=raw,
                            prompt=job.prompt,
                        )
                        results.append(result)
                        db.save_interpretation(result)
                    case LLMError(job=job, error=e):
                        n_errors += 1
                        logger.error(f"Skipping {job.key}: {type(e).__name__}: {e}")

            logger.info(f"Final cost: ${cost_tracker.cost_usd():.2f}")

        db.close()

        error_rate = n_errors / len(remaining) if remaining else 0.0
        if error_rate > 0.2:
            raise RuntimeError(
                f"Error rate {error_rate:.0%} ({n_errors}/{len(remaining)}) exceeds 20% threshold"
            )

        logger.info(f"Completed {len(results)} interpretations -> {db_path}")
        return results

    return asyncio.run(_run())


def get_architecture_info(wandb_path: str) -> ArchitectureInfo:
    from spd.topology import TransformerTopology

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    topology = TransformerTopology(model.target_model)
    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    assert config.tokenizer_name is not None
    return ArchitectureInfo(
        n_blocks=topology.n_blocks,
        c_per_layer=model.module_to_c,
        model_class=config.pretrained_model_class,
        dataset_name=task_config.dataset_name,
        tokenizer_name=config.tokenizer_name,
        layer_descriptions={
            path: topology.target_to_canon(path) for path in model.target_module_paths
        },
    )

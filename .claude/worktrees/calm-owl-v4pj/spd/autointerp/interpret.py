import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openrouter import OpenRouter
from openrouter.components import Effort, Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import AnthropicBatchConfig, OpenRouterConfig, StrategyConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.llm_api import (
    LLMError,
    LLMJob,
    LLMResult,
    make_response_format,
    map_llm_calls,
)
from spd.autointerp.schemas import InterpretationResult, ModelMetadata
from spd.autointerp.strategies.dispatch import INTERPRETATION_SCHEMA, format_prompt
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.log import logger


def _parse_interpretation(job: LLMJob, parsed: dict[str, Any], raw: str) -> InterpretationResult:
    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning_text = parsed["reasoning"]
    assert (
        isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning_text, str)
    )
    return InterpretationResult(
        component_key=job.key,
        label=label,
        confidence=confidence,
        reasoning=reasoning_text,
        raw_response=raw,
        prompt=job.prompt,
    )


async def interpret_component(
    api: OpenRouter,
    model: str,
    reasoning_effort: Effort,
    strategy: StrategyConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> InterpretationResult:
    """Interpret a single component. Used by the app for on-demand interpretation."""
    prompt = format_prompt(
        strategy=strategy,
        component=component,
        model_metadata=model_metadata,
        app_tok=app_tok,
        input_token_stats=input_token_stats,
        output_token_stats=output_token_stats,
    )

    schema = INTERPRETATION_SCHEMA
    response_format = make_response_format("interpretation", schema)

    response = await api.chat.send_async(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        reasoning=Reasoning(effort=reasoning_effort),
    )

    choice = response.choices[0]
    assert isinstance(choice.message.content, str)
    raw = choice.message.content
    parsed = json.loads(raw)

    job = LLMJob(prompt=prompt, schema=schema, key=component.component_key)
    return _parse_interpretation(job, parsed, raw)


def _build_jobs(
    remaining: list[ComponentData],
    token_stats: Any,
    template_strategy: StrategyConfig,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
) -> list[LLMJob]:
    jobs: list[LLMJob] = []
    for component in remaining:
        input_stats = get_input_token_stats(token_stats, component.component_key, app_tok, top_k=20)
        output_stats = get_output_token_stats(
            token_stats, component.component_key, app_tok, top_k=50
        )
        assert input_stats is not None
        assert output_stats is not None
        prompt = format_prompt(
            strategy=template_strategy,
            component=component,
            model_metadata=model_metadata,
            app_tok=app_tok,
            input_token_stats=input_stats,
            output_token_stats=output_stats,
        )
        jobs.append(
            LLMJob(prompt=prompt, schema=INTERPRETATION_SCHEMA, key=component.component_key)
        )
    return jobs


def _process_outcomes(
    outcomes: Iterable[LLMResult | LLMError],
    db: InterpDB,
    n_remaining: int,
) -> list[InterpretationResult]:
    results: list[InterpretationResult] = []
    n_errors = 0

    for outcome in outcomes:
        match outcome:
            case LLMResult(job=job, parsed=parsed, raw=raw):
                result = _parse_interpretation(job, parsed, raw)
                results.append(result)
                db.save_interpretation(result)
            case LLMError(job=job, error=e):
                n_errors += 1
                logger.error(f"Skipping {job.key}: {type(e).__name__}: {e}")

        error_rate = n_errors / (n_errors + len(results))
        if error_rate > 0.2 and n_errors > 10:
            raise RuntimeError(
                f"Error rate {error_rate:.0%} ({n_errors}/{n_remaining}) exceeds 20% threshold"
            )

    return results


def _run_with_anthropic_batch(
    api_key: str,
    backend: AnthropicBatchConfig,
    jobs: list[LLMJob],
    db: InterpDB,
) -> list[InterpretationResult]:
    from spd.autointerp.batch_api import run_batch_llm_calls

    outcomes = run_batch_llm_calls(
        anthropic_api_key=api_key,
        model=backend.model,
        jobs=jobs,
        max_tokens=8000,
        max_retries=backend.max_retries,
    )
    return _process_outcomes(outcomes, db, len(jobs))


async def _run_with_openrouter(
    api_key: str,
    backend: OpenRouterConfig,
    jobs: list[LLMJob],
    db: InterpDB,
    cost_limit_usd: float | None,
) -> list[InterpretationResult]:
    results: list[InterpretationResult] = []
    n_errors = 0

    async for outcome in map_llm_calls(
        openrouter_api_key=api_key,
        model=backend.model,
        reasoning_effort=backend.reasoning_effort,
        jobs=iter(jobs),
        max_tokens=8000,
        max_concurrent=backend.max_concurrent,
        max_requests_per_minute=backend.max_requests_per_minute,
        cost_limit_usd=cost_limit_usd,
        response_schema=INTERPRETATION_SCHEMA,
        n_total=len(jobs),
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed, raw=raw):
                result = _parse_interpretation(job, parsed, raw)
                results.append(result)
                db.save_interpretation(result)
            case LLMError(job=job, error=e):
                n_errors += 1
                logger.error(f"Skipping {job.key}: {type(e).__name__}: {e}")

        error_rate = n_errors / (n_errors + len(results))
        if error_rate > 0.2 and n_errors > 10:
            raise RuntimeError(
                f"Error rate {error_rate:.0%} ({n_errors}/{len(jobs)}) exceeds 20% threshold"
            )

    return results


def run_interpret(
    api_key: str,
    backend: AnthropicBatchConfig | OpenRouterConfig,
    limit: int | None,
    cost_limit_usd: float | None,
    model_metadata: ModelMetadata,
    template_strategy: StrategyConfig,
    harvest: HarvestRepo,
    db_path: Path,
    tokenizer_name: str,
) -> list[InterpretationResult]:
    db = InterpDB(db_path)

    try:
        completed = db.get_completed_keys()
        if completed:
            logger.info(f"Resuming: {len(completed)} already completed")

        logger.info("Loading component summaries...")
        summaries = harvest.get_summary()
        logger.info(f"Loaded {len(summaries)} component summaries")

        eligible_keys = sorted(summaries, key=lambda k: summaries[k].firing_density, reverse=True)
        if limit is not None:
            eligible_keys = eligible_keys[:limit]
        remaining_keys = [k for k in eligible_keys if k not in completed]

        logger.info(f"Loading {len(remaining_keys)} components from harvest...")
        remaining_data = harvest.get_components_bulk(remaining_keys)
        remaining = [remaining_data[k] for k in remaining_keys]
        logger.info(f"Interpreting {len(remaining)} components")

        logger.info("Loading token stats...")
        token_stats = harvest.get_token_stats()
        assert token_stats is not None, "token_stats.pt not found. Run harvest first."
        logger.info("Token stats loaded")

        app_tok = AppTokenizer.from_pretrained(tokenizer_name)

        logger.info("Building prompts...")
        jobs = _build_jobs(remaining, token_stats, template_strategy, model_metadata, app_tok)
        logger.info(f"Built {len(jobs)} jobs")

        match backend:
            case AnthropicBatchConfig():
                results = _run_with_anthropic_batch(api_key, backend, jobs, db)
            case OpenRouterConfig():
                results = asyncio.run(
                    _run_with_openrouter(api_key, backend, jobs, db, cost_limit_usd)
                )

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        db.close()
        raise e

    logger.info(f"Completed {len(results)} interpretations -> {db_path}")
    return results

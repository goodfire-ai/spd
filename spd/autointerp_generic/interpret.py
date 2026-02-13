"""Generic interpretation pipeline.

Takes DecompositionAutointerpData and produces labels for each component via LLM.
"""

import asyncio
import json
from pathlib import Path

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Reasoning

from spd.autointerp.llm_api import (
    BudgetExceededError,
    CostTracker,
    GlobalBackoff,
    LLMClient,
    get_model_pricing,
    make_response_format,
)
from spd.autointerp_generic.db import GenericInterpDB
from spd.autointerp_generic.prompt import INTERPRETATION_SCHEMA, format_interpret_prompt
from spd.autointerp_generic.types import (
    ComponentAutointerpData,
    DecompositionAutointerpData,
    InterpretationResult,
    InterpretConfig,
)
from spd.log import logger


async def interpret_component(
    llm: LLMClient,
    config: InterpretConfig,
    component: ComponentAutointerpData,
    data: DecompositionAutointerpData,
) -> InterpretationResult:
    prompt = format_interpret_prompt(
        config=config,
        component=component,
        decomposition_explanation=data.decomposition_explanation,
        app_tok=data.tokenizer,
    )

    raw = await llm.chat(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        context_label=component.key,
        response_format=make_response_format("interpretation", INTERPRETATION_SCHEMA),
        reasoning=Reasoning(effort=config.reasoning_effort),
    )

    parsed = json.loads(raw)
    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning_text = parsed["reasoning"]
    assert (
        isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning_text, str)
    )

    return InterpretationResult(
        component_key=component.key,
        label=label,
        confidence=confidence,
        reasoning=reasoning_text,
        raw_response=raw,
        prompt=prompt,
    )


def run_interpret(
    data: DecompositionAutointerpData,
    openrouter_api_key: str,
    db_path: Path,
    config: InterpretConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[InterpretationResult]:
    components = data.components
    if limit is not None:
        components = components[:limit]

    async def _run() -> list[InterpretationResult]:
        db = GenericInterpDB(db_path)
        results: list[InterpretationResult] = []
        n_errors = 0

        completed = db.get_completed_keys()
        if completed:
            logger.info(f"Resuming: {len(completed)} already completed")

        remaining = [c for c in components if c.key not in completed]
        logger.info(f"Interpreting {len(remaining)} components")

        semaphore = asyncio.Semaphore(config.max_concurrent)
        output_lock = asyncio.Lock()

        async def process_one(
            component: ComponentAutointerpData, index: int, llm: LLMClient
        ) -> None:
            async with semaphore:
                try:
                    result = await interpret_component(llm, config, component, data)
                except BudgetExceededError:
                    return
                except Exception as e:
                    nonlocal n_errors
                    n_errors += 1
                    logger.error(f"Skipping {component.key}: {type(e).__name__}: {e}")
                    return
                async with output_lock:
                    results.append(result)
                    db.save_interpretation(result)
                    if index % 100 == 0:
                        logger.info(
                            f"[{index}] ${llm.cost_tracker.cost_usd():.2f} "
                            f"({llm.cost_tracker.input_tokens:,} in, "
                            f"{llm.cost_tracker.output_tokens:,} out)"
                        )

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

            await asyncio.gather(*[process_one(c, i, llm) for i, c in enumerate(remaining)])

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

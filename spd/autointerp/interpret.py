import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from openrouter import OpenRouter

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import get_model_n_blocks
from spd.autointerp.config import AutointerpConfig
from spd.autointerp.llm_api import (
    CostTracker,
    RateLimiter,
    chat_with_retry,
    get_model_pricing,
    make_response_format,
)
from spd.autointerp.schemas import ArchitectureInfo, InterpretationResult
from spd.autointerp.strategies.dispatch import (
    format_prompt,
    get_reasoning,
    get_response_schema,
)
from spd.configs import LMTaskConfig
from spd.harvest.analysis import TokenPRLift, get_input_token_stats, get_output_token_stats
from spd.harvest.loaders import load_all_components
from spd.harvest.schemas import ComponentData
from spd.harvest.storage import TokenStatsStorage
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo

MAX_CONCURRENT_REQUESTS = 50
MAX_REQUESTS_PER_MINUTE = 300  # Gemini flash has 400 RPM limit


async def interpret_component(
    client: OpenRouter,
    config: AutointerpConfig,
    component: ComponentData,
    arch: ArchitectureInfo,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    ci_threshold: float,
) -> tuple[InterpretationResult, int, int] | None:
    """Returns (result, input_tokens, output_tokens), or None on failure."""
    prompt = format_prompt(
        config=config,
        component=component,
        arch=arch,
        app_tok=app_tok,
        input_token_stats=input_token_stats,
        output_token_stats=output_token_stats,
        ci_threshold=ci_threshold,
    )

    reasoning = get_reasoning(config)
    schema = get_response_schema(config)

    try:
        raw, in_tok, out_tok = await chat_with_retry(
            client=client,
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,  # High to accommodate reasoning tokens
            context_label=component.component_key,
            response_format=make_response_format("interpretation", schema),
            reasoning=reasoning,
        )
    except RuntimeError as e:
        logger.error(str(e))
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: `{raw}`")
        return None

    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning_text = parsed["reasoning"]
    assert (
        isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning_text, str)
    )

    return (
        InterpretationResult(
            component_key=component.component_key,
            label=label,
            confidence=confidence,
            reasoning=reasoning_text,
            raw_response=raw,
            prompt=prompt,
        ),
        in_tok,
        out_tok,
    )


async def interpret_all(
    components: list[ComponentData],
    arch: ArchitectureInfo,
    openrouter_api_key: str,
    config: AutointerpConfig,
    output_path: Path,
    token_stats: TokenStatsStorage,
    ci_threshold: float,
    limit: int | None,
    cost_limit_usd: float | None = None,
) -> list[InterpretationResult]:
    """Interpret all components with maximum parallelism. Rate limits handled via exponential backoff."""
    results: list[InterpretationResult] = []
    completed = set[str]()
    n_errors = 0

    if output_path.exists():
        print(f"Resuming: {output_path} exists")
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(InterpretationResult(**data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already completed")

    components_sorted = sorted(components, key=lambda c: c.mean_ci, reverse=True)
    remaining = [c for c in components_sorted if c.component_key not in completed]
    if limit is not None:
        remaining = remaining[:limit]
    print(f"Interpreting {len(remaining)} components")
    start_idx = len(results)

    output_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, period_seconds=60.0)

    app_tok = AppTokenizer.from_pretrained(arch.tokenizer_name)

    reasoning = get_reasoning(config)

    async def process_one(
        component: ComponentData,
        index: int,
        client: OpenRouter,
        cost_tracker: CostTracker,
    ) -> None:
        nonlocal n_errors
        if cost_tracker.over_budget():
            return
        await rate_limiter.acquire()
        async with semaphore:
            if cost_tracker.over_budget():
                return
            try:
                # Compute token stats for this component
                input_stats = get_input_token_stats(
                    token_stats, component.component_key, app_tok, top_k=20
                )
                output_stats = get_output_token_stats(
                    token_stats, component.component_key, app_tok, top_k=50
                )
                assert input_stats is not None, (
                    f"No input token stats for {component.component_key}"
                )
                assert output_stats is not None, (
                    f"No output token stats for {component.component_key}"
                )

                res = await interpret_component(
                    client=client,
                    config=config,
                    component=component,
                    arch=arch,
                    app_tok=app_tok,
                    input_token_stats=input_stats,
                    output_token_stats=output_stats,
                    ci_threshold=ci_threshold,
                )
                if res is None:
                    n_errors += 1
                    return
                result, in_tok, out_tok = res

                async with output_lock:
                    results.append(result)
                    await cost_tracker.add(in_tok, out_tok)
                    line = json.dumps(asdict(result)) + "\n"
                    log_progress = index % 100 == 0
                    progress_msg = (
                        f"[{index}] ${cost_tracker.cost_usd():.2f} ({cost_tracker.input_tokens:,} in, {cost_tracker.output_tokens:,} out)"
                        if log_progress
                        else ""
                    )
                with open(output_path, "a") as f:
                    f.write(line)

                if log_progress:
                    logger.info(progress_msg)
            except Exception as e:
                logger.error(f"Skipping {component.component_key}: {type(e).__name__}: {e}")
                n_errors += 1

    async with OpenRouter(api_key=openrouter_api_key) as client:
        input_price, output_price = await get_model_pricing(client, config.model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=cost_limit_usd,
        )
        limit_str = f" (limit: ${cost_limit_usd:.2f})" if cost_limit_usd is not None else ""
        reasoning_str = f"reasoning={reasoning.effort}" if reasoning else "no reasoning"
        print(f"Model: {config.model}, {reasoning_str}")
        print(
            f"Pricing: ${input_price * 1e6:.2f}/M input, ${output_price * 1e6:.2f}/M output{limit_str}"
        )

        await asyncio.gather(
            *[
                process_one(c, i, client, cost_tracker)
                for i, c in enumerate(remaining, start=start_idx)
            ]
        )

    if cost_tracker.over_budget():
        print(
            f"Cost limit reached: ${cost_tracker.cost_usd():.2f} >= ${cost_limit_usd}. "
            f"Completed {len(results)} / {len(remaining) + len(completed)} components."
        )
    if n_errors > 0:
        logger.warning(f"{n_errors} components failed during interpretation")
    print(f"Final cost: ${cost_tracker.cost_usd():.2f}")
    return results


def get_architecture_info(wandb_path: str) -> ArchitectureInfo:
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    n_blocks = get_model_n_blocks(model.target_model)
    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    assert config.tokenizer_name is not None
    return ArchitectureInfo(
        n_blocks=n_blocks,
        c_per_layer=model.module_to_c,
        model_class=config.pretrained_model_class,
        dataset_name=task_config.dataset_name,
        tokenizer_name=config.tokenizer_name,
    )


def run_interpret(
    wandb_path: str,
    openrouter_api_key: str,
    config: AutointerpConfig,
    run_id: str,
    correlations_dir: Path,
    output_path: Path,
    ci_threshold: float,
    limit: int | None,
    cost_limit_usd: float | None = None,
) -> list[InterpretationResult]:
    arch = get_architecture_info(wandb_path)
    components = load_all_components(run_id)

    # Load token stats
    token_stats_path = correlations_dir / "token_stats.pt"
    assert token_stats_path.exists(), (
        f"token_stats.pt not found at {token_stats_path}. Run harvest first."
    )
    token_stats = TokenStatsStorage.load(token_stats_path)

    results = asyncio.run(
        interpret_all(
            components=components,
            arch=arch,
            openrouter_api_key=openrouter_api_key,
            config=config,
            output_path=output_path,
            token_stats=token_stats,
            ci_threshold=ci_threshold,
            limit=limit,
            cost_limit_usd=cost_limit_usd,
        )
    )

    print(f"Completed {len(results)} interpretations -> {output_path}")
    return results

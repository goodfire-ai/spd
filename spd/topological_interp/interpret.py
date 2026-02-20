"""Main three-phase topological interpretation execution.

Phase 1: Output pass (late → early) — "What does this component DO?"
Phase 2: Input pass (early → late) — "What TRIGGERS this component?"
Phase 3: Unification (parallel over all) — Synthesize into unified label.

Each directional pass is a scan: components are processed layer-by-layer, and
each component's prompt includes labels from previously-labeled components in
the same scan direction. The scan accumulator is an in-memory dict; the DB is
a write-only durable sink (read only on resume to seed the accumulator).
"""

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.llm_api import LLMError, LLMJob, LLMResult, map_llm_calls
from spd.autointerp.schemas import ModelMetadata
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.harvest.analysis import get_input_token_stats, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger
from spd.topological_interp.config import TopologicalInterpConfig
from spd.topological_interp.db import TopologicalInterpDB
from spd.topological_interp.graph_context import (
    RelatedComponent,
    get_cofiring_components,
    get_downstream_components,
    get_upstream_components,
)
from spd.topological_interp.ordering import group_and_sort_by_layer
from spd.topological_interp.prompts import (
    LABEL_SCHEMA,
    format_input_prompt,
    format_output_prompt,
    format_unification_prompt,
)
from spd.topological_interp.schemas import LabelResult, PromptEdge


def run_topological_interp(
    openrouter_api_key: str,
    config: TopologicalInterpConfig,
    harvest: HarvestRepo,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    token_stats: TokenStatsStorage,
    model_metadata: ModelMetadata,
    db_path: Path,
    tokenizer_name: str,
) -> None:
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)
    components = harvest.get_all_components()

    components = [c for c in components if c.firing_density > 0.0]
    components = sorted(components, key=lambda c: c.firing_density, reverse=True)

    if config.limit is not None:
        components = components[: config.limit]

    all_keys = [c.component_key for c in components]
    component_by_key = {c.component_key: c for c in components}
    layers_ordered = group_and_sort_by_layer(all_keys, model_metadata.layer_descriptions)

    total = len(all_keys)
    logger.info(f"Topological interp: {total} components across {len(layers_ordered)} layers")

    async def _run() -> None:
        db = TopologicalInterpDB(db_path)

        try:
            logger.section("Phase 1: Output pass (late → early)")
            await _run_output_pass(
                db=db,
                layers_ordered=layers_ordered,
                component_by_key=component_by_key,
                config=config,
                openrouter_api_key=openrouter_api_key,
                model_metadata=model_metadata,
                app_tok=app_tok,
                token_stats=token_stats,
                attribution_storage=attribution_storage,
                correlation_storage=correlation_storage,
                total=total,
            )

            logger.section("Phase 2: Input pass (early → late)")
            await _run_input_pass(
                db=db,
                layers_ordered=layers_ordered,
                component_by_key=component_by_key,
                config=config,
                openrouter_api_key=openrouter_api_key,
                model_metadata=model_metadata,
                app_tok=app_tok,
                token_stats=token_stats,
                attribution_storage=attribution_storage,
                correlation_storage=correlation_storage,
                total=total,
            )

            logger.section("Phase 3: Unification")
            await _run_unification(
                db=db,
                all_keys=all_keys,
                config=config,
                openrouter_api_key=openrouter_api_key,
            )

            output_count = db.get_label_count("output_labels")
            input_count = db.get_label_count("input_labels")
            unified_count = db.get_label_count("unified_labels")
            logger.info(
                f"Completed: {output_count} output, {input_count} input, "
                f"{unified_count} unified labels -> {db_path}"
            )
        finally:
            db.close()

    asyncio.run(_run())


# -- Phase 1: Output pass -----------------------------------------------------


def _build_output_jobs(
    keys: list[str],
    component_by_key: dict[str, ComponentData],
    token_stats: TokenStatsStorage,
    app_tok: AppTokenizer,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    labels_so_far: dict[str, LabelResult],
    db: TopologicalInterpDB,
    config: TopologicalInterpConfig,
    model_metadata: ModelMetadata,
) -> Iterable[LLMJob]:
    for key in keys:
        component = component_by_key[key]

        input_stats = get_input_token_stats(token_stats, key, app_tok, top_k=20)
        output_stats = get_output_token_stats(token_stats, key, app_tok, top_k=50)
        assert input_stats is not None, f"No input token stats for {key}"
        assert output_stats is not None, f"No output token stats for {key}"

        downstream = get_downstream_components(
            key,
            attribution_storage,
            correlation_storage,
            labels_so_far,
            model_metadata.layer_descriptions,
            config.top_k_attributed,
        )

        _save_edges(db, key, downstream, "downstream", "output")

        prompt = format_output_prompt(
            component=component,
            model_metadata=model_metadata,
            app_tok=app_tok,
            input_token_stats=input_stats,
            output_token_stats=output_stats,
            downstream=downstream,
            label_max_words=config.label_max_words,
            max_examples=config.max_examples,
        )
        yield LLMJob(prompt=prompt, schema=LABEL_SCHEMA, key=key)


async def _run_output_pass(
    db: TopologicalInterpDB,
    layers_ordered: list[tuple[str, list[str]]],
    component_by_key: dict[str, ComponentData],
    config: TopologicalInterpConfig,
    openrouter_api_key: str,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    token_stats: TokenStatsStorage,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    total: int,
) -> None:
    # Seed scan accumulator from DB for resume
    labels_so_far: dict[str, LabelResult] = db.get_all_output_labels()
    if labels_so_far:
        logger.info(f"Output pass: resuming, {len(labels_so_far)} already completed")

    completed_so_far = 0

    for layer, keys in reversed(layers_ordered):
        pending = [k for k in keys if k not in labels_so_far]
        if not pending:
            completed_so_far += len(keys)
            continue

        jobs = _build_output_jobs(
            pending,
            component_by_key,
            token_stats,
            app_tok,
            attribution_storage,
            correlation_storage,
            labels_so_far,
            db,
            config,
            model_metadata,
        )

        n_errors = 0
        n_done = 0

        async for outcome in map_llm_calls(
            openrouter_api_key=openrouter_api_key,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            jobs=jobs,
            max_tokens=8000,
            max_concurrent=config.max_concurrent,
            max_requests_per_minute=config.max_requests_per_minute,
            cost_limit_usd=config.cost_limit_usd,
            response_schema=LABEL_SCHEMA,
            n_total=len(pending),
        ):
            match outcome:
                case LLMResult(job=job, parsed=parsed, raw=raw):
                    result = _parsed_to_label_result(job.key, parsed, raw, job.prompt)
                    labels_so_far[job.key] = result
                    db.save_output_label(result)
                    n_done += 1
                case LLMError(job=job, error=e):
                    n_errors += 1
                    logger.error(f"Output pass: skipping {job.key}: {type(e).__name__}: {e}")

            _check_error_rate(n_errors, n_done)

        completed_so_far += len(keys)
        logger.info(f"Output pass: completed layer {layer} ({completed_so_far}/{total})")


# -- Phase 2: Input pass ------------------------------------------------------


def _build_input_jobs(
    keys: list[str],
    component_by_key: dict[str, ComponentData],
    token_stats: TokenStatsStorage,
    app_tok: AppTokenizer,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    labels_so_far: dict[str, LabelResult],
    db: TopologicalInterpDB,
    config: TopologicalInterpConfig,
    model_metadata: ModelMetadata,
) -> Iterable[LLMJob]:
    for key in keys:
        component = component_by_key[key]

        input_stats = get_input_token_stats(token_stats, key, app_tok, top_k=20)
        output_stats = get_output_token_stats(token_stats, key, app_tok, top_k=50)
        assert input_stats is not None, f"No input token stats for {key}"
        assert output_stats is not None, f"No output token stats for {key}"

        upstream = get_upstream_components(
            key,
            attribution_storage,
            correlation_storage,
            labels_so_far,
            model_metadata.layer_descriptions,
            config.top_k_attributed,
        )
        cofiring = get_cofiring_components(key, correlation_storage, config.top_k_correlated)

        _save_edges(db, key, upstream, "upstream", "input")
        _save_edges(db, key, cofiring, "upstream", "input")

        prompt = format_input_prompt(
            component=component,
            model_metadata=model_metadata,
            app_tok=app_tok,
            input_token_stats=input_stats,
            output_token_stats=output_stats,
            upstream=upstream,
            cofiring=cofiring,
            label_max_words=config.label_max_words,
            max_examples=config.max_examples,
        )
        yield LLMJob(prompt=prompt, schema=LABEL_SCHEMA, key=key)


async def _run_input_pass(
    db: TopologicalInterpDB,
    layers_ordered: list[tuple[str, list[str]]],
    component_by_key: dict[str, ComponentData],
    config: TopologicalInterpConfig,
    openrouter_api_key: str,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    token_stats: TokenStatsStorage,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    total: int,
) -> None:
    # Seed scan accumulator from DB for resume
    labels_so_far: dict[str, LabelResult] = db.get_all_input_labels()
    if labels_so_far:
        logger.info(f"Input pass: resuming, {len(labels_so_far)} already completed")

    completed_so_far = 0

    for layer, keys in layers_ordered:
        pending = [k for k in keys if k not in labels_so_far]
        if not pending:
            completed_so_far += len(keys)
            continue

        jobs = _build_input_jobs(
            pending,
            component_by_key,
            token_stats,
            app_tok,
            attribution_storage,
            correlation_storage,
            labels_so_far,
            db,
            config,
            model_metadata,
        )

        n_errors = 0
        n_done = 0

        async for outcome in map_llm_calls(
            openrouter_api_key=openrouter_api_key,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            jobs=jobs,
            max_tokens=8000,
            max_concurrent=config.max_concurrent,
            max_requests_per_minute=config.max_requests_per_minute,
            cost_limit_usd=config.cost_limit_usd,
            response_schema=LABEL_SCHEMA,
            n_total=len(pending),
        ):
            match outcome:
                case LLMResult(job=job, parsed=parsed, raw=raw):
                    result = _parsed_to_label_result(job.key, parsed, raw, job.prompt)
                    labels_so_far[job.key] = result
                    db.save_input_label(result)
                    n_done += 1
                case LLMError(job=job, error=e):
                    n_errors += 1
                    logger.error(f"Input pass: skipping {job.key}: {type(e).__name__}: {e}")

            _check_error_rate(n_errors, n_done)

        completed_so_far += len(keys)
        logger.info(f"Input pass: completed layer {layer} ({completed_so_far}/{total})")


# -- Phase 3: Unification -----------------------------------------------------


def _build_unification_jobs(
    keys: list[str],
    db: TopologicalInterpDB,
    config: TopologicalInterpConfig,
) -> Iterable[LLMJob]:
    n_skipped = 0
    for key in keys:
        output_label = db.get_output_label(key)
        input_label = db.get_input_label(key)
        if output_label is None or input_label is None:
            n_skipped += 1
            logger.warning(
                f"Skipping unification for {key}: "
                f"output={'yes' if output_label else 'MISSING'}, "
                f"input={'yes' if input_label else 'MISSING'}"
            )
            continue

        prompt = format_unification_prompt(
            output_label=output_label,
            input_label=input_label,
            label_max_words=config.label_max_words,
        )
        yield LLMJob(prompt=prompt, schema=LABEL_SCHEMA, key=key)
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} components missing output or input labels")


async def _run_unification(
    db: TopologicalInterpDB,
    all_keys: list[str],
    config: TopologicalInterpConfig,
    openrouter_api_key: str,
) -> None:
    completed = db.get_completed_unified_keys()
    pending = [k for k in all_keys if k not in completed]

    if not pending:
        logger.info("Unification: all labels already completed")
        return

    if completed:
        logger.info(f"Unification: resuming, {len(completed)} already completed")

    logger.info(f"Unifying {len(pending)} components")

    jobs = _build_unification_jobs(pending, db, config)

    n_errors = 0
    n_done = 0

    async for outcome in map_llm_calls(
        openrouter_api_key=openrouter_api_key,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        jobs=jobs,
        max_tokens=4000,
        max_concurrent=config.max_concurrent,
        max_requests_per_minute=config.max_requests_per_minute,
        cost_limit_usd=config.cost_limit_usd,
        response_schema=LABEL_SCHEMA,
        n_total=len(pending),
    ):
        match outcome:
            case LLMResult(job=job, parsed=parsed, raw=raw):
                result = _parsed_to_label_result(job.key, parsed, raw, job.prompt)
                db.save_unified_label(result)
                n_done += 1
            case LLMError(job=job, error=e):
                n_errors += 1
                logger.error(f"Unification: skipping {job.key}: {type(e).__name__}: {e}")

        _check_error_rate(n_errors, n_done)

    logger.info(f"Unification: completed {n_done}/{len(pending)}")


# -- Helpers -------------------------------------------------------------------


def _parsed_to_label_result(
    component_key: str,
    parsed: dict[str, object],
    raw: str,
    prompt: str,
) -> LabelResult:
    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}"
    label = parsed["label"]
    confidence = parsed["confidence"]
    reasoning = parsed["reasoning"]
    assert isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning, str)
    return LabelResult(
        component_key=component_key,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw,
        prompt=prompt,
    )


def _check_error_rate(n_errors: int, n_done: int) -> None:
    total = n_errors + n_done
    if total > 10 and n_errors / total > 0.2:
        raise RuntimeError(
            f"Error rate {n_errors / total:.0%} ({n_errors}/{total}) exceeds 20% threshold"
        )


def _save_edges(
    db: TopologicalInterpDB,
    component_key: str,
    related: list[RelatedComponent],
    direction: Literal["upstream", "downstream"],
    pass_name: Literal["output", "input"],
) -> None:
    if not related:
        return
    edges = [
        PromptEdge(
            component_key=component_key,
            related_key=r.component_key,
            direction=direction,
            pass_name=pass_name,
            attribution=r.attribution,
            related_label=r.label,
            related_confidence=r.confidence,
        )
        for r in related
    ]
    db.save_prompt_edges(edges)

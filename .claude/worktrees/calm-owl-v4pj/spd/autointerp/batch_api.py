"""Anthropic Message Batches API for bulk LLM calls.

Submits all jobs as a single batch (or multiple sub-batches), polls for completion,
streams results, and retries expired items.
"""

import json
import time
from typing import Any

import anthropic
from anthropic.types import ToolParam, ToolUseBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import (
    MessageBatchCanceledResult,
    MessageBatchErroredResult,
    MessageBatchExpiredResult,
    MessageBatchSucceededResult,
)
from anthropic.types.messages.batch_create_params import Request

from spd.autointerp.llm_api import LLMError, LLMJob, LLMResult
from spd.log import logger

_SUB_BATCH_SIZE = 10_000
_POLL_INTERVAL_S = 30


def _make_tool(schema: dict[str, Any]) -> ToolParam:
    return ToolParam(
        name="respond",
        description="Respond with structured output",
        input_schema=schema,
    )


def _job_to_request(job: LLMJob, model: str, max_tokens: int) -> Request:
    return Request(
        custom_id=job.key,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": job.prompt}],
            tools=[_make_tool(job.schema)],
            tool_choice={"type": "tool", "name": "respond"},
        ),
    )


def _extract_tool_input(message: anthropic.types.Message) -> dict[str, Any]:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            assert isinstance(block.input, dict)
            return block.input
    raise ValueError(f"No tool_use block in response: {message.content}")


def _submit_and_poll(
    client: anthropic.Anthropic,
    requests: list[Request],
    batch_label: str,
) -> str:
    batch = client.messages.batches.create(requests=requests)
    logger.info(f"[{batch_label}] Submitted batch {batch.id} ({len(requests)} requests)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        logger.info(
            f"[{batch_label}] {batch.id}: "
            f"{counts.succeeded} succeeded, {counts.errored} errored, "
            f"{counts.expired} expired, {counts.processing} processing"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(_POLL_INTERVAL_S)

    return batch.id


def _collect_results(
    client: anthropic.Anthropic,
    batch_id: str,
    jobs_by_key: dict[str, LLMJob],
) -> tuple[list[LLMResult], list[LLMError], list[LLMJob]]:
    results: list[LLMResult] = []
    errors: list[LLMError] = []
    expired_jobs: list[LLMJob] = []

    for item in client.messages.batches.results(batch_id):
        job = jobs_by_key[item.custom_id]
        result = item.result
        match result:
            case MessageBatchSucceededResult():
                try:
                    parsed = _extract_tool_input(result.message)
                    raw = json.dumps(parsed)
                    results.append(LLMResult(job=job, parsed=parsed, raw=raw))
                except Exception as e:
                    errors.append(LLMError(job=job, error=e))
            case MessageBatchErroredResult():
                error_msg = str(result.error) if result.error else "Unknown error"
                errors.append(LLMError(job=job, error=RuntimeError(error_msg)))
            case MessageBatchExpiredResult():
                expired_jobs.append(job)
            case MessageBatchCanceledResult():
                errors.append(LLMError(job=job, error=RuntimeError("Request canceled")))

    return results, errors, expired_jobs


def run_batch_llm_calls(
    anthropic_api_key: str,
    model: str,
    jobs: list[LLMJob],
    max_tokens: int,
    max_retries: int,
) -> list[LLMResult | LLMError]:
    """Submit jobs via Anthropic Batches API, poll for completion, return results.

    Splits into sub-batches of 10K, auto-retries expired items up to max_retries times.
    """
    assert jobs, "No jobs to process"

    client = anthropic.Anthropic(api_key=anthropic_api_key)
    jobs_by_key = {job.key: job for job in jobs}

    all_results: list[LLMResult] = []
    all_errors: list[LLMError] = []
    pending_jobs = list(jobs)

    for attempt in range(1 + max_retries):
        if not pending_jobs:
            break

        label = "batch" if attempt == 0 else f"retry-{attempt}"
        requests = [_job_to_request(j, model, max_tokens) for j in pending_jobs]

        for i in range(0, len(requests), _SUB_BATCH_SIZE):
            chunk = requests[i : i + _SUB_BATCH_SIZE]
            chunk_label = (
                f"{label}"
                if len(requests) <= _SUB_BATCH_SIZE
                else f"{label}-{i // _SUB_BATCH_SIZE}"
            )
            batch_id = _submit_and_poll(client, chunk, chunk_label)
            results, errors, expired = _collect_results(client, batch_id, jobs_by_key)
            all_results.extend(results)
            all_errors.extend(errors)
            pending_jobs = expired

        if pending_jobs:
            logger.warning(f"[{label}] {len(pending_jobs)} requests expired, will retry")

    if pending_jobs:
        for job in pending_jobs:
            all_errors.append(LLMError(job=job, error=RuntimeError("Expired after all retries")))

    logger.info(f"Batch complete: {len(all_results)} succeeded, {len(all_errors)} failed")

    out: list[LLMResult | LLMError] = []
    out.extend(all_results)
    out.extend(all_errors)
    return out

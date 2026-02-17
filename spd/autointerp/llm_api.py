"""LLM API client with rate limiting, cost tracking, and retry."""

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import (
    JSONSchemaConfig,
    MessageTypedDict,
    Reasoning,
    ResponseFormatJSONSchema,
)
from openrouter.errors import (
    BadGatewayResponseError,
    ChatError,
    EdgeNetworkTimeoutResponseError,
    InternalServerResponseError,
    OpenRouterDefaultError,
    OpenRouterError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)

from spd.log import logger

MAX_RETRIES = 8
BASE_DELAY_S = 0.5
MAX_DELAY_S = 60.0
JITTER_FACTOR = 0.5
REQUEST_TIMEOUT_MS = 120_000
JSON_PARSE_RETRIES = 3

RETRYABLE_ERRORS = (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    BadGatewayResponseError,
    InternalServerResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
    ChatError,
    OpenRouterDefaultError,
    httpx.TransportError,
)


class BudgetExceededError(Exception):
    pass


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0
    limit_usd: float | None = None
    _budget_exceeded: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if self.limit_usd is not None and self.cost_usd() >= self.limit_usd:
                self._budget_exceeded.set()

    def over_budget(self) -> bool:
        return self._budget_exceeded.is_set()

    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )


MAX_BACKOFF_S = 600.0


class GlobalBackoff:
    """Shared backoff that pauses all coroutines when the API pushes back."""

    def __init__(self) -> None:
        self._resume_at = 0.0
        self._lock = asyncio.Lock()

    async def set_backoff(self, seconds: float) -> None:
        assert seconds <= MAX_BACKOFF_S, (
            f"Server requested {seconds:.0f}s backoff, exceeds {MAX_BACKOFF_S:.0f}s cap"
        )
        async with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)

    async def wait(self) -> None:
        delay = self._resume_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


def make_response_format(name: str, schema: dict[str, Any]) -> ResponseFormatJSONSchema:
    return ResponseFormatJSONSchema(
        json_schema=JSONSchemaConfig(
            name=name,
            schema_={**schema, "additionalProperties": False},
            strict=True,
        )
    )


async def get_model_pricing(api: OpenRouter, model_id: str) -> tuple[float, float]:
    """Returns (input_price, output_price) per token."""
    response = await api.models.list_async()
    for model in response.data:
        if model.id == model_id:
            return float(model.pricing.prompt), float(model.pricing.completion)
    raise ValueError(f"Model {model_id} not found")


def _get_retry_after(e: Exception) -> float | None:
    """Extract Retry-After seconds from an OpenRouter error, if present."""
    if not isinstance(e, OpenRouterError):
        return None
    val = e.headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


@dataclass
class LLMClient:
    """OpenRouter client with rate limiting, cost tracking, and retry.

    All API calls go through `chat()`, which enforces budget limits,
    rate limiting (token bucket via aiolimiter), and retries with
    exponential backoff. A shared GlobalBackoff pauses all coroutines
    when the API returns Retry-After headers.
    """

    api: OpenRouter
    rate_limiter: AsyncLimiter
    backoff: GlobalBackoff
    cost_tracker: CostTracker

    async def chat(
        self,
        model: str,
        messages: list[MessageTypedDict],
        max_tokens: int,
        context_label: str,
        response_format: ResponseFormatJSONSchema | None = None,
        reasoning: Reasoning | None = None,
    ) -> str:
        """Send a chat request. Returns response content.

        Raises BudgetExceededError if cost limit is reached.
        Raises RuntimeError if all retries are exhausted.
        """
        if self.cost_tracker.over_budget():
            raise BudgetExceededError(f"${self.cost_tracker.cost_usd():.2f}")

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            await self.backoff.wait()
            async with self.rate_limiter:
                try:
                    kwargs: dict[str, Any] = dict(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        timeout_ms=REQUEST_TIMEOUT_MS,
                    )
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    if reasoning is not None:
                        kwargs["reasoning"] = reasoning

                    response = await self.api.chat.send_async(**kwargs)
                    choice = response.choices[0]
                    message = choice.message
                    assert isinstance(message.content, str)
                    assert response.usage is not None

                    if choice.finish_reason == "length":
                        logger.warning(
                            f"{context_label}: Response truncated at {max_tokens} tokens"
                        )

                    await self.cost_tracker.add(
                        int(response.usage.prompt_tokens),
                        int(response.usage.completion_tokens),
                    )

                    return message.content
                except RETRYABLE_ERRORS as e:
                    last_error = e
                    if attempt == MAX_RETRIES - 1:
                        break

                    retry_after = _get_retry_after(e)
                    if retry_after is not None:
                        await self.backoff.set_backoff(retry_after)
                        delay = retry_after
                    else:
                        delay = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
                        jitter = delay * JITTER_FACTOR * random.random()
                        delay = delay + jitter

                    logger.warning(
                        f"[retry {attempt + 1}/{MAX_RETRIES}] ({context_label}) "
                        f"{type(e).__name__}, backing off {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        assert last_error is not None
        raise RuntimeError(f"Max retries exceeded for {context_label}: {last_error}")


@dataclass
class LLMJob:
    prompt: str
    schema: dict[str, Any]
    key: str


@dataclass
class LLMResult:
    job: LLMJob
    parsed: dict[str, Any]
    raw: str


@dataclass
class LLMError:
    job: LLMJob
    error: Exception


async def map_api_calls(
    jobs: Sequence[LLMJob],
    llm: LLMClient,
    model: str,
    max_tokens: int,
    reasoning: Reasoning | None,
    max_concurrent: int,
) -> AsyncGenerator[LLMResult | LLMError]:
    """Fan out LLM calls concurrently, yielding parsed results as they complete.

    Handles rate limiting (via LLMClient), JSON parsing with retry, and progress logging.
    Yields LLMResult on success, LLMError on failure. BudgetExceededError silently stops
    remaining jobs.
    """
    queue: asyncio.Queue[LLMResult | LLMError | None] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrent)
    n_total = len(jobs)
    n_done = 0
    budget_exceeded = False
    response_format = make_response_format("response", jobs[0].schema) if jobs else None

    async def process_one(job: LLMJob) -> None:
        nonlocal n_done, budget_exceeded
        if budget_exceeded:
            return
        async with semaphore:
            try:
                assert response_format is not None
                raw = ""
                parsed = None
                for attempt in range(JSON_PARSE_RETRIES):
                    raw = await llm.chat(
                        model=model,
                        messages=[{"role": "user", "content": job.prompt}],
                        max_tokens=max_tokens,
                        context_label=job.key,
                        response_format=response_format,
                        reasoning=reasoning,
                    )
                    try:
                        parsed = json.loads(raw)
                        break
                    except json.JSONDecodeError:
                        if attempt == JSON_PARSE_RETRIES - 1:
                            raise
                        logger.warning(
                            f"{job.key}: invalid JSON "
                            f"(attempt {attempt + 1}/{JSON_PARSE_RETRIES}), retrying"
                        )
                assert parsed is not None
                await queue.put(LLMResult(job=job, parsed=parsed, raw=raw))
            except BudgetExceededError:
                budget_exceeded = True
                return
            except Exception as e:
                await queue.put(LLMError(job=job, error=e))
        n_done += 1
        if n_done % 100 == 0 or n_done == n_total:
            logger.info(
                f"[{n_done}/{n_total}] ${llm.cost_tracker.cost_usd():.2f} "
                f"({llm.cost_tracker.input_tokens:,} in, "
                f"{llm.cost_tracker.output_tokens:,} out)"
            )

    async def run_all() -> None:
        await asyncio.gather(*[process_one(job) for job in jobs])
        await queue.put(None)

    task = asyncio.create_task(run_all())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

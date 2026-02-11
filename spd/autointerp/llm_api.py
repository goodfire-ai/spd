"""Shared LLM API utilities: retry logic, rate limiting, cost tracking, pricing, scoring pipeline."""

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
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
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)

from spd.harvest.schemas import ComponentData
from spd.log import logger

MAX_RETRIES = 8
BASE_DELAY_S = 0.5
MAX_DELAY_S = 60.0
JITTER_FACTOR = 0.5

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


class RateLimiter:
    """Sliding window rate limiter for async code."""

    def __init__(self, max_requests: int, period_seconds: float = 60.0):
        self.max_requests = max_requests
        self.period = period_seconds
        self.timestamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                self.timestamps = [t for t in self.timestamps if now - t < self.period]

                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return

                sleep_time = self.timestamps[0] + self.period - now

            # Sleep OUTSIDE the lock so other coroutines aren't blocked
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


def make_response_format(name: str, schema: dict[str, Any]) -> ResponseFormatJSONSchema:
    return ResponseFormatJSONSchema(
        json_schema=JSONSchemaConfig(
            name=name,
            schema_={**schema, "additionalProperties": False},
            strict=True,
        )
    )


async def chat_with_retry(
    client: OpenRouter,
    model: str,
    messages: list[MessageTypedDict],
    max_tokens: int,
    context_label: str,
    response_format: ResponseFormatJSONSchema | None = None,
    reasoning: Reasoning | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, int, int]:
    """Send chat request with exponential backoff retry. Returns (content, input_tokens, output_tokens)."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if rate_limiter is not None:
            await rate_limiter.acquire()
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            if reasoning is not None:
                kwargs["reasoning"] = reasoning

            response = await client.chat.send_async(**kwargs)
            choice = response.choices[0]
            message = choice.message
            assert isinstance(message.content, str)
            assert response.usage is not None

            if choice.finish_reason == "length":
                logger.warning(f"{context_label}: Response truncated at {max_tokens} tokens")

            return (
                message.content,
                int(response.usage.prompt_tokens),
                int(response.usage.completion_tokens),
            )
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break

            delay = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
            jitter = delay * JITTER_FACTOR * random.random()
            total_delay = delay + jitter

            logger.warning(
                f"[retry {attempt + 1}/{MAX_RETRIES}] ({context_label}) "
                f"{type(e).__name__}, backing off {total_delay:.1f}s"
            )
            await asyncio.sleep(total_delay)

    assert last_error is not None
    raise RuntimeError(f"Max retries exceeded for {context_label}: {last_error}")


async def get_model_pricing(client: OpenRouter, model_id: str) -> tuple[float, float]:
    """Returns (input_price, output_price) per token."""
    response = await client.models.list_async()
    for model in response.data:
        if model.id == model_id:
            return float(model.pricing.prompt), float(model.pricing.completion)
    raise ValueError(f"Model {model_id} not found")


# ---------------------------------------------------------------------------
# Shared scoring pipeline
# ---------------------------------------------------------------------------

T = TypeVar("T")

MAX_CONCURRENT_REQUESTS = 50
MAX_REQUESTS_PER_MINUTE = 200


async def run_scoring_pipeline(
    *,
    eligible: list[ComponentData],
    score_fn: Callable[[OpenRouter, ComponentData, RateLimiter], Awaitable[T]],
    serialize_fn: Callable[[T], dict[str, Any]],
    deserialize_fn: Callable[[dict[str, Any]], T],
    model: str,
    openrouter_api_key: str,
    output_path: Path,
    cost_limit_usd: float | None = None,
) -> list[T]:
    """Shared async scoring pipeline with resume, rate limiting, and cost tracking.

    Args:
        eligible: Components to score (already filtered for eligibility).
        score_fn: Async function (client, component, rate_limiter) -> result dataclass.
        serialize_fn: Convert a result to a JSON-serializable dict.
        deserialize_fn: Construct a result from a JSON dict (for resume).
        model: OpenRouter model ID (for pricing lookup).
        openrouter_api_key: API key.
        output_path: JSONL file for append-only results.
        cost_limit_usd: Optional budget cap.
    """
    results: list[T] = []
    completed = set[str]()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(deserialize_fn(data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    print(f"Scoring {len(remaining)} components")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
    output_lock = asyncio.Lock()

    async def process_one(
        component: ComponentData,
        index: int,
        client: OpenRouter,
        cost_tracker: CostTracker,
    ) -> None:
        if cost_tracker.over_budget():
            return
        async with semaphore:
            if cost_tracker.over_budget():
                return
            try:
                result = await score_fn(client, component, rate_limiter)
                async with output_lock:
                    results.append(result)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(serialize_fn(result)) + "\n")
                    if index % 100 == 0:
                        logger.info(
                            f"[{index}] scored {len(results)}, ${cost_tracker.cost_usd():.2f}"
                        )
            except Exception as e:
                logger.error(f"Skipping {component.component_key}: {type(e).__name__}: {e}")

    async with OpenRouter(api_key=openrouter_api_key) as client:
        input_price, output_price = await get_model_pricing(client, model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=cost_limit_usd,
        )
        limit_str = f" (limit: ${cost_limit_usd:.2f})" if cost_limit_usd is not None else ""
        print(
            f"Pricing: ${input_price * 1e6:.2f}/M input, ${output_price * 1e6:.2f}/M output{limit_str}"
        )

        await asyncio.gather(
            *[process_one(c, i, client, cost_tracker) for i, c in enumerate(remaining)]
        )

    if cost_tracker.over_budget():
        print(f"Cost limit reached: ${cost_tracker.cost_usd():.2f}")
    print(f"Final cost: ${cost_tracker.cost_usd():.2f}")
    print(f"Scored {len(results)} components -> {output_path}")
    return results

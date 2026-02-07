"""Shared LLM API utilities: retry logic, rate limiting, cost tracking, pricing."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pydantic
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
    pydantic.ValidationError,
)


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0
    limit_usd: float | None = None
    _budget_exceeded: asyncio.Event = field(default_factory=asyncio.Event)

    def add(self, input_tokens: int, output_tokens: int) -> None:
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
        async with self.lock:
            now = time.monotonic()
            self.timestamps = [t for t in self.timestamps if now - t < self.period]

            if len(self.timestamps) >= self.max_requests:
                sleep_time = self.timestamps[0] + self.period - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.timestamps = self.timestamps[1:]

            self.timestamps.append(time.monotonic())


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
) -> tuple[str, int, int]:
    """Send chat request with exponential backoff retry. Returns (content, input_tokens, output_tokens)."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
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


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1]
        s = s.rsplit("```", 1)[0]
    return s.strip()

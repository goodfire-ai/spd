"""Shared LLM API utilities for scoring: retry logic, rate limiting, cost tracking."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pydantic
from openrouter import OpenRouter
from openrouter.components import JSONSchemaConfig, ResponseFormatJSONSchema
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

MAX_RETRIES = 8
BASE_DELAY_S = 0.5
MAX_DELAY_S = 60.0
JITTER_FACTOR = 0.25

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
    def __init__(self, max_per_period: int, period_seconds: float = 60.0) -> None:
        self.max_per_period = max_per_period
        self.period_seconds = period_seconds
        self.timestamps: list[float] = []

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
            if len(self.timestamps) < self.max_per_period:
                break
            sleep_time = self.timestamps[0] + self.period_seconds - now + 0.01
            await asyncio.sleep(sleep_time)
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
    messages: list[dict[str, str]],
    max_tokens: int,
    context_label: str,
    response_format: ResponseFormatJSONSchema | None = None,
) -> tuple[str, int, int]:
    """Send chat request with retry. Returns (content, input_tokens, output_tokens)."""
    msgs: list[Any] = messages
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                messages=msgs,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = await client.chat.send_async(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            assert isinstance(content, str)
            assert response.usage is not None
            return content, int(response.usage.prompt_tokens), int(response.usage.completion_tokens)
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break
            delay = min(BASE_DELAY_S * (2**attempt), MAX_DELAY_S)
            jitter = delay * JITTER_FACTOR * random.random()
            await asyncio.sleep(delay + jitter)

    assert last_error is not None
    raise RuntimeError(f"Max retries exceeded for {context_label}: {last_error}")


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1]
        s = s.rsplit("```", 1)[0]
    return s.strip()

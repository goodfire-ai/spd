"""LLM API client with rate limiting, cost tracking, and retry."""

import asyncio
import random
import time
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

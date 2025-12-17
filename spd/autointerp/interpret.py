import asyncio
import json
import random
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from openrouter import OpenRouter
from openrouter.components import JSONSchemaConfig, MessageTypedDict, ResponseFormatJSONSchema
from openrouter.errors import (
    BadGatewayResponseError,
    EdgeNetworkTimeoutResponseError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.compute import get_model_n_blocks
from spd.autointerp.prompt_template import format_prompt_template
from spd.autointerp.schemas import ArchitectureInfo, InterpretationResult
from spd.experiments.lm.configs import LMTaskConfig
from spd.harvest.harvest import HarvestResult
from spd.harvest.schemas import ComponentData
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo

# Retry config
MAX_RETRIES = 8
BASE_DELAY_S = 0.5
MAX_DELAY_S = 60.0
JITTER_FACTOR = 0.5

RETRYABLE_ERRORS = (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    BadGatewayResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
)


class OpenRouterModelName(StrEnum):
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "3-10 word label describing what the component detects/represents",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How clear-cut the interpretation is",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the evidence and ambiguities",
        },
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}

FAKING_RESPONSES = False


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )


async def chat_with_retry(
    client: OpenRouter,
    model: str,
    messages: list[MessageTypedDict],
    response_format: ResponseFormatJSONSchema,
    max_tokens: int,
    context_label: str,
) -> tuple[str, int, int]:
    """Send chat request with exponential backoff retry. Returns (content, input_tokens, output_tokens)."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.send_async(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                response_format=response_format,
            )
            message = response.choices[0].message
            assert isinstance(message.content, str)
            assert response.usage is not None
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

            tqdm_asyncio.write(
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


async def interpret_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[InterpretationResult, int, int] | None:
    """Returns (result, input_tokens, output_tokens), or None on failure."""
    prompt = format_prompt_template(component, arch, tokenizer)

    if FAKING_RESPONSES:
        return (
            InterpretationResult(
                component_key=component.component_key,
                label="The concept of love",
                confidence="high",
                reasoning="The component fires when the word 'love' is present in the input.",
                raw_response='{"label": "The concept of love", "confidence": "high", "reasoning": "..."}',
            ),
            0,
            0,
        )

    try:
        raw, in_tok, out_tok = await chat_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchemaConfig(
                    name="interpretation",
                    schema_={**INTERPRETATION_SCHEMA, "additionalProperties": False},
                    strict=True,
                )
            ),
            max_tokens=300,
            context_label=component.component_key,
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
    reasoning = parsed["reasoning"]
    assert isinstance(label, str) and isinstance(confidence, str) and isinstance(reasoning, str)

    return (
        InterpretationResult(
            component_key=component.component_key,
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw,
        ),
        in_tok,
        out_tok,
    )


async def interpret_all(
    components: list[ComponentData],
    arch: ArchitectureInfo,
    openrouter_api_key: str,
    interpreter_model: str,
    output_path: Path,
    budget: float | None = None,
) -> list[InterpretationResult]:
    """Interpret all components with maximum parallelism. Rate limits handled via exponential backoff."""
    results: list[InterpretationResult] = []
    completed = set[str]()

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
    budget_str = f"${budget:.2f} budget" if budget else "unlimited budget"
    print(f"Interpreting {len(remaining)} components ({budget_str})")
    start_idx = len(results)

    output_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    # Scale jitter window to target ~1000 req/s initial rate
    initial_jitter_window = len(remaining) / 1000.0

    tokenizer = AutoTokenizer.from_pretrained(arch.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    async def process_one(
        component: ComponentData,
        index: int,
        client: OpenRouter,
        cost_tracker: CostTracker,
    ) -> None:
        if stop_event.is_set():
            return

        # Stagger initial requests to avoid thundering herd
        await asyncio.sleep(random.random() * initial_jitter_window)

        res = await interpret_component(client, interpreter_model, component, arch, tokenizer)
        if res is None:
            logger.error(f"Failed to interpret {component.component_key}")
            return
        result, in_tok, out_tok = res

        async with output_lock:
            if stop_event.is_set():
                return

            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            cost_tracker.add(in_tok, out_tok)

            current_cost = cost_tracker.cost_usd()
            if budget is not None and current_cost >= budget:
                tqdm_asyncio.write(f"Budget exhausted: ${current_cost:.2f} >= ${budget:.2f}")
                stop_event.set()

            if index % 100 == 0:
                tqdm_asyncio.write(
                    f"[{index}] ${current_cost:.2f} ({cost_tracker.input_tokens:,} in, {cost_tracker.output_tokens:,} out)"
                )

    async with OpenRouter(api_key=openrouter_api_key) as client:
        input_price, output_price = await get_model_pricing(client, interpreter_model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price, output_price_per_token=output_price
        )
        print(f"Pricing: ${input_price * 1e6:.2f}/M input, ${output_price * 1e6:.2f}/M output")

        await tqdm_asyncio.gather(
            *[
                process_one(c, i, client, cost_tracker)
                for i, c in enumerate(remaining, start=start_idx)
            ],
            desc="Interpreting",
        )

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
        c=model.C,
        model_class=config.pretrained_model_class,
        dataset_name=task_config.dataset_name,
        tokenizer_name=config.tokenizer_name,
    )


def run_interpret(
    wandb_path: str,
    openrouter_api_key: str,
    interpreter_model: str,
    activation_contexts_dir: Path,
    autointerp_dir: Path,
    budget: float | None = None,
) -> list[InterpretationResult]:
    arch = get_architecture_info(wandb_path)
    components = HarvestResult.load_components(activation_contexts_dir)
    output_path = autointerp_dir / "results.jsonl"

    results = asyncio.run(
        interpret_all(components, arch, openrouter_api_key, interpreter_model, output_path, budget)
    )

    print(f"Completed {len(results)} interpretations -> {output_path}")
    return results

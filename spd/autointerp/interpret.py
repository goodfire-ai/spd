import asyncio
import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from openrouter import OpenRouter
from openrouter.components import JSONSchemaConfig, ResponseFormatJSONSchema
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


class OpenRouterModelName(StrEnum):
    # HAIKU_4_5_20251001 = "anthropic/claude-haiku-4.5"  # haiku doesn't seem to support response_format via OpenRouter API
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"


class CostTracker:
    """Tracks API cost by accumulating tokens and fetching pricing from OpenRouter."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_price_per_token: float = 0
        self.output_price_per_token: float = 0

    async def init_pricing(self, client: OpenRouter, model_id: str) -> None:
        """Fetch pricing for the model from OpenRouter API."""
        response = await client.models.list_async()
        for model in response.data:
            if model.id == model_id:
                self.input_price_per_token = float(model.pricing.prompt)
                self.output_price_per_token = float(model.pricing.completion)
                return
        raise ValueError(f"Model {model_id} not found in OpenRouter models")

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_price_per_token
            + self.output_tokens * self.output_price_per_token
        )


INTERPRETATION_SCHEMA: dict[str, Any] = {
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


async def interpret_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[InterpretationResult, int, int] | None:
    """Send a single interpretation request. Returns (result, input_tokens, output_tokens)."""
    prompt = format_prompt_template(component, arch, tokenizer)

    if FAKING_RESPONSES:
        result = InterpretationResult(
            component_key=component.component_key,
            label="The concept of love",
            confidence="high",
            reasoning="The component fires when the word 'love' is present in the input.",
            raw_response="""\
{
    "label": "The concept of love",
    "confidence": "high",
    "reasoning": "The component fires when the word 'love' is present in the input."
}""",
        )
        return result, 0, 0
    else:
        response = await client.chat.send_async(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchemaConfig(
                    name="interpretation",
                    schema_={**INTERPRETATION_SCHEMA, "additionalProperties": False},
                    strict=True,
                )
            ),
        )

        message = response.choices[0].message
        assert isinstance(message.content, str), (
            f"Expected string content, got {type(message.content)}"
        )
        raw = message.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: `{raw}`")
        return None

    assert len(parsed) == 3, f"Expected 3 fields, got {len(parsed)}: {parsed}"
    assert isinstance(label := (parsed["label"]), str), (
        f"Expected 'label' to be a string, got {parsed['label']}"
    )
    assert isinstance(confidence := (parsed["confidence"]), str), (
        f"Expected 'confidence' to be a string, got {parsed['confidence']}"
    )
    assert isinstance(reasoning := (parsed["reasoning"]), str), (
        f"Expected 'reasoning' to be a string, got {parsed['reasoning']}"
    )

    result = InterpretationResult(
        component_key=component.component_key,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw,
    )

    assert response.usage is not None, "Expected usage in response"
    return result, int(response.usage.prompt_tokens), int(response.usage.completion_tokens)


async def interpret_all(
    components: list[ComponentData],
    arch: ArchitectureInfo,
    openrouter_api_key: str,
    interpreter_model: str,
    max_concurrent: int,
    output_path: Path,
    budget: float | None = None,
) -> list[InterpretationResult]:
    """Interpret all components with bounded concurrency.

    Args:
        budget: Stop after spending this much. None = unlimited.
    """
    results: list[InterpretationResult] = []
    completed = set[str]()

    # Resume: load existing results
    if output_path.exists():
        print(f"Resuming: {output_path} exists")
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(InterpretationResult(**data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already completed")

    components_mean_ci_desc = sorted(components, key=lambda c: c.mean_ci, reverse=True)
    remaining = [c for c in components_mean_ci_desc if c.component_key not in completed]
    budget_str = f"${budget:.2f} budget" if budget else "unlimited budget"
    print(f"Interpreting {len(remaining)} components ({max_concurrent} concurrent, {budget_str})")
    start_idx = len(results)

    semaphore = asyncio.Semaphore(max_concurrent)
    output_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    tokenizer = AutoTokenizer.from_pretrained(arch.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    cost_tracker = CostTracker()

    async def process_one(component: ComponentData, index: int, client: OpenRouter) -> None:
        # Check if we should stop before acquiring semaphore
        if stop_event.is_set():
            return

        async with semaphore:
            # Check again after acquiring (budget may have been exceeded while waiting)
            if stop_event.is_set():
                return

            res = await interpret_component(client, interpreter_model, component, arch, tokenizer)
            if res is None:
                logger.error(f"Failed to interpret component {component.component_key}")
                return
            result, in_tok, out_tok = res

        async with output_lock:
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            cost_tracker.add(in_tok, out_tok)

            # Check budget and stop if exceeded
            current_cost = cost_tracker.cost_usd()
            if budget is not None and current_cost >= budget and not stop_event.is_set():
                tqdm_asyncio.write(f"Budget exhausted: ${current_cost:.2f} >= ${budget:.2f}")
                stop_event.set()

            if index % 100 == 0:
                tqdm_asyncio.write(
                    f"[{index}] ${current_cost:.2f} ({cost_tracker.input_tokens:,} in, {cost_tracker.output_tokens:,} out)"
                )

    async with OpenRouter(api_key=openrouter_api_key) as client:
        await cost_tracker.init_pricing(client, interpreter_model)
        print(
            f"Pricing: ${cost_tracker.input_price_per_token * 1e6:.2f}/M input, "
            f"${cost_tracker.output_price_per_token * 1e6:.2f}/M output"
        )

        await tqdm_asyncio.gather(
            *[process_one(c, index, client) for index, c in enumerate(remaining, start=start_idx)],
            desc="Interpreting components",
        )

    print(f"Final cost: ${cost_tracker.cost_usd():.2f}")
    return results



def get_architecture_info(wandb_path: str) -> ArchitectureInfo:
    """Load architecture info from run info."""
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    n_blocks = get_model_n_blocks(model.target_model)
    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    assert config.tokenizer_name is not None, "tokenizer_name is required"
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
    max_concurrent: int,
    activation_contexts_dir: Path,
    autointerp_dir: Path,
    budget: float | None = None,
) -> list[InterpretationResult]:
    """Main entrypoint: load harvest, interpret all components, save results."""
    arch = get_architecture_info(wandb_path)

    components, _ = HarvestResult.load_components(activation_contexts_dir)

    output_path = autointerp_dir / "results.jsonl"

    results = asyncio.run(
        interpret_all(
            components,
            arch,
            openrouter_api_key,
            interpreter_model,
            max_concurrent,
            output_path,
            budget,
        )
    )

    print(f"Completed {len(results)} interpretations -> {output_path}")
    return results


def load_interpretations(out_dir: Path) -> list[InterpretationResult]:
    """Load interpretation results from disk."""
    output_path = out_dir / "results.jsonl"
    assert output_path.exists(), f"No interpretations found at {output_path}"
    results: list[InterpretationResult] = []
    with open(output_path) as f:
        for line in f:
            results.append(InterpretationResult(**json.loads(line)))
    return results

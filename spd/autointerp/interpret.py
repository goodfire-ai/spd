"""Interpret components using Claude API with asyncio for concurrency."""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anthropic
from tqdm.asyncio import tqdm_asyncio

from spd.app.backend.compute import get_model_n_blocks
from spd.autointerp.harvest import HarvestResult, load_harvest
from spd.autointerp.prompt_template import format_prompt
from spd.autointerp.schemas import (
    AUTOINTERP_DATA_DIR,
    ArchitectureInfo,
    ComponentData,
    InterpretationResult,
)
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.wandb_utils import parse_wandb_run_path

HAIKU_4_5_20251001 = "claude-haiku-4-5-20251001"

# Pricing per million tokens (as of Dec 2024)
MODEL_PRICING_IO_PER_MILLION: dict[str, tuple[float, float]] = {
    HAIKU_4_5_20251001: (1.00, 5.00),
}


@dataclass
class CostTracker:
    """Track cumulative API costs."""

    input_tokens: int = 0
    output_tokens: int = 0
    completed: int = 0
    total: int = 0
    model: str = ""

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.completed += 1

    def cost_usd(self) -> float:
        input_price, output_price = MODEL_PRICING_IO_PER_MILLION.get(self.model, (1.00, 5.00))
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

    def log(self) -> None:
        cost = self.cost_usd()
        print(
            f"  [{self.completed}/{self.total}] ${cost:.2f} "
            f"({self.input_tokens:,} in, {self.output_tokens:,} out)"
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


async def interpret_component(
    client: anthropic.AsyncAnthropic,
    model: str,
    component: ComponentData,
    arch: ArchitectureInfo,
) -> tuple[InterpretationResult, int, int]:
    """Send a single interpretation request. Returns (result, input_tokens, output_tokens)."""
    prompt = format_prompt(component, arch)

    response = await client.beta.messages.create(
        model=model,
        max_tokens=300,
        betas=["structured-outputs-2025-11-13"],
        messages=[{"role": "user", "content": prompt}],
        output_format={
            "type": "json_schema",
            "schema": INTERPRETATION_SCHEMA,
        },
    )

    content_block = response.content[0]
    assert content_block.type == "text"
    raw = content_block.text
    parsed = json.loads(raw)
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

    return result, response.usage.input_tokens, response.usage.output_tokens


async def interpret_all(
    harvest: HarvestResult,
    arch: ArchitectureInfo,
    api_key: str,
    interpreter_model: str,
    max_concurrent: int,
    output_path: Path,
) -> list[InterpretationResult]:
    """Interpret all components with bounded concurrency."""
    results: list[InterpretationResult] = []
    completed = set[str]()

    # Resume: load existing results
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(InterpretationResult(**data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already completed")

    remaining = [c for c in harvest.components if c.component_key not in completed]
    print(f"Interpreting {len(remaining)} components ({max_concurrent} concurrent)")

    semaphore = asyncio.Semaphore(max_concurrent)
    output_lock = asyncio.Lock()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    cost_tracker = CostTracker(total=len(remaining), model=interpreter_model)

    async def process_one(component: ComponentData) -> None:
        async with semaphore:
            result, in_tok, out_tok = await interpret_component(
                client, interpreter_model, component, arch
            )

        async with output_lock:
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            cost_tracker.add(in_tok, out_tok)
            cost_tracker.log()

    await tqdm_asyncio.gather(*[process_one(c) for c in remaining], desc="Interpreting components")

    return results


DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": "SimpleStories is a dataset of children's stories generated by a language model. It contains 2M+ stories, with a vocabulary of 4019 tokens.",
}


def get_architecture_info(wandb_path: str) -> ArchitectureInfo:
    """Load architecture info from run info."""
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    n_blocks = get_model_n_blocks(model.target_model)
    config = run_info.config
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    return ArchitectureInfo(
        n_layers=n_blocks,
        c=model.C,
        model_class=config.pretrained_model_class,
        dataset_name=task_config.dataset_name,
        dataset_description=DATASET_DESCRIPTIONS[task_config.dataset_name],
    )


def run_interpret(
    wandb_path: str,
    api_key: str,
    interpreter_model: str,
    max_concurrent: int,
) -> list[InterpretationResult]:
    """Main entrypoint: load harvest, interpret all components, save results."""
    _, _, run_id = parse_wandb_run_path(wandb_path)

    arch = get_architecture_info(wandb_path)

    harvest = load_harvest(run_id)

    out_dir = AUTOINTERP_DATA_DIR / run_id / "interpretations"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "results.jsonl"

    results = asyncio.run(
        interpret_all(
            harvest,
            arch,
            api_key,
            interpreter_model,
            max_concurrent,
            output_path,
        )
    )

    print(f"Completed {len(results)} interpretations -> {output_path}")
    return results


def load_interpretations(run_id: str) -> list[InterpretationResult]:
    """Load interpretation results from disk."""
    path = AUTOINTERP_DATA_DIR / run_id / "interpretations" / "results.jsonl"
    assert path.exists(), f"No interpretations found at {path}"

    results = []
    with open(path) as f:
        for line in f:
            results.append(InterpretationResult(**json.loads(line)))
    return results

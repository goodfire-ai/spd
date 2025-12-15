"""Interpret components using Claude API with trio for concurrency."""

import json
import re
from dataclasses import asdict
from pathlib import Path

import httpx
import trio

from spd.autointerp.harvest import HarvestResult, load_harvest
from spd.autointerp.prompt_template import format_prompt
from spd.autointerp.schemas import (
    AUTOINTERP_DATA_DIR,
    ArchitectureInfo,
    ComponentData,
    InterpretationResult,
)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


async def interpret_component(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    component: ComponentData,
    arch: ArchitectureInfo,
) -> InterpretationResult:
    """Send a single interpretation request."""
    prompt = format_prompt(component, arch)

    response = await client.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60.0,
    )
    response.raise_for_status()

    data = response.json()
    raw = data["content"][0]["text"]

    label, confidence, reasoning = _parse_response(raw)

    return InterpretationResult(
        component_key=component.component_key,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw,
    )


def _parse_response(text: str) -> tuple[str, str, str]:
    """Parse label/confidence/reasoning from response."""
    # Try to find the structured format
    label_match = re.search(r"Label:\s*(.+?)(?:\n|$)", text)
    conf_match = re.search(r"Confidence:\s*(\w+)", text)
    reason_match = re.search(r"Reasoning:\s*(.+?)(?:```|$)", text, re.DOTALL)

    label = label_match.group(1).strip() if label_match else "PARSE_ERROR"
    confidence = conf_match.group(1).strip().lower() if conf_match else "unknown"
    reasoning = reason_match.group(1).strip() if reason_match else text

    return label, confidence, reasoning


async def interpret_all(
    harvest: HarvestResult,
    arch: ArchitectureInfo,
    api_key: str,
    model: str,
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

    semaphore = trio.Semaphore(max_concurrent)
    output_lock = trio.Lock()

    async def process_one(client: httpx.AsyncClient, component: ComponentData) -> None:
        async with semaphore:
            result = await interpret_component(client, api_key, model, component, arch)

        async with output_lock:
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

    async with httpx.AsyncClient() as client, trio.open_nursery() as nursery:
        for component in remaining:
            nursery.start_soon(process_one, client, component)

    return results


def run_interpret(
    run_id: str,
    arch: ArchitectureInfo,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    max_concurrent: int = 50,
) -> list[InterpretationResult]:
    """Main entrypoint: load harvest, interpret all components, save results."""
    harvest = load_harvest(run_id)

    out_dir = AUTOINTERP_DATA_DIR / run_id / "interpretations"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "results.jsonl"

    results = trio.run(
        interpret_all,
        harvest,
        arch,
        api_key,
        model,
        max_concurrent,
        output_path,
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

"""Detection scoring.

Tests whether a component's interpretation label is predictive of its activations by asking
an LLM to classify plain text examples as activating or non-activating.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).

Usage:
    python -m spd.autointerp.scoring.scripts.run_scoring <wandb_path> --scorer detection
"""

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openrouter import OpenRouter

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.llm_api import (
    BudgetExceededError,
    CostTracker,
    LLMClient,
    LLMClientConfig,
    RateLimiter,
    get_model_pricing,
    make_response_format,
)
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

MAX_CONCURRENT = 50

N_ACTIVATING = 5
N_NON_ACTIVATING = 5
N_TRIALS = 5

DETECTION_RESPONSE_FORMAT = make_response_format(
    "detection_response",
    {
        "type": "object",
        "properties": {
            "activating": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "1-indexed example numbers that activate the component",
            },
        },
        "required": ["activating"],
    },
)


@dataclass
class DetectionTrial:
    predicted_activating: list[int]  # 1-indexed example numbers the LLM said activate
    actual_activating: list[int]  # ground truth 1-indexed
    tpr: float
    tnr: float
    balanced_acc: float


@dataclass
class DetectionResult:
    component_key: str
    score: float  # mean balanced accuracy across trials
    trials: list[DetectionTrial]
    n_errors: int


def _format_example_with_center_token(
    example: ActivationExample,
    app_tok: AppTokenizer,
) -> str:
    """Format an example with the center token marked with <<delimiters>>.

    Harvest windows are centered on the firing position, so the center token
    is always the one that triggered collection. We mark center for both
    activating and non-activating examples to avoid positional leakage.
    """
    valid_ids = [tid for tid in example.token_ids if tid >= 0]
    center = len(valid_ids) // 2
    spans = app_tok.get_spans(valid_ids)
    tokens = [(span, i == center) for i, span in enumerate(spans)]
    return delimit_tokens(tokens)


def _sample_activating_examples(
    component: ComponentData,
    n: int,
    rng: random.Random,
) -> list[ActivationExample]:
    """Sample activating examples from different activation strength deciles if possible."""
    examples = component.activation_examples
    if len(examples) <= n:
        return list(examples)

    # Sort by mean CI to get a spread across activation strengths
    sorted_examples = sorted(examples, key=lambda e: sum(e.ci_values) / max(len(e.ci_values), 1))
    n_examples = len(sorted_examples)

    # Pick one from each of n evenly-spaced decile bins
    sampled: list[ActivationExample] = []
    for i in range(n):
        bin_start = i * n_examples // n
        bin_end = (i + 1) * n_examples // n
        sampled.append(rng.choice(sorted_examples[bin_start:bin_end]))
    return sampled


def _sample_non_activating_examples(
    target_component: ComponentData,
    all_components: list[ComponentData],
    n: int,
    rng: random.Random,
) -> list[ActivationExample]:
    """Sample non-activating examples from other components."""
    other_components = [
        c
        for c in all_components
        if c.component_key != target_component.component_key and len(c.activation_examples) >= 1
    ]
    assert other_components, "No other components available for non-activating sampling"

    sampled: list[ActivationExample] = []
    for _ in range(n):
        donor = rng.choice(other_components)
        sampled.append(rng.choice(donor.activation_examples))
    return sampled


def _build_detection_prompt(
    label: str,
    examples_with_labels: list[tuple[str, bool]],
) -> str:
    n_total = len(examples_with_labels)

    examples_text = ""
    for i, (text, _) in enumerate(examples_with_labels):
        examples_text += f"Example {i + 1}: {text}\n\n"

    return f"""\
A neural network component has been labeled as: "{label}"

Below are {n_total} text snippets. In each, one token is marked between <<delimiters>>. \
For some examples, the marked token is one where this component fires. \
For others, the marked token is random.

{examples_text}\
Based on the label, in which examples is the <<marked>> token one where this component fires?

Respond with the list of activating example numbers."""


async def score_component(
    llm: LLMClient,
    model: str,
    component: ComponentData,
    all_components: list[ComponentData],
    app_tok: AppTokenizer,
    label: str,
) -> DetectionResult:
    assert len(component.activation_examples) >= N_ACTIVATING

    rng = random.Random(hash(component.component_key))
    trials: list[DetectionTrial] = []
    n_errors = 0

    for trial_idx in range(N_TRIALS):
        activating = _sample_activating_examples(component, N_ACTIVATING, rng)
        non_activating = _sample_non_activating_examples(
            component, all_components, N_NON_ACTIVATING, rng
        )

        formatted: list[tuple[str, bool]] = []
        for ex in activating:
            formatted.append((_format_example_with_center_token(ex, app_tok), True))
        for ex in non_activating:
            formatted.append((_format_example_with_center_token(ex, app_tok), False))

        rng.shuffle(formatted)

        prompt = _build_detection_prompt(label, formatted)

        try:
            response = await llm.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                context_label=f"{component.component_key}/trial{trial_idx}",
                response_format=DETECTION_RESPONSE_FORMAT,
            )
            parsed = json.loads(response)
            predicted_activating = set(int(x) for x in parsed["activating"])

            actual_activating = set(i + 1 for i, (_, is_act) in enumerate(formatted) if is_act)
            actual_non_activating = set(
                i + 1 for i, (_, is_act) in enumerate(formatted) if not is_act
            )

            tp = len(predicted_activating & actual_activating)
            tn = len(actual_non_activating - predicted_activating)
            tpr = tp / len(actual_activating) if actual_activating else 0.0
            tnr = tn / len(actual_non_activating) if actual_non_activating else 0.0
            balanced_acc = (tpr + tnr) / 2

            trials.append(
                DetectionTrial(
                    predicted_activating=sorted(predicted_activating),
                    actual_activating=sorted(actual_activating),
                    tpr=tpr,
                    tnr=tnr,
                    balanced_acc=balanced_acc,
                )
            )
        except Exception as e:
            logger.error(f"{component.component_key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    score = sum(t.balanced_acc for t in trials) / len(trials) if trials else 0.0

    return DetectionResult(
        component_key=component.component_key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


def _deserialize_result(data: dict[str, Any]) -> DetectionResult:
    return DetectionResult(
        component_key=data["component_key"],
        score=data["score"],
        trials=[DetectionTrial(**t) for t in data["trials"]],
        n_errors=data["n_errors"],
    )


async def run_detection_scoring(
    components: list[ComponentData],
    labels: dict[str, str],
    model: str,
    openrouter_api_key: str,
    tokenizer_name: str,
    output_path: Path,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[DetectionResult]:
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    eligible = [
        c
        for c in components
        if c.component_key in labels and len(c.activation_examples) >= N_ACTIVATING
    ]
    if limit is not None:
        eligible = eligible[:limit]

    llm_config = LLMClientConfig(
        openrouter_api_key=openrouter_api_key,
        model=model,
        cost_limit_usd=cost_limit_usd,
    )

    results: list[DetectionResult] = []
    completed = set[str]()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(_deserialize_result(data))
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    print(f"Scoring {len(remaining)} components")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    output_lock = asyncio.Lock()

    async def process_one(component: ComponentData, index: int, llm: LLMClient) -> None:
        async with semaphore:
            try:
                result = await score_component(
                    llm, model, component, components, app_tok, labels[component.component_key]
                )
            except BudgetExceededError:
                return
            except Exception as e:
                logger.error(f"Skipping {component.component_key}: {type(e).__name__}: {e}")
                return
            async with output_lock:
                results.append(result)
                with open(output_path, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
                if index % 100 == 0:
                    logger.info(
                        f"[{index}] scored {len(results)}, ${llm.cost_tracker.cost_usd():.2f}"
                    )

    async with OpenRouter(api_key=llm_config.openrouter_api_key) as api:
        input_price, output_price = await get_model_pricing(api, llm_config.model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=llm_config.cost_limit_usd,
        )
        llm = LLMClient(
            api=api,
            rate_limiter=RateLimiter(llm_config.max_requests_per_minute),
            cost_tracker=cost_tracker,
        )

        await asyncio.gather(*[process_one(c, i, llm) for i, c in enumerate(remaining)])

        print(f"Final cost: ${cost_tracker.cost_usd():.2f}")

    print(f"Scored {len(results)} components -> {output_path}")
    return results

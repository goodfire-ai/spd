"""Detection scoring.

Tests whether a component's interpretation label is predictive of its activations by asking
an LLM to classify a mix of activating and non-activating examples.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).

Usage:
    python -m spd.autointerp.scoring.scripts.run_scoring <wandb_path> --scorer detection
"""

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from openrouter import OpenRouter
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.autointerp.llm_api import (
    CostTracker,
    RateLimiter,
    chat_with_retry,
    get_model_pricing,
    make_response_format,
)
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

N_ACTIVATING = 5
N_NON_ACTIVATING = 5
N_TRIALS = 5
MAX_TOKENS_PER_EXAMPLE = 64
CI_THRESHOLD = 0.3
MAX_CONCURRENT_REQUESTS = 50
MAX_REQUESTS_PER_MINUTE = 200

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


def _format_activating_example(
    example: ActivationExample,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Format an activating example with high-CI tokens bolded.

    NOTE: Potential leakage — real CI patterns have spatial structure (consecutive tokens
    tend to be co-active), while non-activating examples use i.i.d. Bernoulli bolding.
    An LLM could distinguish bursty vs uniform bold patterns without understanding the label.
    Fixing this properly would require matching the run-length distribution of bold spans.
    """
    tokens: list[str] = []
    for tid, ci in zip(example.token_ids, example.ci_values, strict=True):
        if tid < 0:
            continue
        decoded = tokenizer.decode([tid])
        if ci > CI_THRESHOLD:
            tokens.append(f"**{decoded}**")
        else:
            tokens.append(decoded)
    return "".join(tokens[:MAX_TOKENS_PER_EXAMPLE])


def _measure_bold_density(examples: list[ActivationExample]) -> float:
    """Measure the fraction of valid tokens above CI_THRESHOLD across activating examples."""
    n_bold = 0
    n_total = 0
    for ex in examples:
        for tid, ci in zip(ex.token_ids, ex.ci_values, strict=True):
            if tid < 0:
                continue
            n_total += 1
            if ci > CI_THRESHOLD:
                n_bold += 1
    return n_bold / n_total if n_total > 0 else 0.0


def _format_non_activating_example(
    example: ActivationExample,
    tokenizer: PreTrainedTokenizerBase,
    rng: random.Random,
    bold_density: float,
) -> str:
    """Format a non-activating example with random tokens bolded to match activating density."""
    tokens: list[str] = []
    for tid in example.token_ids:
        if tid < 0:
            continue
        decoded = tokenizer.decode([tid])
        if rng.random() < bold_density:
            tokens.append(f"**{decoded}**")
        else:
            tokens.append(decoded)
    return "".join(tokens[:MAX_TOKENS_PER_EXAMPLE])


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
    """Build the detection prompt.

    Args:
        label: The interpretation label for the component.
        examples_with_labels: List of (formatted_text, is_activating) in shuffled order.
            The bool is NOT shown to the LLM -- it's used only for scoring.
    """
    n_total = len(examples_with_labels)

    examples_text = ""
    for i, (text, _) in enumerate(examples_with_labels):
        examples_text += f"Example {i + 1}: {text}\n\n"

    return f"""\
You are evaluating whether a neural network component's interpretation label is predictive \
of its activations.

The component has been labeled as: "{label}"

Below are {n_total} text examples. Some of them activate this component, and some do not.
Tokens in **bold** are highlighted for emphasis.

{examples_text}
For each example, decide whether it activates the component described by the label above.

Respond with the list of activating example numbers."""


async def score_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    all_components: list[ComponentData],
    tokenizer: PreTrainedTokenizerBase,
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

        bold_density = _measure_bold_density(activating)

        # Format examples
        formatted: list[tuple[str, bool]] = []
        for ex in activating:
            formatted.append((_format_activating_example(ex, tokenizer), True))
        for ex in non_activating:
            formatted.append(
                (_format_non_activating_example(ex, tokenizer, rng, bold_density), False)
            )

        # Shuffle
        rng.shuffle(formatted)

        prompt = _build_detection_prompt(label, formatted)

        try:
            response, _, _ = await chat_with_retry(
                client=client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                context_label=f"{component.component_key}/trial{trial_idx}",
                response_format=DETECTION_RESPONSE_FORMAT,
            )
            parsed = json.loads(response)
            predicted_activating = set(int(x) for x in parsed["activating"])

            # Ground truth: which example numbers are actually activating
            actual_activating = set(i + 1 for i, (_, is_act) in enumerate(formatted) if is_act)
            actual_non_activating = set(
                i + 1 for i, (_, is_act) in enumerate(formatted) if not is_act
            )

            # Balanced accuracy = (TPR + TNR) / 2
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    results: list[DetectionResult] = []
    completed = set[str]()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(
                    DetectionResult(
                        component_key=data["component_key"],
                        score=data["score"],
                        trials=[DetectionTrial(**t) for t in data["trials"]],
                        n_errors=data["n_errors"],
                    )
                )
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already scored")

    eligible = [
        c
        for c in components
        if c.component_key not in completed
        and c.component_key in labels
        and len(c.activation_examples) >= N_ACTIVATING
    ]
    if limit is not None:
        eligible = eligible[:limit]
    print(f"Scoring {len(eligible)} components")

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
        await rate_limiter.acquire()
        async with semaphore:
            if cost_tracker.over_budget():
                return
            try:
                result = await score_component(
                    client,
                    model,
                    component,
                    components,
                    tokenizer,
                    labels[component.component_key],
                )
                async with output_lock:
                    results.append(result)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(asdict(result)) + "\n")
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
            *[process_one(c, i, client, cost_tracker) for i, c in enumerate(eligible)]
        )

    if cost_tracker.over_budget():
        print(f"Cost limit reached: ${cost_tracker.cost_usd():.2f}")
    print(f"Final cost: ${cost_tracker.cost_usd():.2f}")
    print(f"Scored {len(results)} components -> {output_path}")
    return results

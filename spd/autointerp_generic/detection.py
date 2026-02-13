"""Detection scoring (label predictiveness).

Tests whether a component's interpretation label is predictive of its activations by asking
an LLM to classify plain text examples as activating or non-activating.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).
"""

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.llm_api import (
    BudgetExceededError,
    CostTracker,
    GlobalBackoff,
    LLMClient,
    get_model_pricing,
    make_response_format,
)
from spd.autointerp_generic.db import GenericInterpDB
from spd.autointerp_generic.types import (
    ActivatingExample,
    ComponentAutointerpData,
    DecompositionAutointerpData,
    EvalConfig,
)
from spd.log import logger

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
    predicted_activating: list[int]
    actual_activating: list[int]
    tpr: float
    tnr: float
    balanced_acc: float


@dataclass
class DetectionResult:
    component_key: str
    score: float
    trials: list[DetectionTrial]
    n_errors: int


def _format_example_with_center_token(
    example: ActivatingExample,
    app_tok: AppTokenizer,
) -> str:
    """Format an example with the center token marked with <<delimiters>>.

    Windows are centered on the firing position, so the center token
    is always the one that triggered collection. We mark center for both
    activating and non-activating examples to avoid positional leakage.
    """
    spans = app_tok.get_spans(example.tokens)
    center = len(spans) // 2
    tokens = [(span, i == center) for i, span in enumerate(spans)]
    return delimit_tokens(tokens)


def _sample_activating_examples(
    component: ComponentAutointerpData,
    n: int,
    rng: random.Random,
) -> list[ActivatingExample]:
    """Sample activating examples spread across activation strengths."""
    examples = component.activating_examples
    if len(examples) <= n:
        return list(examples)

    # Sort by bold density to get a spread
    sorted_examples = sorted(examples, key=lambda e: sum(e.bold) / max(len(e.bold), 1))
    n_examples = len(sorted_examples)

    sampled: list[ActivatingExample] = []
    for i in range(n):
        bin_start = i * n_examples // n
        bin_end = (i + 1) * n_examples // n
        sampled.append(rng.choice(sorted_examples[bin_start:bin_end]))
    return sampled


def _sample_non_activating_examples(
    target: ComponentAutointerpData,
    all_components: list[ComponentAutointerpData],
    n: int,
    rng: random.Random,
) -> list[ActivatingExample]:
    """Sample non-activating examples from other components."""
    other = [c for c in all_components if c.key != target.key and len(c.activating_examples) >= 1]
    assert other, "No other components available for non-activating sampling"

    sampled: list[ActivatingExample] = []
    for _ in range(n):
        donor = rng.choice(other)
        sampled.append(rng.choice(donor.activating_examples))
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
    config: EvalConfig,
    component: ComponentAutointerpData,
    all_components: list[ComponentAutointerpData],
    app_tok: AppTokenizer,
    label: str,
    n_activating: int,
    n_non_activating: int,
    n_trials: int,
) -> DetectionResult:
    assert len(component.activating_examples) >= n_activating

    rng = random.Random()
    trials: list[DetectionTrial] = []
    n_errors = 0

    for trial_idx in range(n_trials):
        activating = _sample_activating_examples(component, n_activating, rng)
        non_activating = _sample_non_activating_examples(
            component, all_components, n_non_activating, rng
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
                model=config.model,
                reasoning=Reasoning(effort=config.reasoning_effort),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                context_label=f"{component.key}/trial{trial_idx}",
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
            logger.error(f"{component.key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    score = sum(t.balanced_acc for t in trials) / len(trials) if trials else 0.0

    return DetectionResult(
        component_key=component.key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


def run_detection_scoring(
    data: DecompositionAutointerpData,
    labels: dict[str, str],
    openrouter_api_key: str,
    db_path: Path,
    config: EvalConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[DetectionResult]:
    n_activating = config.detection_n_activating
    n_non_activating = config.detection_n_non_activating
    n_trials = config.detection_n_trials

    eligible = [
        c for c in data.components if c.key in labels and len(c.activating_examples) >= n_activating
    ]
    if limit is not None:
        eligible = eligible[:limit]

    async def _run() -> list[DetectionResult]:
        db = GenericInterpDB(db_path)
        results: list[DetectionResult] = []

        existing_scores = db.get_scores("detection")
        completed = set(existing_scores.keys())
        if completed:
            logger.info(f"Resuming: {len(completed)} already scored")

        remaining = [c for c in eligible if c.key not in completed]
        logger.info(f"Scoring {len(remaining)} components")

        semaphore = asyncio.Semaphore(config.detection_max_concurrent)
        output_lock = asyncio.Lock()

        async def process_one(
            component: ComponentAutointerpData, index: int, llm: LLMClient
        ) -> None:
            async with semaphore:
                try:
                    result = await score_component(
                        llm,
                        config,
                        component,
                        data.components,
                        data.tokenizer,
                        labels[component.key],
                        n_activating=n_activating,
                        n_non_activating=n_non_activating,
                        n_trials=n_trials,
                    )
                except BudgetExceededError:
                    return
                except Exception as e:
                    logger.error(f"Skipping {component.key}: {type(e).__name__}: {e}")
                    return
                async with output_lock:
                    results.append(result)
                    details = json.dumps(asdict(result))
                    db.save_score(result.component_key, "detection", result.score, details)
                    if index % 100 == 0:
                        logger.info(
                            f"[{index}] scored {len(results)}, ${llm.cost_tracker.cost_usd():.2f}"
                        )

        async with OpenRouter(api_key=openrouter_api_key) as api:
            input_price, output_price = await get_model_pricing(api, config.model)
            cost_tracker = CostTracker(
                input_price_per_token=input_price,
                output_price_per_token=output_price,
                limit_usd=cost_limit_usd,
            )
            llm = LLMClient(
                api=api,
                rate_limiter=AsyncLimiter(max_rate=config.max_requests_per_minute, time_period=60),
                backoff=GlobalBackoff(),
                cost_tracker=cost_tracker,
            )

            await asyncio.gather(*[process_one(c, i, llm) for i, c in enumerate(remaining)])

            logger.info(f"Final cost: ${cost_tracker.cost_usd():.2f}")

        db.close()
        logger.info(f"Scored {len(results)} components")
        return results

    return asyncio.run(_run())

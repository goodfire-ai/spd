"""Fuzzing scoring.

Tests the *specificity* of an interpretation label by checking if an LLM can
distinguish correctly-highlighted activating tokens from incorrectly-highlighted ones.
Catches labels that are too vague or generic.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openrouter import OpenRouter

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.llm_api import (
    RateLimiter,
    chat_with_retry,
    make_response_format,
    run_scoring_pipeline,
)
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

N_CORRECT = 5
N_INCORRECT = 2
N_TRIALS = 5

FUZZING_RESPONSE_FORMAT = make_response_format(
    "fuzzing_response",
    {
        "type": "object",
        "properties": {
            "correct_examples": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "1-indexed example numbers with correct highlighting",
            },
            "reasoning": {"type": "string", "description": "Brief explanation"},
        },
        "required": ["correct_examples", "reasoning"],
    },
)


@dataclass
class FuzzingTrial:
    correct_positions: list[int]  # 1-indexed positions with correct highlighting
    predicted_correct: list[int]  # what the LLM said was correct
    tp: int
    tn: int
    n_correct: int
    n_incorrect: int


@dataclass
class FuzzingResult:
    component_key: str
    score: float  # balanced accuracy = (TPR + TNR) / 2
    trials: list[FuzzingTrial]
    n_errors: int


def _delimit_high_ci_tokens(
    example: ActivationExample,
    app_tok: AppTokenizer,
    ci_threshold: float,
) -> tuple[str, int]:
    """Format example with high-CI tokens in <<delimiters>>. Returns (text, n_delimited)."""
    valid = [
        (tid, ci) for tid, ci in zip(example.token_ids, example.ci_values, strict=True) if tid >= 0
    ]
    spans = app_tok.get_spans([tid for tid, _ in valid])
    tokens = [(span, ci > ci_threshold) for span, (_, ci) in zip(spans, valid, strict=True)]
    n_delimited = sum(1 for _, active in tokens if active)
    return delimit_tokens(tokens), n_delimited


def _delimit_random_low_ci_tokens(
    example: ActivationExample,
    app_tok: AppTokenizer,
    n_to_delimit: int,
    rng: random.Random,
    ci_threshold: float,
) -> str:
    """Format example with random LOW-CI tokens in <<delimiters>> instead of high-CI ones."""
    valid = [
        (tid, ci) for tid, ci in zip(example.token_ids, example.ci_values, strict=True) if tid >= 0
    ]
    low_ci_indices = [j for j, (_, ci) in enumerate(valid) if ci <= ci_threshold]

    if len(low_ci_indices) < n_to_delimit:
        delimit_set = set(low_ci_indices)
    else:
        delimit_set = set(rng.sample(low_ci_indices, n_to_delimit))

    spans = app_tok.get_spans([tid for tid, _ in valid])
    tokens = [(span, j in delimit_set) for j, span in enumerate(spans)]
    return delimit_tokens(tokens)


def _build_fuzzing_prompt(
    label: str,
    formatted_examples: list[tuple[str, bool]],
) -> str:
    """Build the fuzzing detection prompt.

    Args:
        label: The interpretation label for this component.
        formatted_examples: List of (formatted_text, is_correct) tuples, already shuffled.
    """
    n_examples = len(formatted_examples)

    examples_text = ""
    for i, (text, _) in enumerate(formatted_examples):
        examples_text += f"Example {i + 1}: {text}\n\n"

    return f"""\
A neural network component has been interpreted as: "{label}"

Below are {n_examples} text examples where this component is active. In each example, some tokens \
are marked between <<delimiters>>. In some examples, the <<delimited>> tokens correctly indicate \
where the component fires most strongly. In other examples, the <<delimited>> tokens are random \
and unrelated to the component's actual firing pattern.

{examples_text}\
Based on the interpretation "{label}", which examples have correctly-marked tokens \
(consistent with the label) vs. randomly-marked tokens?

Respond with the list of correctly-highlighted example numbers and brief reasoning.\
"""


async def score_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    app_tok: AppTokenizer,
    label: str,
    ci_threshold: float,
    rate_limiter: RateLimiter | None = None,
) -> FuzzingResult:
    min_examples = N_CORRECT + N_INCORRECT
    assert len(component.activation_examples) >= min_examples

    rng = random.Random(hash(component.component_key))
    trials: list[FuzzingTrial] = []
    n_errors = 0
    total_tp = 0
    total_tn = 0
    total_pos = 0
    total_neg = 0

    for trial_idx in range(N_TRIALS):
        sampled = rng.sample(component.activation_examples, N_CORRECT + N_INCORRECT)
        correct_examples = sampled[:N_CORRECT]
        incorrect_examples = sampled[N_CORRECT:]

        formatted: list[tuple[str, bool]] = []

        for ex in correct_examples:
            text, _ = _delimit_high_ci_tokens(ex, app_tok, ci_threshold)
            formatted.append((text, True))

        for ex in incorrect_examples:
            # Count how many tokens would be bolded in the correct version
            _, n_delimited = _delimit_high_ci_tokens(ex, app_tok, ci_threshold)
            n_to_delimit = max(n_delimited, 1)
            text = _delimit_random_low_ci_tokens(ex, app_tok, n_to_delimit, rng, ci_threshold)
            formatted.append((text, False))

        rng.shuffle(formatted)

        # Track which 1-indexed positions are correct
        correct_positions = {i + 1 for i, (_, is_correct) in enumerate(formatted) if is_correct}
        incorrect_positions = {
            i + 1 for i, (_, is_correct) in enumerate(formatted) if not is_correct
        }

        prompt = _build_fuzzing_prompt(label, formatted)

        try:
            response, _, _ = await chat_with_retry(
                client=client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                context_label=f"{component.component_key}/trial{trial_idx}",
                response_format=FUZZING_RESPONSE_FORMAT,
                rate_limiter=rate_limiter,
            )
            parsed = json.loads(response)
            predicted_correct = set(parsed["correct_examples"])

            # True positives: correctly identified as correct
            tp = len(correct_positions & predicted_correct)
            # True negatives: correctly identified as incorrect (not in predicted)
            tn = len(incorrect_positions - predicted_correct)

            total_tp += tp
            total_tn += tn
            total_pos += len(correct_positions)
            total_neg += len(incorrect_positions)

            trials.append(
                FuzzingTrial(
                    correct_positions=sorted(correct_positions),
                    predicted_correct=sorted(predicted_correct),
                    tp=tp,
                    tn=tn,
                    n_correct=len(correct_positions),
                    n_incorrect=len(incorrect_positions),
                )
            )
        except Exception as e:
            logger.error(f"{component.component_key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    # Balanced accuracy = (TPR + TNR) / 2
    tpr = total_tp / total_pos if total_pos > 0 else 0.0
    tnr = total_tn / total_neg if total_neg > 0 else 0.0
    score = (tpr + tnr) / 2 if (total_pos > 0 and total_neg > 0) else 0.0

    return FuzzingResult(
        component_key=component.component_key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


def _deserialize_result(data: dict[str, Any]) -> FuzzingResult:
    return FuzzingResult(
        component_key=data["component_key"],
        score=data["score"],
        trials=[FuzzingTrial(**t) for t in data["trials"]],
        n_errors=data["n_errors"],
    )


async def run_fuzzing_scoring(
    components: list[ComponentData],
    labels: dict[str, str],
    model: str,
    openrouter_api_key: str,
    tokenizer_name: str,
    output_path: Path,
    ci_threshold: float,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[FuzzingResult]:
    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    min_examples = N_CORRECT + N_INCORRECT
    eligible = [
        c
        for c in components
        if c.component_key in labels and len(c.activation_examples) >= min_examples
    ]
    if limit is not None:
        eligible = eligible[:limit]

    async def _score(
        client: OpenRouter, component: ComponentData, rate_limiter: RateLimiter
    ) -> FuzzingResult:
        return await score_component(
            client, model, component, app_tok, labels[component.component_key], ci_threshold,
            rate_limiter,
        )

    return await run_scoring_pipeline(
        eligible=eligible,
        score_fn=_score,
        serialize_fn=asdict,
        deserialize_fn=_deserialize_result,
        model=model,
        openrouter_api_key=openrouter_api_key,
        output_path=output_path,
        cost_limit_usd=cost_limit_usd,
    )

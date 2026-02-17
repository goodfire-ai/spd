"""Fuzzing scoring.

Tests the *specificity* of an interpretation label by checking if an LLM can
distinguish correctly-highlighted activating tokens from incorrectly-highlighted ones.
Catches labels that are too vague or generic.

Based on: EleutherAI's sae-auto-interp (https://blog.eleuther.ai/autointerp/).
"""

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import AutointerpEvalConfig
from spd.autointerp.db import InterpDB
from spd.autointerp.llm_api import (
    CostTracker,
    GlobalBackoff,
    LLMClient,
    LLMError,
    LLMJob,
    LLMResult,
    get_model_pricing,
    map_api_calls,
)
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

FUZZING_SCHEMA = {
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
}


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


def _delimit_tokens(
    example: ActivationExample,
    app_tok: AppTokenizer,
) -> tuple[str, int]:
    """Format example with firing tokens in <<delimiters>>. Returns (text, n_delimited)."""
    spans = app_tok.get_spans(example.token_ids)
    tokens = [(span, firing) for span, firing in zip(spans, example.firings, strict=True)]
    n_delimited = sum(example.firings)
    return delimit_tokens(tokens), n_delimited


def _delimit_random_tokens(
    example: ActivationExample,
    app_tok: AppTokenizer,
    n_to_delimit: int,
    rng: random.Random,
) -> str:
    """Format example with random tokens in <<delimiters>> instead of firing ones."""
    n_toks = len(example.token_ids)

    delimit_set = set(rng.sample(range(n_toks), min(n_to_delimit, n_toks)))
    spans = app_tok.get_spans(example.token_ids)
    tokens = [(span, j in delimit_set) for j, span in enumerate(spans)]
    return delimit_tokens(tokens)


def _build_fuzzing_prompt(
    label: str,
    formatted_examples: list[tuple[str, bool]],
) -> str:
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


@dataclass
class _TrialGroundTruth:
    component_key: str
    correct_positions: set[int]
    incorrect_positions: set[int]


async def run_fuzzing_scoring(
    components: list[ComponentData],
    labels: dict[str, str],
    model: str,
    openrouter_api_key: str,
    tokenizer_name: str,
    db: InterpDB,
    eval_config: AutointerpEvalConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[FuzzingResult]:
    n_correct = eval_config.fuzzing_n_correct
    n_incorrect = eval_config.fuzzing_n_incorrect
    n_trials = eval_config.fuzzing_n_trials
    max_concurrent = eval_config.fuzzing_max_concurrent

    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    min_examples = n_correct + n_incorrect
    eligible = [
        c
        for c in components
        if c.component_key in labels and len(c.activation_examples) >= min_examples
    ]
    if limit is not None:
        eligible = eligible[:limit]

    existing_scores = db.get_scores("fuzzing")
    completed = set(existing_scores.keys())
    if completed:
        logger.info(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    logger.info(f"Scoring {len(remaining)} components ({len(remaining) * n_trials} trials)")

    rng = random.Random()
    jobs: list[LLMJob] = []
    ground_truth: dict[str, _TrialGroundTruth] = {}

    for component in remaining:
        label = labels[component.component_key]
        for trial_idx in range(n_trials):
            sampled = rng.sample(component.activation_examples, n_correct + n_incorrect)
            correct_examples = sampled[:n_correct]
            incorrect_examples = sampled[n_correct:]

            formatted: list[tuple[str, bool]] = []
            for ex in correct_examples:
                text, _ = _delimit_tokens(ex, app_tok)
                formatted.append((text, True))
            for ex in incorrect_examples:
                _, n_delimited = _delimit_tokens(ex, app_tok)
                n_to_delimit = max(n_delimited, 1)
                text = _delimit_random_tokens(ex, app_tok, n_to_delimit, rng)
                formatted.append((text, False))
            rng.shuffle(formatted)

            key = f"{component.component_key}/trial{trial_idx}"
            correct_pos = {i + 1 for i, (_, is_correct) in enumerate(formatted) if is_correct}
            incorrect_pos = {i + 1 for i, (_, is_correct) in enumerate(formatted) if not is_correct}
            jobs.append(
                LLMJob(
                    prompt=_build_fuzzing_prompt(label, formatted), schema=FUZZING_SCHEMA, key=key
                )
            )
            ground_truth[key] = _TrialGroundTruth(
                component_key=component.component_key,
                correct_positions=correct_pos,
                incorrect_positions=incorrect_pos,
            )

    component_trials: defaultdict[str, list[FuzzingTrial]] = defaultdict(list)
    component_errors: defaultdict[str, int] = defaultdict(int)

    async with OpenRouter(api_key=openrouter_api_key) as api:
        input_price, output_price = await get_model_pricing(api, model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=cost_limit_usd,
        )
        llm = LLMClient(
            api=api,
            rate_limiter=AsyncLimiter(max_rate=eval_config.max_requests_per_minute, time_period=60),
            backoff=GlobalBackoff(),
            cost_tracker=cost_tracker,
        )

        async for outcome in map_api_calls(
            jobs,
            llm,
            model=model,
            max_tokens=5000,
            reasoning=Reasoning(effort=eval_config.reasoning_effort),
            max_concurrent=max_concurrent,
        ):
            match outcome:
                case LLMResult(job=job, parsed=parsed):
                    gt = ground_truth[job.key]
                    predicted_correct = set(parsed["correct_examples"])
                    tp = len(gt.correct_positions & predicted_correct)
                    tn = len(gt.incorrect_positions - predicted_correct)
                    component_trials[gt.component_key].append(
                        FuzzingTrial(
                            correct_positions=sorted(gt.correct_positions),
                            predicted_correct=sorted(predicted_correct),
                            tp=tp,
                            tn=tn,
                            n_correct=len(gt.correct_positions),
                            n_incorrect=len(gt.incorrect_positions),
                        )
                    )
                case LLMError(job=job, error=e):
                    gt = ground_truth[job.key]
                    component_errors[gt.component_key] += 1
                    logger.error(f"{job.key}: {type(e).__name__}: {e}")

        logger.info(f"Final cost: ${cost_tracker.cost_usd():.2f}")

    results: list[FuzzingResult] = []
    for component in remaining:
        ck = component.component_key
        trials = component_trials.get(ck, [])
        n_err = component_errors.get(ck, 0)
        total_tp = sum(t.tp for t in trials)
        total_tn = sum(t.tn for t in trials)
        total_pos = sum(t.n_correct for t in trials)
        total_neg = sum(t.n_incorrect for t in trials)
        tpr = total_tp / total_pos if total_pos > 0 else 0.0
        tnr = total_tn / total_neg if total_neg > 0 else 0.0
        score = (tpr + tnr) / 2 if (total_pos > 0 and total_neg > 0) else 0.0
        result = FuzzingResult(component_key=ck, score=score, trials=trials, n_errors=n_err)
        results.append(result)
        db.save_score(ck, "fuzzing", score, json.dumps(asdict(result)))

    logger.info(f"Scored {len(results)} components")
    return results

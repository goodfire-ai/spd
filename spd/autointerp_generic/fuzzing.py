"""Fuzzing scoring (label specificity).

Tests the specificity of an interpretation label by checking if an LLM can
distinguish correctly-highlighted activating tokens from incorrectly-highlighted ones.

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
    correct_positions: list[int]
    predicted_correct: list[int]
    tp: int
    tn: int
    n_correct: int
    n_incorrect: int


@dataclass
class FuzzingResult:
    component_key: str
    score: float
    trials: list[FuzzingTrial]
    n_errors: int


def _delimit_bold_tokens(
    example: ActivatingExample,
    app_tok: AppTokenizer,
) -> tuple[str, int]:
    """Format example with bold tokens in <<delimiters>>. Returns (text, n_delimited)."""
    spans = app_tok.get_spans(example.tokens)
    assert len(spans) == len(example.bold)
    tokens = list(zip(spans, example.bold, strict=True))
    n_delimited = sum(example.bold)
    return delimit_tokens(tokens), n_delimited


def _delimit_random_non_bold_tokens(
    example: ActivatingExample,
    app_tok: AppTokenizer,
    n_to_delimit: int,
    rng: random.Random,
) -> str:
    """Format example with random non-bold tokens in <<delimiters>> instead of bold ones."""
    non_bold_indices = [j for j, b in enumerate(example.bold) if not b]

    if len(non_bold_indices) < n_to_delimit:
        delimit_set = set(non_bold_indices)
    else:
        delimit_set = set(rng.sample(non_bold_indices, n_to_delimit))

    spans = app_tok.get_spans(example.tokens)
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


async def score_component(
    llm: LLMClient,
    config: EvalConfig,
    component: ComponentAutointerpData,
    app_tok: AppTokenizer,
    label: str,
    n_correct: int,
    n_incorrect: int,
    n_trials: int,
) -> FuzzingResult:
    min_examples = n_correct + n_incorrect
    assert len(component.activating_examples) >= min_examples

    rng = random.Random()
    trials: list[FuzzingTrial] = []
    n_errors = 0
    total_tp = 0
    total_tn = 0
    total_pos = 0
    total_neg = 0

    for trial_idx in range(n_trials):
        sampled = rng.sample(component.activating_examples, n_correct + n_incorrect)
        correct_examples = sampled[:n_correct]
        incorrect_examples = sampled[n_correct:]

        formatted: list[tuple[str, bool]] = []

        for ex in correct_examples:
            text, _ = _delimit_bold_tokens(ex, app_tok)
            formatted.append((text, True))

        for ex in incorrect_examples:
            _, n_delimited = _delimit_bold_tokens(ex, app_tok)
            n_to_delimit = max(n_delimited, 1)
            text = _delimit_random_non_bold_tokens(ex, app_tok, n_to_delimit, rng)
            formatted.append((text, False))

        rng.shuffle(formatted)

        correct_positions = {i + 1 for i, (_, is_correct) in enumerate(formatted) if is_correct}
        incorrect_positions = {
            i + 1 for i, (_, is_correct) in enumerate(formatted) if not is_correct
        }

        prompt = _build_fuzzing_prompt(label, formatted)

        try:
            response = await llm.chat(
                model=config.model,
                reasoning=Reasoning(effort=config.reasoning_effort),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                context_label=f"{component.key}/trial{trial_idx}",
                response_format=FUZZING_RESPONSE_FORMAT,
            )
            parsed = json.loads(response)
            predicted_correct = set(parsed["correct_examples"])

            tp = len(correct_positions & predicted_correct)
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
            logger.error(f"{component.key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    tpr = total_tp / total_pos if total_pos > 0 else 0.0
    tnr = total_tn / total_neg if total_neg > 0 else 0.0
    score = (tpr + tnr) / 2 if (total_pos > 0 and total_neg > 0) else 0.0

    return FuzzingResult(
        component_key=component.key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


def run_fuzzing_scoring(
    data: DecompositionAutointerpData,
    labels: dict[str, str],
    openrouter_api_key: str,
    db_path: Path,
    config: EvalConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[FuzzingResult]:
    n_correct = config.fuzzing_n_correct
    n_incorrect = config.fuzzing_n_incorrect
    n_trials = config.fuzzing_n_trials
    min_examples = n_correct + n_incorrect

    eligible = [
        c for c in data.components if c.key in labels and len(c.activating_examples) >= min_examples
    ]
    if limit is not None:
        eligible = eligible[:limit]

    async def _run() -> list[FuzzingResult]:
        db = GenericInterpDB(db_path)
        results: list[FuzzingResult] = []

        existing_scores = db.get_scores("fuzzing")
        completed = set(existing_scores.keys())
        if completed:
            logger.info(f"Resuming: {len(completed)} already scored")

        remaining = [c for c in eligible if c.key not in completed]
        logger.info(f"Scoring {len(remaining)} components")

        semaphore = asyncio.Semaphore(config.fuzzing_max_concurrent)
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
                        data.tokenizer,
                        labels[component.key],
                        n_correct=n_correct,
                        n_incorrect=n_incorrect,
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
                    db.save_score(result.component_key, "fuzzing", result.score, details)
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

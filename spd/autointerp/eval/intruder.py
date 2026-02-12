"""Intruder detection scoring.

Tests whether a component's activating examples are coherent by asking an LLM
to identify an "intruder" example drawn from a different component. No labels needed.

Based on: "Evaluating SAE interpretability without explanations" (2025).

Usage:
    python -m spd.autointerp.scoring.scripts.run_intruder <wandb_path> --limit 100
"""

import asyncio
import bisect
import json
import random
from dataclasses import asdict, dataclass

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Reasoning

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import IntruderEvalConfig, ReasoningEffort
from spd.autointerp.llm_api import (
    BudgetExceededError,
    CostTracker,
    GlobalBackoff,
    LLMClient,
    get_model_pricing,
    make_response_format,
)
from spd.harvest.db import HarvestDB
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

INTRUDER_RESPONSE_FORMAT = make_response_format(
    "intruder_response",
    {
        "type": "object",
        "properties": {
            "intruder": {
                "type": "integer",
                "description": "1-indexed example number of the intruder",
            },
            "reasoning": {"type": "string", "description": "Brief explanation"},
        },
        "required": ["intruder", "reasoning"],
    },
)


@dataclass
class IntruderTrial:
    correct_answer: int
    predicted: int
    is_correct: bool


@dataclass
class IntruderResult:
    component_key: str
    score: float
    trials: list[IntruderTrial]
    n_errors: int


def _bold_density(component: ComponentData, ci_threshold: float) -> float:
    """Fraction of valid tokens above ci_threshold across a component's activation examples."""
    n_bold = 0
    n_total = 0
    for ex in component.activation_examples:
        for tid, ci in zip(ex.token_ids, ex.ci_values, strict=True):
            if tid < 0:
                continue
            n_total += 1
            if ci > ci_threshold:
                n_bold += 1
    return n_bold / n_total if n_total > 0 else 0.0


class DensityIndex:
    """Index of components sorted by bold density for efficient similar-density lookup."""

    def __init__(
        self, components: list[ComponentData], min_examples: int, ci_threshold: float
    ) -> None:
        eligible = [c for c in components if len(c.activation_examples) >= min_examples]
        pairs = [(c, _bold_density(c, ci_threshold)) for c in eligible]
        pairs.sort(key=lambda p: p[1])
        self._components = [c for c, _ in pairs]
        self._densities = [d for _, d in pairs]
        self._key_to_idx = {c.component_key: i for i, c in enumerate(self._components)}

    def sample_similar(
        self,
        target: ComponentData,
        rng: random.Random,
        tolerance: float,
    ) -> ComponentData:
        """Sample a different component with similar bold density."""
        assert target.component_key in self._key_to_idx
        target_density = self._densities[self._key_to_idx[target.component_key]]

        lo = bisect.bisect_left(self._densities, target_density - tolerance)
        hi = bisect.bisect_right(self._densities, target_density + tolerance)

        candidates = [
            self._components[i]
            for i in range(lo, hi)
            if self._components[i].component_key != target.component_key
        ]

        # Widen search if no candidates in tolerance band
        if not candidates:
            candidates = [c for c in self._components if c.component_key != target.component_key]

        return rng.choice(candidates)


def _format_example(
    example: ActivationExample,
    app_tok: AppTokenizer,
    ci_threshold: float,
) -> str:
    valid = [
        (tid, ci) for tid, ci in zip(example.token_ids, example.ci_values, strict=True) if tid >= 0
    ]
    spans = app_tok.get_spans([tid for tid, _ in valid])
    tokens = [(span, ci > ci_threshold) for span, (_, ci) in zip(spans, valid, strict=True)]
    return delimit_tokens(tokens)


def _sample_intruder(
    target: ComponentData,
    density_index: DensityIndex,
    rng: random.Random,
    density_tolerance: float,
) -> ActivationExample:
    """Sample an intruder example from a component with similar bold density."""
    donor = density_index.sample_similar(target, rng, tolerance=density_tolerance)
    return rng.choice(donor.activation_examples)


def _build_prompt(
    real_examples: list[ActivationExample],
    intruder: ActivationExample,
    intruder_position: int,
    app_tok: AppTokenizer,
    ci_threshold: float,
) -> str:
    all_examples = list(real_examples)
    all_examples.insert(intruder_position, intruder)
    n_total = len(all_examples)
    n_real = len(real_examples)

    examples_text = ""
    for i, ex in enumerate(all_examples):
        examples_text += f"Example {i + 1}: {_format_example(ex, app_tok, ci_threshold)}\n\n"

    return f"""\
Below are {n_total} text snippets from a neural network's training data. {n_real} come from contexts \
where the SAME component fires strongly. One is an INTRUDER from a DIFFERENT component.

Tokens between <<delimiters>> are where the component fires most strongly.

{examples_text}\
Which example is the intruder? Identify what pattern the majority share, then pick \
the example that does not fit.

Respond with the intruder example number (1-{n_total}) and brief reasoning."""


async def score_component(
    llm: LLMClient,
    model: str,
    reasoning_effort: ReasoningEffort,
    component: ComponentData,
    density_index: DensityIndex,
    app_tok: AppTokenizer,
    ci_threshold: float,
    n_real: int,
    n_trials: int,
    density_tolerance: float,
) -> IntruderResult:
    assert len(component.activation_examples) >= n_real + 1

    rng = random.Random()
    trials: list[IntruderTrial] = []
    n_errors = 0

    for trial_idx in range(n_trials):
        real_examples = rng.sample(component.activation_examples, n_real)
        intruder = _sample_intruder(component, density_index, rng, density_tolerance)
        intruder_pos = rng.randint(0, n_real)
        correct_answer = intruder_pos + 1

        prompt = _build_prompt(real_examples, intruder, intruder_pos, app_tok, ci_threshold)

        try:
            response = await llm.chat(
                model=model,
                reasoning=Reasoning(effort=reasoning_effort),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                context_label=f"{component.component_key}/trial{trial_idx}",
                response_format=INTRUDER_RESPONSE_FORMAT,
            )
            parsed = json.loads(response)
            predicted = int(parsed["intruder"])
            trials.append(
                IntruderTrial(
                    correct_answer=correct_answer,
                    predicted=predicted,
                    is_correct=predicted == correct_answer,
                )
            )
        except Exception as e:
            logger.error(f"{component.component_key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    correct = sum(1 for t in trials if t.is_correct)
    score = correct / len(trials) if trials else 0.0

    return IntruderResult(
        component_key=component.component_key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


async def run_intruder_scoring(
    components: list[ComponentData],
    model: str,
    openrouter_api_key: str,
    tokenizer_name: str,
    db: HarvestDB,
    ci_threshold: float,
    eval_config: IntruderEvalConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[IntruderResult]:
    n_real = eval_config.n_real
    n_trials = eval_config.n_trials
    density_tolerance = eval_config.density_tolerance
    max_concurrent = eval_config.max_concurrent

    app_tok = AppTokenizer.from_pretrained(tokenizer_name)

    eligible = [c for c in components if len(c.activation_examples) >= n_real + 1]
    if limit is not None:
        eligible = eligible[:limit]

    density_index = DensityIndex(components, min_examples=n_real + 1, ci_threshold=ci_threshold)

    results: list[IntruderResult] = []

    existing_scores = db.get_scores("intruder")
    completed = set(existing_scores.keys())
    if completed:
        logger.info(f"Resuming: {len(completed)} already scored")

    remaining = [c for c in eligible if c.component_key not in completed]
    logger.info(f"Scoring {len(remaining)} components")

    semaphore = asyncio.Semaphore(max_concurrent)
    output_lock = asyncio.Lock()

    async def process_one(component: ComponentData, index: int, llm: LLMClient) -> None:
        async with semaphore:
            try:
                result = await score_component(
                    llm,
                    model,
                    eval_config.reasoning_effort,
                    component,
                    density_index,
                    app_tok,
                    ci_threshold,
                    n_real=n_real,
                    n_trials=n_trials,
                    density_tolerance=density_tolerance,
                )
            except BudgetExceededError:
                return
            except Exception as e:
                logger.error(f"Skipping {component.component_key}: {type(e).__name__}: {e}")
                return
            async with output_lock:
                results.append(result)
                details = json.dumps(asdict(result))
                db.save_score(result.component_key, "intruder", result.score, details)
                if index % 100 == 0:
                    logger.info(
                        f"[{index}] scored {len(results)}, ${llm.cost_tracker.cost_usd():.2f}"
                    )

    async with OpenRouter(api_key=openrouter_api_key) as api:
        input_price, output_price = await get_model_pricing(api, model)
        cost_tracker = CostTracker(
            input_price_per_token=input_price,
            output_price_per_token=output_price,
            limit_usd=cost_limit_usd,
        )
        llm = LLMClient(
            api=api,
            rate_limiter=AsyncLimiter(max_rate=5000, time_period=60),
            backoff=GlobalBackoff(),
            cost_tracker=cost_tracker,
        )

        await asyncio.gather(*[process_one(c, i, llm) for i, c in enumerate(remaining)])

        logger.info(f"Final cost: ${cost_tracker.cost_usd():.2f}")

    logger.info(f"Scored {len(results)} components")
    return results

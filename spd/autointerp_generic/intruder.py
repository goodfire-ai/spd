"""Intruder detection scoring (decomposition quality, label-free).

Tests whether a component's activating examples are coherent by asking an LLM
to identify an "intruder" example drawn from a different component.

Based on: "Evaluating SAE interpretability without explanations" (2025).
"""

import asyncio
import bisect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from aiolimiter import AsyncLimiter
from openrouter import OpenRouter
from openrouter.components import Effort, Reasoning

from spd.autointerp.llm_api import (
    BudgetExceededError,
    CostTracker,
    GlobalBackoff,
    LLMClient,
    get_model_pricing,
    make_response_format,
)
from spd.autointerp_generic.db import GenericInterpDB
from spd.autointerp_generic.prompt import format_example
from spd.autointerp_generic.types import (
    ComponentAutointerpData,
    DecompositionAutointerpData,
    IntruderConfig,
)
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


def _bold_density(component: ComponentAutointerpData) -> float:
    """Fraction of bold tokens across a component's activation examples."""
    n_bold = 0
    n_total = 0
    for ex in component.activating_examples:
        n_total += len(ex.bold)
        n_bold += sum(ex.bold)
    return n_bold / n_total if n_total > 0 else 0.0


class DensityIndex:
    """Index of components sorted by bold density for efficient similar-density lookup."""

    def __init__(self, components: list[ComponentAutointerpData], min_examples: int) -> None:
        eligible = [c for c in components if len(c.activating_examples) >= min_examples]
        pairs = [(c, _bold_density(c)) for c in eligible]
        pairs.sort(key=lambda p: p[1])
        self._components = [c for c, _ in pairs]
        self._densities = [d for _, d in pairs]
        self._key_to_idx = {c.key: i for i, c in enumerate(self._components)}

    def sample_similar(
        self,
        target: ComponentAutointerpData,
        rng: random.Random,
        tolerance: float,
    ) -> ComponentAutointerpData:
        """Sample a different component with similar bold density."""
        assert target.key in self._key_to_idx
        target_density = self._densities[self._key_to_idx[target.key]]

        lo = bisect.bisect_left(self._densities, target_density - tolerance)
        hi = bisect.bisect_right(self._densities, target_density + tolerance)

        candidates = [
            self._components[i] for i in range(lo, hi) if self._components[i].key != target.key
        ]

        if not candidates:
            candidates = [c for c in self._components if c.key != target.key]

        return rng.choice(candidates)


def _build_prompt(
    real_examples: list[str],
    intruder_text: str,
    intruder_position: int,
) -> str:
    all_texts = list(real_examples)
    all_texts.insert(intruder_position, intruder_text)
    n_total = len(all_texts)
    n_real = len(real_examples)

    examples_text = ""
    for i, text in enumerate(all_texts):
        examples_text += f"Example {i + 1}: {text}\n\n"

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
    reasoning_effort: Effort,
    component: ComponentAutointerpData,
    density_index: DensityIndex,
    data: DecompositionAutointerpData,
    n_real: int,
    n_trials: int,
    density_tolerance: float,
) -> IntruderResult:
    assert len(component.activating_examples) >= n_real + 1

    rng = random.Random()
    trials: list[IntruderTrial] = []
    n_errors = 0

    for trial_idx in range(n_trials):
        real_examples = rng.sample(component.activating_examples, n_real)
        donor = density_index.sample_similar(component, rng, tolerance=density_tolerance)
        intruder = rng.choice(donor.activating_examples)
        intruder_pos = rng.randint(0, n_real)
        correct_answer = intruder_pos + 1

        real_texts = [format_example(ex, data.tokenizer) for ex in real_examples]
        intruder_text = format_example(intruder, data.tokenizer)
        prompt = _build_prompt(real_texts, intruder_text, intruder_pos)

        try:
            response = await llm.chat(
                model=model,
                reasoning=Reasoning(effort=reasoning_effort),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                context_label=f"{component.key}/trial{trial_idx}",
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
            logger.error(f"{component.key}/trial{trial_idx}: {type(e).__name__}: {e}")
            n_errors += 1

    correct = sum(1 for t in trials if t.is_correct)
    score = correct / len(trials) if trials else 0.0

    return IntruderResult(
        component_key=component.key,
        score=score,
        trials=trials,
        n_errors=n_errors,
    )


def run_intruder_scoring(
    data: DecompositionAutointerpData,
    openrouter_api_key: str,
    db_path: Path,
    config: IntruderConfig,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[IntruderResult]:
    n_real = config.n_real
    min_examples = n_real + 1

    eligible = [c for c in data.components if len(c.activating_examples) >= min_examples]
    if limit is not None:
        eligible = eligible[:limit]

    density_index = DensityIndex(data.components, min_examples=min_examples)

    async def _run() -> list[IntruderResult]:
        db = GenericInterpDB(db_path)
        results: list[IntruderResult] = []

        existing_scores = db.get_scores("intruder")
        completed = set(existing_scores.keys())
        if completed:
            logger.info(f"Resuming: {len(completed)} already scored")

        remaining = [c for c in eligible if c.key not in completed]
        logger.info(f"Scoring {len(remaining)} components")

        semaphore = asyncio.Semaphore(config.max_concurrent)
        output_lock = asyncio.Lock()

        async def process_one(
            component: ComponentAutointerpData, index: int, llm: LLMClient
        ) -> None:
            async with semaphore:
                try:
                    result = await score_component(
                        llm,
                        config.model,
                        config.reasoning_effort,
                        component,
                        density_index,
                        data,
                        n_real=n_real,
                        n_trials=config.n_trials,
                        density_tolerance=config.density_tolerance,
                    )
                except BudgetExceededError:
                    return
                except Exception as e:
                    logger.error(f"Skipping {component.key}: {type(e).__name__}: {e}")
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

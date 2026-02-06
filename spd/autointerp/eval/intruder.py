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
from pathlib import Path

from openrouter import OpenRouter
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup, delimit_tokens
from spd.autointerp.llm_api import (
    CostTracker,
    RateLimiter,
    chat_with_retry,
    get_model_pricing,
    make_response_format,
)
from spd.harvest.schemas import ActivationExample, ComponentData
from spd.log import logger

N_REAL = 4
N_TRIALS = 10
MAX_TOKENS_PER_EXAMPLE = 64
CI_THRESHOLD = 0.3
DENSITY_TOLERANCE = 0.05
MAX_CONCURRENT_REQUESTS = 50
MAX_REQUESTS_PER_MINUTE = 200

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


def _bold_density(component: ComponentData) -> float:
    """Fraction of valid tokens above CI_THRESHOLD across a component's activation examples."""
    n_bold = 0
    n_total = 0
    for ex in component.activation_examples:
        for tid, ci in zip(ex.token_ids, ex.ci_values, strict=True):
            if tid < 0:
                continue
            n_total += 1
            if ci > CI_THRESHOLD:
                n_bold += 1
    return n_bold / n_total if n_total > 0 else 0.0


class DensityIndex:
    """Index of components sorted by bold density for efficient similar-density lookup."""

    def __init__(self, components: list[ComponentData], min_examples: int) -> None:
        eligible = [c for c in components if len(c.activation_examples) >= min_examples]
        pairs = [(c, _bold_density(c)) for c in eligible]
        pairs.sort(key=lambda p: p[1])
        self._components = [c for c, _ in pairs]
        self._densities = [d for _, d in pairs]
        self._key_to_idx = {c.component_key: i for i, c in enumerate(self._components)}

    def sample_similar(
        self,
        target: ComponentData,
        rng: random.Random,
        tolerance: float = DENSITY_TOLERANCE,
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
    lookup: dict[int, str],
    ci_threshold: float = CI_THRESHOLD,
) -> str:
    tokens = [
        (lookup[tid], ci > ci_threshold)
        for tid, ci in zip(example.token_ids, example.ci_values, strict=True)
        if tid >= 0
    ]
    return delimit_tokens(tokens[:MAX_TOKENS_PER_EXAMPLE])


def _sample_intruder(
    target: ComponentData,
    density_index: DensityIndex,
    rng: random.Random,
) -> ActivationExample:
    """Sample an intruder example from a component with similar bold density."""
    donor = density_index.sample_similar(target, rng)
    return rng.choice(donor.activation_examples)


def _build_prompt(
    real_examples: list[ActivationExample],
    intruder: ActivationExample,
    intruder_position: int,
    lookup: dict[int, str],
) -> str:
    all_examples = list(real_examples)
    all_examples.insert(intruder_position, intruder)

    examples_text = ""
    for i, ex in enumerate(all_examples):
        examples_text += f"Example {i + 1}: {_format_example(ex, lookup)}\n\n"

    return f"""\
You are evaluating the coherence of a neural network component's activations.

Below are 5 text examples. Four of them activate the SAME component in a neural network
(they share a common pattern). One is an INTRUDER — it activates a DIFFERENT component.

Tokens between <<delimiters>> are where the component is most active.

{examples_text}
Which example is the intruder? Think step by step about what pattern the majority share,
then identify which example does not fit.

Respond with the intruder example number (1-5) and brief reasoning."""


async def score_component(
    client: OpenRouter,
    model: str,
    component: ComponentData,
    density_index: DensityIndex,
    lookup: dict[int, str],
) -> IntruderResult:
    assert len(component.activation_examples) >= N_REAL + 1

    rng = random.Random(hash(component.component_key))
    trials: list[IntruderTrial] = []
    n_errors = 0

    for trial_idx in range(N_TRIALS):
        real_examples = rng.sample(component.activation_examples, N_REAL)
        intruder = _sample_intruder(component, density_index, rng)
        intruder_pos = rng.randint(0, N_REAL)
        correct_answer = intruder_pos + 1

        prompt = _build_prompt(real_examples, intruder, intruder_pos, lookup)

        try:
            response, _, _ = await chat_with_retry(
                client=client,
                model=model,
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
    output_path: Path,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> list[IntruderResult]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    lookup = build_token_lookup(tokenizer, tokenizer_name)

    results: list[IntruderResult] = []
    completed = set[str]()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                results.append(
                    IntruderResult(
                        component_key=data["component_key"],
                        score=data["score"],
                        trials=[IntruderTrial(**t) for t in data["trials"]],
                        n_errors=data["n_errors"],
                    )
                )
                completed.add(data["component_key"])
        print(f"Resuming: {len(completed)} already scored")

    eligible = [
        c
        for c in components
        if c.component_key not in completed and len(c.activation_examples) >= N_REAL + 1
    ]
    if limit is not None:
        eligible = eligible[:limit]
    print(f"Scoring {len(eligible)} components")

    density_index = DensityIndex(components, min_examples=N_REAL + 1)

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
                result = await score_component(client, model, component, density_index, lookup)
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

"""Build the in-memory `a x b` modular-arithmetic prompt grid and eval probe.

The grid is the `[a_range] x [b_range]` set of `"<a><symbol><b>="` prompts, tokenized
with the target's tokenizer — one prompt per row, all rows one shared token length,
row-major `(a, b)`. It is a pure function of the spec + the tokenizer, so every rank
builds the identical grid at startup with no coordination; unlike the streaming corpus
there is no packing and no offline artifact.

Two contracts layer on the same construction:

- `build_arithmetic_prompt_grid` — the shared-length invariant only. The tPD training
  pool (`targeted_data`) needs nothing more: it draws prompt rows and never scores an
  answer position.
- `build_arithmetic_probe` (the `ArithmeticCIGrid` eval metric) — additionally requires
  every answer to be a SINGLE token, so the `=` position's logits predict the answer and
  per-component CI / activation vectors reshape into `a x b` heatmaps.

Tokenizer geometry decides which ranges clear which assert: Llama-3.1 merges 1-3 digit
integers into one token, so `1..100` operands share a length and every sum 2..200 is a
single token; the Qwen tokenizers split numbers PER DIGIT, so grid operands must share a
digit count, and only single-digit answers clear the probe's answer premise. A range/op
that breaks an invariant fails fast rather than silently padding.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from param_decomp.experiments.lm.arithmetic_eval import ArithmeticGrid

# Operation -> (display symbol, result fn). Only addition is exercised today; subtraction
# (negative results) and multiplication tokenize differently and must re-clear the
# single-token-answer assert before use.
OPERATIONS = {
    "add": ("+", lambda a, b: a + b),
    "sub": ("-", lambda a, b: a - b),
    "mul": ("*", lambda a, b: a * b),
}


class PromptEncoder(Protocol):
    """The slice of a HF tokenizer the probe needs (`PreTrainedTokenizer.encode`)."""

    def encode(self, text: str, /, *, add_special_tokens: bool) -> Sequence[int] | np.ndarray: ...


@dataclass(frozen=True)
class ArithmeticPromptGrid:
    """`tokens[i * grid.n_b + j]` is the prompt for `a_values[i] <op> b_values[j] =`."""

    tokens: np.ndarray
    """`(n_prompts, T)` int32, row-major `(a, b)` order."""
    grid: ArithmeticGrid


@dataclass(frozen=True)
class ArithmeticProbe:
    """A prompt grid whose answers are all single tokens — the eval-probe contract."""

    tokens: np.ndarray
    """`(n_prompts, T)` int32, row-major `(a, b)` order."""
    grid: ArithmeticGrid
    answer_position: int
    """The `=` token; its logits predict the (single-token) answer."""


def build_arithmetic_prompt_grid(
    operation: str,
    a_range: tuple[int, int],
    b_range: tuple[int, int],
    tokenizer: PromptEncoder,
) -> ArithmeticPromptGrid:
    assert operation in OPERATIONS, f"operation must be one of {sorted(OPERATIONS)}"
    symbol, _ = OPERATIONS[operation]
    a_values = tuple(range(a_range[0], a_range[1] + 1))
    b_values = tuple(range(b_range[0], b_range[1] + 1))
    assert a_values and b_values, (a_range, b_range)

    rows: list[list[int]] = []
    seq_len: int | None = None
    for a in a_values:
        for b in b_values:
            prompt = f"{a}{symbol}{b}="
            ids = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=True)]
            if seq_len is None:
                seq_len = len(ids)
            assert len(ids) == seq_len, (
                f"prompt {prompt!r} tokenizes to {len(ids)} tokens but expected {seq_len}; "
                f"all prompts must share one length (padding is disabled) — pick an operand "
                f"range whose operands all tokenize to the same token count"
            )
            rows.append(ids)
    return ArithmeticPromptGrid(
        tokens=np.asarray(rows, dtype=np.int32),
        grid=ArithmeticGrid(a_values=a_values, b_values=b_values, symbol=symbol),
    )


def build_arithmetic_probe(
    operation: str,
    a_range: tuple[int, int],
    b_range: tuple[int, int],
    tokenizer: PromptEncoder,
) -> ArithmeticProbe:
    prompts = build_arithmetic_prompt_grid(operation, a_range, b_range, tokenizer)
    _, result_fn = OPERATIONS[operation]
    grid = prompts.grid
    for a in grid.a_values:
        for b in grid.b_values:
            answer = result_fn(a, b)
            answer_tokens = tokenizer.encode(str(answer), add_special_tokens=False)
            assert len(answer_tokens) == 1, (
                f"answer {answer} for '{a}{grid.symbol}{b}=' is {len(answer_tokens)} tokens, "
                f"not 1; the probe premise is a single answer token predicted at the `=` "
                f"position"
            )
    return ArithmeticProbe(
        tokens=prompts.tokens,
        grid=grid,
        answer_position=prompts.tokens.shape[1] - 1,
    )

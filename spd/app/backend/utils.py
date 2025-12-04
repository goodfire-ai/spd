"""Shared utilities for the SPD backend."""

import functools
import re
import traceback
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from spd.log import logger

# Expected format from frontend: entity/project/runId (8-char lowercase alphanumeric)
WANDB_PATH_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/([a-z0-9]{8})$")


def log_errors[T: Callable[..., Any]](func: T) -> T:
    """Decorator to log errors with full traceback for easier debugging."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions normally
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            raise

    return wrapper  # pyright: ignore[reportReturnType]


def validate_wandb_path(path: str) -> tuple[str, str, str]:
    """Validate that path is in expected entity/project/runId format.

    The frontend handles all format parsing and normalization.
    Backend just validates the expected normalized format.

    Returns (entity, project, run_id) tuple.
    """
    m = WANDB_PATH_RE.match(path.strip())
    if not m:
        raise ValueError(
            f'Invalid W&B path format. Expected "entity/project/runId" '
            f"(8-char lowercase alphanumeric run id). Got: {path}"
        )
    return m.groups()  # pyright: ignore[reportReturnType]


# Characters that don't get a space prefix in wordpiece
_PUNCT_NO_SPACE = set(".,!?;:'\")-]}>/")


def build_token_lookup(
    tokenizer: Any,
    tokenizer_name: str,
) -> dict[int, str]:
    """Build token ID -> string lookup.

    Uses tokenizer-specific strategy to produce strings that concatenate correctly.
    """
    lookup: dict[int, str] = {}
    vocab_size: int = tokenizer.vocab_size

    for tid in range(vocab_size):
        decoded: str = tokenizer.decode([tid], skip_special_tokens=False)

        match tokenizer_name:
            case "SimpleStories/test-SimpleStories-gpt2-1.25M":
                # WordPiece handling:
                if decoded.startswith("##"):
                    lookup[tid] = decoded[2:]
                elif decoded and decoded[0] in _PUNCT_NO_SPACE:
                    lookup[tid] = decoded
                else:
                    lookup[tid] = " " + decoded
            case "openai-community/gpt2":
                # BPE (GPT-2 style): spaces encoded in token via Ġ -> space
                lookup[tid] = decoded
            case _:
                raise ValueError(f"Unsupported tokenizer name: {tokenizer_name}")

    return lookup

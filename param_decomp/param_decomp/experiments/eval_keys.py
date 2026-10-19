"""Eval randomness: a fixed namespace of the run's key, independent of the configured
training length. Training draws `fold_in(key, step)`; eval keys fold a stream and a batch
index under `eval_key(key)`, a level below any training key."""

from enum import IntEnum

import numpy as np
from jax import random
from jaxtyping import PRNGKeyArray

_EVAL_NAMESPACE = np.uint32(2**32 - 1)


def eval_key(key: PRNGKeyArray) -> PRNGKeyArray:
    """The eval namespace of `key`, the same for every run length."""
    return random.fold_in(key, _EVAL_NAMESPACE)


class EvalKeyStream(IntEnum):
    """Named, disjoint language-model eval streams; append rather than renumber."""

    SCALARS = 1
    ATTENTION_PATTERNS = 2
    ARITHMETIC = 4
    WELL_TEMPEREDNESS = 5
    ROUTER_DIVERGENCE = 6


def eval_stream_key(run_key: PRNGKeyArray, stream: EvalKeyStream) -> PRNGKeyArray:
    """The root key of `stream`, the same at every eval pass of the run."""
    return random.fold_in(eval_key(run_key), stream)

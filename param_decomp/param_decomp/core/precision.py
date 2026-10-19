"""Shared floating-point policy for decomposition compute."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

COMPUTE_DT = jnp.bfloat16
"""Forward-compute dtype; trainable parameters remain fp32 masters."""


def cast_floating(tree: Any, dtype: Any) -> Any:
    """Cast every inexact array leaf while preserving static and discrete leaves."""
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)

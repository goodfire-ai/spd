from __future__ import annotations

from dataclasses import dataclass
import random
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from spd.clustering.merge_matrix import GroupMerge
from spd.spd_types import Probability

def format_scientific_latex(value: float) -> str:
	"""Format a number in LaTeX scientific notation style."""
	if value == 0:
		return r"$0$"
	
	import math
	exponent: int = int(math.floor(math.log10(abs(value))))
	mantissa: float = value / (10 ** exponent)
	
	return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
from collections.abc import Callable
from typing import Any


def named_lambda[T_callable: Callable[[Any], Any]](name: str, fn: T_callable) -> T_callable:
    """Helper to create a named lambda function for the sweep."""
    fn.__name__ = name
    return fn


def format_scientific_latex(value: float) -> str:
    """Format a number in LaTeX scientific notation style."""
    if value == 0:
        return r"$0$"

    import math

    exponent: int = int(math.floor(math.log10(abs(value))))
    mantissa: float = value / (10**exponent)

    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"

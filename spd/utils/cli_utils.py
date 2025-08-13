from collections.abc import Callable
from typing import Any


def format_function_docstring[T_callable: Callable[..., Any]](
    mapping: dict[str, Any],
    /,
) -> Callable[[T_callable], T_callable]:
    """Decorator to format function docstring with the given keyword arguments"""

    # I think we don't need to use functools.wraps here, since we return the same function
    def decorator(func: T_callable) -> T_callable:
        assert func.__doc__ is not None, "Function must have a docstring to format."
        func.__doc__ = func.__doc__.format_map(mapping)
        return func

    return decorator

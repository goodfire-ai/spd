"""`eqx.filter_jit` + `compiler_options` passthrough.

`equinox.filter_jit` forwards `**jitkwargs` to `jax.jit` at runtime, but its typed
`@overload`s expose only `fun` + `donate` — so passing `compiler_options` (the native,
in-process way to set XLA compiler flags, and the way they enter the compile-cache key)
fails basedpyright with "no overloads match". This wrapper centralizes that one cast so
call sites stay clean and typed.
"""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax

DonateMode = Literal["all", "all-except-first", "warn", "warn-except-first", "none"]


def filter_jit[**P, T](
    fn: Callable[P, T],
    *,
    donate: DonateMode = "none",
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[P, T]:
    """`eqx.filter_jit(fn, donate=…, compiler_options=…)` — `compiler_options` is forwarded
    to `jax.jit` (XLA compiler flags, native + in the compile-cache key). `None` or an
    empty dict means no options. CPU backends accept and ignore GPU flags."""
    return eqx.filter_jit(fn, donate=donate, compiler_options=compiler_options or {})  # pyright: ignore[reportCallIssue]


def aot_compile[**P, T](
    jitted: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> jax.stages.Compiled:
    """Compile one call signature of a `filter_jit`ted fn ahead of time and hand back the
    underlying `jax.stages.Compiled` (for its post-optimization cost analysis). jax shares
    the executable with the normal dispatch path, so the later first call does not
    recompile. Lowering only traces: donation-marked args are not consumed."""
    lowered = jitted.lower(*args, **kwargs)  # pyright: ignore[reportFunctionMemberAccess]
    return lowered.compile().compiled

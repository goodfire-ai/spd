import re
from re import Pattern
from typing import Any


def grep_repr(
    obj: Any,
    pattern: str | Pattern[str],
    *,
    char_context: int | None = 20,
    line_context: int | None = None,
    flags: int = 0,
    highlight: bool = True,
) -> None:
    """grep-like search on ``repr(obj)`` with optional ANSI highlighting.

    Converts *obj* to its ``repr`` string, finds every *pattern* match, and
    prints each match surrounded by configurable character or line context.

    # Parameters:
     - `obj : Any`
        object to search; its ``repr`` string is scanned
     - `pattern : str | Pattern[str]`
        regular-expression pattern (string or pre-compiled)
     - `char_context : int | None`
        characters of context before **and** after each match
        (ignored when `line_context` is given; defaults to `20`)
     - `line_context : int | None`
        lines of context before **and** after each match; overrides
        `char_context` when not ``None`` (defaults to ``None``)
     - `flags : int`
        flags forwarded to ``re.compile`` (defaults to ``0``)
     - `highlight : bool`
        if ``True`` wrap matches with ``ESC[1;31m … ESC[0m``
        (defaults to ``True``)

    # Returns:
     - `None`
        nothing; the function prints to ``stdout``

    # Modifies:
     - None

    # Usage:

    ```python
    >>> grep_repr([1, 2, 42, 3], r"42")
    [1, 2, ... 42 ... 3]               # 42 is colored bright-red
    ```

    # Raises:
     - `re.error` : invalid regular expression
    """
    text: str = repr(obj)
    regex: Pattern[str] = re.compile(pattern, flags) if isinstance(pattern, str) else pattern

    def _color(segment: str) -> str:
        return (
            regex.sub(lambda m: f"\033[1;31m{m.group(0)}\033[0m", segment) if highlight else segment
        )

    if line_context is not None:
        lines: list[str] = text.splitlines()
        start_idx: list[int] = []
        pos: int = 0
        for ln in lines:
            start_idx.append(pos)
            pos += len(ln) + 1  # account for removed newline

        for m in regex.finditer(text):
            first_char: int = m.start()
            line_no: int = max(i for i, s in enumerate(start_idx) if s <= first_char)
            lo: int = max(0, line_no - line_context)
            hi: int = min(len(lines), line_no + line_context + 1)
            snippet: str = "\n".join(lines[lo:hi])
            print(_color(snippet))
    else:
        ctx: int = 0 if char_context is None else char_context
        for m in regex.finditer(text):
            start: int = max(0, m.start() - ctx)
            end: int = min(len(text), m.end() + ctx)
            snippet: str = text[start:end]
            print(_color(snippet))

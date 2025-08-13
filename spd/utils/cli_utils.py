import argparse
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, override


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


# Default token sets (lowercase). You can override per-option.
TRUE_SET_DEFAULT: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
FALSE_SET_DEFAULT: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}


def _normalize_set(tokens: Iterable[str] | None, fallback: set[str]) -> set[str]:
    """Normalize a collection of tokens to a lowercase set, or return fallback."""
    if tokens is None:
        return set(fallback)
    return {str(t).lower() for t in tokens}


def parse_bool_token(
    token: str,
    true_set: set[str] | None = None,
    false_set: set[str] | None = None,
) -> bool:
    """Strict string-to-bool converter for argparse and friends.

    # Parameters:
     - `token : str`
        input token
     - `true_set : set[str] | None`
        accepted truthy strings (case-insensitive).
        Defaults to TRUE_SET_DEFAULT when None.
     - `false_set : set[str] | None`
        accepted falsy strings (case-insensitive).
        Defaults to FALSE_SET_DEFAULT when None.

    # Returns:
     - `bool`
        parsed boolean

    # Raises:
     - `argparse.ArgumentTypeError` : if not a recognized boolean string
    """
    ts: set[str] = _normalize_set(true_set, TRUE_SET_DEFAULT)
    fs: set[str] = _normalize_set(false_set, FALSE_SET_DEFAULT)
    v: str = token.lower()
    if v in ts:
        return True
    if v in fs:
        return False
    valid: list[str] = sorted(ts | fs)
    raise argparse.ArgumentTypeError(f"expected one of {valid}")


class BoolFlagOrValue(argparse.Action):
    """summary

    Configurable boolean action supporting any combination of:
      --flag                 -> True  (if allow_bare)
      --no-flag              -> False (if allow_no and --no-flag is registered)
      --flag true|false      -> parsed via custom sets
      --flag=true|false      -> parsed via custom sets

    Notes:
      - The --no-flag form never accepts a value. It forces False.
      - If allow_no is False but you still register a --no-flag alias,
        using it will produce a usage error.
      - Do not pass type= to this action.

    # Parameters:
     - `option_strings : list[str]`
        provided by argparse
     - `dest : str`
        attribute name on the namespace
     - `nargs : int | str | None`
        must be '?' for optional value
     - `true_set : set[str] | None`
        accepted truthy strings (case-insensitive). Defaults provided.
     - `false_set : set[str] | None`
        accepted falsy strings (case-insensitive). Defaults provided.
     - `allow_no : bool`
        whether the --no-flag form is allowed (defaults to True)
     - `allow_bare : bool`
        whether bare --flag (no value) is allowed (defaults to True)
     - `**kwargs`
        forwarded to base class

    # Raises:
     - `ValueError` : if nargs is not '?' or if type= is provided
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = None,
        **kwargs: bool | set[str] | None,
    ) -> None:
        # Extract custom kwargs before calling super().__init__
        true_set_opt: set[str] | None = kwargs.pop("true_set", None)  # pyright: ignore[reportAssignmentType]
        false_set_opt: set[str] | None = kwargs.pop("false_set", None)  # pyright: ignore[reportAssignmentType]
        allow_no_opt: bool = bool(kwargs.pop("allow_no", True))
        allow_bare_opt: bool = bool(kwargs.pop("allow_bare", True))

        if "type" in kwargs and kwargs["type"] is not None:
            raise ValueError("BoolFlagOrValue does not accept type=. Remove it.")

        if nargs not in (None, "?"):
            raise ValueError("BoolFlagOrValue requires nargs='?'")

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        # Store normalized config
        self.true_set: set[str] = _normalize_set(true_set_opt, TRUE_SET_DEFAULT)
        self.false_set: set[str] = _normalize_set(false_set_opt, FALSE_SET_DEFAULT)
        self.allow_no: bool = allow_no_opt
        self.allow_bare: bool = allow_bare_opt

    def _parse_token(self, token: str) -> bool:
        """Parse a boolean token using this action's configured sets."""
        return parse_bool_token(token, self.true_set, self.false_set)

    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        # Negated form handling
        if option_string is not None and option_string.startswith("--no-"):
            if not self.allow_no:
                parser.error(f"{option_string} is not allowed for this option")
                return  # pyright: ignore[reportUnreachable]
            if values is not None:
                dest_flag: str = self.dest.replace("_", "-")
                parser.error(f"{option_string} does not take a value; use --{dest_flag} true|false")
                return  # pyright: ignore[reportUnreachable]
            setattr(namespace, self.dest, False)
            return

        # Bare positive flag -> True (if allowed)
        if values is None:
            if not self.allow_bare:
                valid: list[str] = sorted(self.true_set | self.false_set)
                parser.error(f"option {option_string} requires a value; expected one of {valid}")
                return  # pyright: ignore[reportUnreachable]
            setattr(namespace, self.dest, True)
            return

        # we take only one value
        if not isinstance(values, str):
            if len(values) != 1:
                parser.error(
                    f"{option_string} expects a single value, got {len(values) = }, {values = }"
                )
                return  # pyright: ignore[reportUnreachable]
            values = values[0]  # type: ignore[assignment]

        # Positive flag with explicit value -> parse
        try:
            val: bool = self._parse_token(values)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
            return  # pyright: ignore[reportUnreachable]
        setattr(namespace, self.dest, val)


def add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool = False,
    help: str = "",
    true_set: set[str] | None = None,
    false_set: set[str] | None = None,
    allow_no: bool = False,
    allow_bare: bool = True,
) -> None:
    """summary

    Add a configurable boolean option that supports (depending on options):
      --<name>                  (bare positive, if allow_bare)
      --no-<name>               (negated, if allow_no)
      --<name> true|false
      --<name>=true|false

    # Parameters:
     - `parser : argparse.ArgumentParser`
        parser to modify
     - `name : str`
        base long option name (without leading dashes)
     - `default : bool`
        default value (defaults to False)
     - `help : str`
        help text (optional)
     - `true_set : set[str] | None`
        accepted truthy strings (case-insensitive). Defaults used when None.
     - `false_set : set[str] | None`
        accepted falsy strings (case-insensitive). Defaults used when None.
     - `allow_no : bool`
        whether to register/allow the --no-<name> alias (defaults to True)
     - `allow_bare : bool`
        whether bare --<name> implies True (defaults to True)

    # Returns:
     - `None`
        nothing; parser is modified

    # Modifies:
     - `parser` : adds a new argument with dest `<name>` (hyphens -> underscores)

    # Usage:
    ```python
    p = argparse.ArgumentParser()
    add_bool_flag(p, "feature", default=False, help="enable/disable feature")
    ns = p.parse_args(["--feature=false"])
    assert ns.feature is False
    ```
    """
    long_opt: str = f"--{name}"
    dest: str = name.replace("-", "_")
    option_strings: list[str] = [long_opt]
    if allow_no:
        option_strings.append(f"--no-{name}")

    tokens_preview: str = "{true,false}"
    readable_name: str = name.replace("-", " ")
    arg_help: str = help or (f"enable/disable {readable_name}; also accepts explicit true|false")

    parser.add_argument(
        *option_strings,
        dest=dest,
        action=BoolFlagOrValue,
        nargs="?",
        default=default,
        metavar=tokens_preview,
        help=arg_help,
        true_set=true_set,
        false_set=false_set,
        allow_no=allow_no,
        allow_bare=allow_bare,
    )

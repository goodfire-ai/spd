from __future__ import annotations

import argparse
from typing import Callable, Sequence, Tuple

from spd.utils.cli_utils import BoolFlagOrValue, add_bool_flag, parse_bool_token

def _build_parser_feature(
    *,
    default: bool = False,
    allow_no: bool = True,
    allow_bare: bool = True,
    true_set: set[str] | None = None,
    false_set: set[str] | None = None,
) -> argparse.ArgumentParser:
    p: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    add_bool_flag(
        p,
        "feature",
        default=default,
        allow_no=allow_no,
        allow_bare=allow_bare,
        true_set=true_set,
        false_set=false_set,
        help="test feature",
    )
    return p


def _must_parse(p: argparse.ArgumentParser, argv: Sequence[str]) -> bool:
    ns: argparse.Namespace = p.parse_args(list(argv))
    v: bool = getattr(ns, "feature")
    return v


def _must_exit2(p: argparse.ArgumentParser, argv: Sequence[str]) -> None:
    try:
        _ = p.parse_args(list(argv))
    except SystemExit as e:
        assert e.code == 2, f"expected exit code 2, got {e.code} for argv={argv}"
        return
    raise AssertionError(f"expected SystemExit for argv={argv}")


def test_parse_bool_token_defaults() -> None:
    assert parse_bool_token("true") is True
    assert parse_bool_token("FALSE") is False
    assert parse_bool_token("1") is True
    assert parse_bool_token("0") is False


def test_parse_bool_token_custom_sets() -> None:
    ts: set[str] = {"enable", "go"}
    fs: set[str] = {"disable", "stop"}
    assert parse_bool_token("enable", ts, fs) is True
    assert parse_bool_token("STOP", ts, fs) is False
    try:
        _ = parse_bool_token("true", ts, fs)
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("expected argparse.ArgumentTypeError for 'true' with custom sets")


def test_default_config_allows_bare_and_no_and_values() -> None:
    p: argparse.ArgumentParser = _build_parser_feature(default=False)
    assert _must_parse(p, []) is False  # default
    assert _must_parse(p, ["--feature"]) is True  # bare
    assert _must_parse(p, ["--no-feature"]) is False  # negated
    assert _must_parse(p, ["--feature", "true"]) is True  # space value
    assert _must_parse(p, ["--feature=false"]) is False  # equals value
    assert _must_parse(p, ["--feature=ON"]) is True  # case-insensitive


def test_last_one_wins() -> None:
    p: argparse.ArgumentParser = _build_parser_feature(default=False)
    assert _must_parse(p, ["--feature", "false", "--feature", "true"]) is True
    assert _must_parse(p, ["--feature", "true", "--no-feature"]) is False
    assert _must_parse(p, ["--no-feature", "--feature=false"]) is False


def test_disallow_no_form_by_not_registering() -> None:
    p: argparse.ArgumentParser = _build_parser_feature(allow_no=False)
    _must_exit2(p, ["--no-feature"])  # unrecognized argument, exit 2


def test_disallow_bare_form_requires_value() -> None:
    p: argparse.ArgumentParser = _build_parser_feature(allow_bare=False)
    _must_exit2(p, ["--feature"])  # missing value
    assert _must_parse(p, ["--feature=false"]) is False
    assert _must_parse(p, ["--feature", "true"]) is True


def test_custom_sets_on_action() -> None:
    p: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--feature",
        "--no-feature",
        dest="feature",
        action=BoolFlagOrValue,
        nargs="?",
        allow_no=False,
        allow_bare=True,
        true_set={"enable"},
        false_set={"disable"},
        default=False,
    )
    assert _must_parse(p, ["--feature", "enable"]) is True
    _must_exit2(p, ["--feature", "true"])        # not in custom sets
    _must_exit2(p, ["--no-feature"])             # negated disallowed at parse-time


def test_negated_never_takes_value() -> None:
    p: argparse.ArgumentParser = _build_parser_feature(default=True)
    _must_exit2(p, ["--no-feature=true"])
    _must_exit2(p, ["--no-feature", "false"])


def test_nargs_validation_on_add() -> None:
    p: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    try:
        p.add_argument("--feature", action=BoolFlagOrValue, nargs=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for nargs != '?'")


def test_reject_type_kwarg() -> None:
    p: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    try:
        p.add_argument("--feature", action=BoolFlagOrValue, type=str)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when type= is passed to action")


def test_multiple_independent_flags() -> None:
    p: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    add_bool_flag(p, "feature", default=False, allow_no=True, allow_bare=True)
    add_bool_flag(p, "debug", default=True, allow_no=False, allow_bare=False)
    ns: argparse.Namespace = p.parse_args(["--feature", "--debug=false"])
    assert ns.feature is True
    assert ns.debug is False

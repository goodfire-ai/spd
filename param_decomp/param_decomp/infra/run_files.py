"""Run identifiers."""

import secrets
from typing import Final, Literal

RunType = Literal["param_decomp", "train"]

RUN_TYPE_ABBREVIATIONS: Final[dict[RunType, str]] = {"param_decomp": "p", "train": "t"}


def generate_run_id(run_type: RunType) -> str:
    """Generate a unique run identifier.

    Format: `{type_abbr}-{random_hex}`
    """
    type_abbr = RUN_TYPE_ABBREVIATIONS[run_type]
    return f"{type_abbr}-{secrets.token_hex(4)}"

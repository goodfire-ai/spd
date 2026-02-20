"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""

import os
from collections.abc import Iterable
from netrc import netrc
from pathlib import Path
from urllib.parse import urlparse

import pytest
from pytest import Config, Item, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "requires_wandb: mark test as requiring WANDB credentials")


def _wandb_host() -> str:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    parsed = urlparse(base_url)
    host = parsed.netloc or parsed.path or "api.wandb.ai"
    return host.split("/")[0]


def _have_wandb_credentials() -> bool:
    """Check if we have WANDB credentials.

    We check for either of:
    - WANDB_API_KEY environment variable
    - .netrc file in the home directory
    """

    if os.environ.get("WANDB_API_KEY"):
        return True
    host = _wandb_host()
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        n = netrc(netrc_path.as_posix())
        return n.authenticators(host) is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config: Config, items: Iterable[Item]) -> None:
    runslow = config.getoption("--runslow")
    have_wandb = _have_wandb_credentials()
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_wandb = pytest.mark.skip(
        reason="No WANDB credentials (set WANDB_API_KEY or login via CLI)"
    )
    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "requires_wandb" in item.keywords and not have_wandb:
            item.add_marker(skip_wandb)

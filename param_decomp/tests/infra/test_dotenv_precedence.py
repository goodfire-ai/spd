"""`.env` must fill in unset variables, never clobber real ones (standard
python-dotenv precedence — the operator's shell/CI environment wins)."""

from pathlib import Path
from typing import Any

import pytest

from param_decomp.infra import wandb as wandb_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ENV = _REPO_ROOT / ".env"


def test_load_dotenv_keeps_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_wandb_entity` loads `.env` with override=False so exported variables win."""
    calls: list[dict[str, Any]] = []

    def spy_load_dotenv(*_args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(wandb_module, "load_dotenv", spy_load_dotenv)
    monkeypatch.setenv("WANDB_ENTITY", "from-environment")
    assert wandb_module.get_wandb_entity() == "from-environment"
    assert calls == [{"override": False}]


@pytest.mark.skipif(_REPO_ENV.exists(), reason="a developer .env in the repo root would interfere")
def test_dotenv_file_does_not_clobber_exported_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a stray .env with a different entity loses to the real env var."""
    _REPO_ENV.write_text("WANDB_ENTITY=from-dotenv-file\n")
    try:
        monkeypatch.setenv("WANDB_ENTITY", "from-environment")
        assert wandb_module.get_wandb_entity() == "from-environment"
    finally:
        _REPO_ENV.unlink()

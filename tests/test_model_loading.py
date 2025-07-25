import pytest

from spd.models.component_model import ComponentModel
from spd.registry import CANONICAL_RUNS

CANONICAL_RUNS_TUPLES: list[tuple[str, str]] = list(CANONICAL_RUNS.items())


@pytest.mark.parametrize(
    "run_id, wandb_url",
    CANONICAL_RUNS_TUPLES,
    ids=[
        f"{run_id}|{wandb_url.removeprefix('wandb:goodfire/spd/')}"
        for run_id, wandb_url in CANONICAL_RUNS_TUPLES
    ],
)
def test_load_canonical_runs(run_id: str, wandb_url: str) -> None:
    component_model, cfg, path = ComponentModel.from_pretrained(wandb_url)
    assert component_model is not None
    assert cfg is not None
    assert path.exists()
    assert path.is_file()
    assert component_model.run_id == run_id
    assert component_model.wandb_url == wandb_url

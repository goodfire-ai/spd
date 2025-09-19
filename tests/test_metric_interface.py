from typing import Mapping

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor

from spd.configs import Config
from spd.eval import Metric
from spd.models.component_model import ComponentModel


class _ToyMetric(Metric):
    SLOW = False

    def __init__(self, model: ComponentModel, config: Config) -> None:  # type: ignore[override]
        self.calls = []

    def watch_batch(  # type: ignore[override]
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        self.calls.append(("watch_batch", batch.shape, target_out.shape, list(ci.keys())))

    def compute(self) -> Mapping[str, float | Image.Image]:  # type: ignore[override]
        self.calls.append(("compute",))
        return {"toy/value": 1.0}


def test_metric_forward_calls_watch_then_compute(monkeypatch):
    dummy_model = object()  # not used
    dummy_config = object()  # not used
    metric = _ToyMetric(dummy_model, dummy_config)  # type: ignore[arg-type]

    batch = torch.zeros(2, 3, dtype=torch.long)
    target_out = torch.zeros(2, 3, 5)
    ci = {"layer": torch.zeros(2, 3, 7)}

    out = metric.forward(batch=batch, target_out=target_out, ci=ci)

    assert out == {"toy/value": 1.0}
    # Ensure call order
    assert metric.calls[0][0] == "watch_batch"
    assert metric.calls[1][0] == "compute"

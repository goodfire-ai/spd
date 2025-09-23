import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS
from spd.models.component_model import ComponentModel


def compute_total_loss(
    loss_metric_configs: list[MetricConfig],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    ci: dict[str, Float[Tensor, "batch C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    device: str,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict). terms_dict contains raw per-term values (no coeffs) and a weighted total.
    """
    total = torch.tensor(0.0, device=device)
    terms: dict[str, float] = {}

    for loss_metric_config in loss_metric_configs:
        loss_metric_cls: type[Metric] = METRICS[loss_metric_config.classname]
        loss_metric = loss_metric_cls(
            model, config, **loss_metric_config.extra_init_kwargs, sync_on_compute=False
        ).to(device)
        loss = loss_metric(
            batch=batch,
            target_out=target_out,
            ci=ci,
            ci_upper_leaky=ci_upper_leaky,
            weight_deltas=weight_deltas,
        )
        total = total + loss_metric_config.coeff * loss
        terms[loss_metric_config.classname] = loss.item()

    terms["total"] = total.item()

    return total, terms

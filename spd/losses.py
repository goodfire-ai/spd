import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics import Metric


def compute_total_loss(
    loss_metrics: list[Metric],
    loss_coeffs: dict[str, float],
    batch: Int[Tensor, "..."],
    ci: dict[str, Float[Tensor, "batch C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
    current_frac_of_training: float,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict). terms_dict contains raw per-term values (no coeffs) and a weighted total.
    """
    total = torch.tensor(0.0, device=batch.device)
    terms: dict[str, float] = {}

    for loss_metric in loss_metrics:
        loss = loss_metric(
            batch=batch,
            target_out=target_out,
            ci=ci,
            ci_upper_leaky=ci_upper_leaky,
            weight_deltas=weight_deltas,
            current_frac_of_training=current_frac_of_training,
        )
        classname = loss_metric.__class__.__name__
        total = total + loss_coeffs[classname] * loss
        terms[classname] = loss.item()

    terms["total"] = total.item()

    return total, terms

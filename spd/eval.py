"""Evaluation utilities using the new Metric classes."""

from collections.abc import Iterator, Mapping
from typing import Any

from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS, calc_weight_deltas
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def _should_run_metric(cfg: MetricConfig, cls: type, run_slow: bool) -> bool:
    is_slow = cfg.slow if cfg.slow is not None else bool(getattr(cls, "slow", False))
    return not (is_slow and not run_slow)


def evaluate(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    config: Config,
    run_slow: bool,
    n_steps: int,
) -> Mapping[str, float | Image.Image]:
    """Run evaluation and return a flat mapping of metric names to values/images.

    Returns keys without the "eval/" prefix. The caller is responsible for namespacing.
    """

    # Instantiate metrics (eval + loss) in a single pass
    eval_metrics: list[Any] = []
    combined_cfgs: list[MetricConfig] = [
        *config.eval_metric_configs,
        *config.loss_metric_configs,
    ]
    for cfg in combined_cfgs:
        metric_cls = METRICS[cfg.classname]
        if not _should_run_metric(cfg, metric_cls, run_slow):
            continue
        metric_obj = metric_cls(model, config, **cfg.extra_init_kwargs, sync_on_compute=False)
        metric_obj = metric_obj.to(device)
        eval_metrics.append(metric_obj)

    # Weight deltas can be computed once per eval since params are frozen
    weight_deltas = calc_weight_deltas(model, device=device) if config.use_delta_component else {}

    for _ in range(n_steps):
        batch_raw = next(eval_iterator)
        batch = extract_batch_data(batch_raw).to(device)

        # Get target outputs and causal importances
        target_out, pre_weight_acts = model(
            batch,
            mode="pre_forward_cache",
            module_names=list(model.components.keys()),
        )
        ci, ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=config.sigmoid_type,
            detach_inputs=False,
            sampling=config.sampling,
        )

        # Update eval metrics
        for metric in eval_metrics:
            metric.update(
                batch=batch,
                target_out=target_out,
                ci=ci,
                ci_upper_leaky=ci_upper_leaky,
                weight_deltas=weight_deltas,
            )

    # Finalize outputs
    outputs: dict[str, float | Image.Image] = {}

    # Compute metrics and aggregate outputs uniformly
    for metric in eval_metrics:
        computed = metric.compute()
        if hasattr(computed, "items"):
            for k, v in computed.items():
                outputs[k] = v
        else:
            # Scalar metric
            name = type(metric).__name__
            outputs[name] = (
                float(computed.item()) if hasattr(computed, "item") else float(computed)  # type: ignore[arg-type]
            )

    return outputs

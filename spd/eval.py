"""Evaluation utilities using the new Metric classes."""

from collections.abc import Iterator
from typing import Any

from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.types import Number
from torchmetrics import Metric
from wandb.plot.custom_chart import CustomChart

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import avg_metrics_across_ranks, is_distributed
from spd.utils.general_utils import extract_batch_data


def _should_run_metric(cfg: MetricConfig, cls: type, run_slow: bool) -> bool:
    is_slow = cfg.slow if cfg.slow is not None else bool(getattr(cls, "slow", False))
    return not (is_slow and not run_slow)


MetricOutType = dict[str, str | Number | Image.Image | CustomChart]
DistMetricOutType = dict[str, str | float | Image.Image | CustomChart]


def clean_metric_output(metric_name: str, computed_raw: Any) -> MetricOutType:
    """Clean metric output by converting tensors to floats/ints and ensuring the correct types.

    Expects outputs to be either a scalar tensor or a mapping of strings to scalars/images/tensors.
    """
    computed: MetricOutType = {}
    assert isinstance(computed_raw, dict | Tensor), f"{type(computed_raw)} not supported"
    if isinstance(computed_raw, Tensor):
        assert computed_raw.numel() == 1, (
            f"Only scalar tensors supported, got shape {computed_raw.shape}"
        )
        item = computed_raw.item()
        computed[metric_name] = item
    else:
        for k, v in computed_raw.items():
            assert isinstance(k, str), f"Only supports string keys, got {type(k)}"
            assert isinstance(v, str | Number | Image.Image | CustomChart | Tensor), (
                f"{type(v)} not supported"
            )
            if isinstance(v, Tensor):
                v = v.item()

            computed[k] = v
    return computed


def avg_eval_metrics_across_ranks(metrics: MetricOutType, device: str) -> DistMetricOutType:
    """Get the average of eval metrics across ranks.

    Ignores any metrics that are not numbers. Currently, the image metrics do not need to be
    averaged. If this changes for future metrics, we will need to do a reduce during calculcation
    of the metric.
    """
    assert is_distributed(), "Can only average metrics across ranks if running in distributed mode"
    metrics_keys_to_avg = {k: v for k, v in metrics.items() if isinstance(v, Number)}
    if metrics_keys_to_avg:
        avg_metrics = avg_metrics_across_ranks(metrics_keys_to_avg, device)
    else:
        avg_metrics = {}
    return {**metrics, **avg_metrics}


def evaluate(
    metric_configs: list[MetricConfig],
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    config: Config,
    run_slow: bool,
    n_eval_steps: int,
    current_frac_of_training: float,
) -> MetricOutType:
    """Run evaluation and return a flat mapping of metric names to values/images."""

    metrics: list[Metric] = []
    for cfg in metric_configs:
        metric_cls: type[Metric] = METRICS[cfg.classname]
        if not _should_run_metric(cfg, metric_cls, run_slow):
            continue
        metric = metric_cls(
            model=model, config=config, **cfg.extra_init_kwargs, sync_on_compute=True
        ).to(device)
        metrics.append(metric)

    # Weight deltas can be computed once per eval since params are frozen
    weight_deltas = model.calc_weight_deltas()

    for _ in range(n_eval_steps):
        batch_raw = next(eval_iterator)
        batch = extract_batch_data(batch_raw).to(device)

        target_out, pre_weight_acts = model(
            batch, mode="input_cache", module_names=list(model.components.keys())
        )
        ci, ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=config.sigmoid_type,
            detach_inputs=False,
            sampling=config.sampling,
        )

        for metric in metrics:
            metric.update(
                batch=batch,
                target_out=target_out,
                ci=ci,
                ci_upper_leaky=ci_upper_leaky,
                weight_deltas=weight_deltas,
                current_frac_of_training=current_frac_of_training,
            )

    outputs: MetricOutType = {}
    for metric in metrics:
        computed_raw: Any = metric.compute()
        computed = clean_metric_output(metric_name=type(metric).__name__, computed_raw=computed_raw)
        outputs.update(computed)

    return outputs

"""Evaluation utilities using the new Metric classes."""

from collections.abc import Iterator, Mapping

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def _should_run_metric(cfg: MetricConfig, cls: type, run_slow: bool) -> bool:
    is_slow = cfg.slow if cfg.slow is not None else bool(getattr(cls, "slow", False))
    return not (is_slow and not run_slow)


def clean_metric_outputs(
    metric_name: str,
    computed_raw: Mapping[str, int | float | Image.Image | Tensor] | Tensor,
) -> Mapping[str, int | float | Image.Image]:
    """Clean metric outputs by converting tensors to floats/ints and ensuring the correct types.

    Expects outputs to be either a scalar tensor or a mapping of strings to scalars/images/tensors.
    """
    computed: dict[str, int | float | Image.Image] = {}
    if isinstance(computed_raw, Tensor):
        # Convert tensor to float/int
        item = computed_raw.item()
        assert isinstance(item, float | int)
        computed[metric_name] = item
    else:
        assert isinstance(computed_raw, Mapping)
        for k, v in computed_raw.items():
            assert isinstance(k, str)
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, float | int | Image.Image)
            computed[k] = v
    return computed


def evaluate(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    config: Config,
    run_slow: bool,
    n_steps: int,
) -> Mapping[str, int | float | Image.Image]:
    """Run evaluation and return a flat mapping of metric names to values/images.

    Returns keys without the "eval/" prefix. The caller is responsible for namespacing.
    """

    eval_metrics: list[Metric] = []
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
    weight_deltas = model.calc_weight_deltas()

    for _ in range(n_steps):
        batch_raw = next(eval_iterator)
        batch = extract_batch_data(batch_raw).to(device)

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

        for metric in eval_metrics:
            metric.update(
                batch=batch,
                target_out=target_out,
                ci=ci,
                ci_upper_leaky=ci_upper_leaky,
                weight_deltas=weight_deltas,
            )

    outputs: dict[str, float | Image.Image] = {}

    for metric in eval_metrics:
        computed_raw = metric.compute()
        computed = clean_metric_outputs(type(metric).__name__, computed_raw)
        outputs.update(computed)

    return outputs

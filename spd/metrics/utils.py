from torchmetrics import Metric

from spd.configs import Config, MetricConfig
from spd.metrics import METRICS
from spd.models.component_model import ComponentModel


def create_metrics(
    metric_configs: list[MetricConfig],
    component_model: ComponentModel,
    config: Config,
    sync_on_compute: bool,
) -> list[Metric]:
    """Create metrics from config list.

    Args:
        metric_configs: List of metric configs.
        component_model: Component model.
        config: Config.
        sync_on_compute: Whether to synchronize metrics across processes when calling compute().
            This should be False when training with DDP (as DDP handles its own synchronization).

    Returns:
        List of metrics.
    """
    device = next(iter(component_model.parameters())).device
    metrics: list[Metric] = []
    for cfg in metric_configs:
        metric = METRICS[cfg.classname](
            model=component_model,
            config=config,
            **cfg.extra_init_kwargs,
            sync_on_compute=sync_on_compute,
        ).to(device)
        metrics.append(metric)
    return metrics

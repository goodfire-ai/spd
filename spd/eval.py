"""Evaluation utilities using the new Metric classes."""

from collections.abc import Iterator
from typing import Any

from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.types import Number
from wandb.plot.custom_chart import CustomChart

from spd.configs import (
    CEandKLLossesConfig,
    CI_L0Config,
    CIHistogramsConfig,
    CIMaskedReconLayerwiseLossTrainConfig,
    CIMaskedReconLossTrainConfig,
    CIMaskedReconSubsetLossTrainConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    Config,
    EvalMetricConfig,
    EvalMetricConfigType,
    FaithfulnessLossTrainConfig,
    IdentityCIErrorConfig,
    ImportanceMinimalityLossTrainConfig,
    PermutedCIPlotsConfig,
    StochasticReconLayerwiseLossTrainConfig,
    StochasticReconLossTrainConfig,
    StochasticReconSubsetCEAndKLConfig,
    StochasticReconSubsetLossTrainConfig,
    TrainMetricConfig,
    TrainMetricConfigType,
    UVPlotsConfig,
)
from spd.metrics.base import Metric
from spd.metrics.ce_and_kl_losses import CEandKLLosses
from spd.metrics.ci_histograms import CIHistograms
from spd.metrics.ci_l0 import CI_L0
from spd.metrics.ci_masked_recon_layerwise_loss import CIMaskedReconLayerwiseLoss
from spd.metrics.ci_masked_recon_loss import CIMaskedReconLoss
from spd.metrics.ci_masked_recon_subset_loss import CIMaskedReconSubsetLoss
from spd.metrics.ci_mean_per_component import CIMeanPerComponent
from spd.metrics.component_activation_density import ComponentActivationDensity
from spd.metrics.faithfulness_loss import FaithfulnessLoss
from spd.metrics.identity_ci_error import IdentityCIError
from spd.metrics.importance_minimality_loss import ImportanceMinimalityLoss
from spd.metrics.permuted_ci_plots import PermutedCIPlots
from spd.metrics.stochastic_recon_layerwise_loss import StochasticReconLayerwiseLoss
from spd.metrics.stochastic_recon_loss import StochasticReconLoss
from spd.metrics.stochastic_recon_subset_ce_and_kl import StochasticReconSubsetCEAndKL
from spd.metrics.stochastic_recon_subset_loss import StochasticReconSubsetLoss
from spd.metrics.uv_plots import UVPlots
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import avg_metrics_across_ranks, is_distributed
from spd.utils.general_utils import extract_batch_data


def _should_run_metric(cfg: EvalMetricConfig | TrainMetricConfig, slow_step: bool) -> bool:
    if not slow_step or isinstance(cfg, TrainMetricConfig):
        return True
    return cfg.slow


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


def init_metric(
    cfg: EvalMetricConfigType | TrainMetricConfigType,
    model: ComponentModel,
    run_config: Config,
    device: str,
) -> Metric:
    match cfg:
        case ImportanceMinimalityLossTrainConfig():
            metric = ImportanceMinimalityLoss(
                model=model,
                pnorm=cfg.pnorm,
                p_anneal_start_frac=cfg.p_anneal_start_frac,
                p_anneal_final_p=cfg.p_anneal_final_p,
                p_anneal_end_frac=cfg.p_anneal_end_frac,
            )
        case CEandKLLossesConfig():
            metric = CEandKLLosses(
                model=model, sampling=run_config.sampling, rounding_threshold=cfg.rounding_threshold
            )
        case CIHistogramsConfig():
            metric = CIHistograms(model=model, n_batches_accum=cfg.n_batches_accum)
        case CI_L0Config():
            metric = CI_L0(
                model=model, ci_alive_threshold=run_config.ci_alive_threshold, groups=cfg.groups
            )
        case CIMaskedReconSubsetLossTrainConfig():
            metric = CIMaskedReconSubsetLoss(
                model=model, output_loss_type=run_config.output_loss_type
            )
        case CIMaskedReconLayerwiseLossTrainConfig():
            metric = CIMaskedReconLayerwiseLoss(
                model=model, output_loss_type=run_config.output_loss_type
            )
        case CIMaskedReconLossTrainConfig():
            metric = CIMaskedReconLoss(model=model, output_loss_type=run_config.output_loss_type)
        case CIMeanPerComponentConfig():
            metric = CIMeanPerComponent(model=model)
        case ComponentActivationDensityConfig():
            metric = ComponentActivationDensity(
                model=model, ci_alive_threshold=run_config.ci_alive_threshold
            )
        case FaithfulnessLossTrainConfig():
            metric = FaithfulnessLoss(model=model)
        case IdentityCIErrorConfig():
            metric = IdentityCIError(
                model=model,
                sampling=run_config.sampling,
                sigmoid_type=run_config.sigmoid_type,
                identity_ci=cfg.identity_ci,
                dense_ci=cfg.dense_ci,
            )
        case PermutedCIPlotsConfig():
            metric = PermutedCIPlots(
                model=model,
                sampling=run_config.sampling,
                sigmoid_type=cfg.sigmoid_type,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
        case StochasticReconLayerwiseLossTrainConfig():
            metric = StochasticReconLayerwiseLoss(
                model=model,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
            )
        case StochasticReconLossTrainConfig():
            metric = StochasticReconLoss(
                model=model,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
            )
        case StochasticReconSubsetLossTrainConfig():
            metric = StochasticReconSubsetLoss(
                model=model,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
            )
        case StochasticReconSubsetCEAndKLConfig():
            metric = StochasticReconSubsetCEAndKL(
                model=model,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )
        case UVPlotsConfig():
            metric = UVPlots(
                model=model,
                sampling=run_config.sampling,
                sigmoid_type=run_config.sigmoid_type,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
    metric.to(device)
    return metric


def evaluate(
    metric_configs: list[EvalMetricConfigType | TrainMetricConfigType],
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    run_config: Config,
    slow_step: bool,
    n_eval_steps: int,
    current_frac_of_training: float,
) -> MetricOutType:
    """Run evaluation and return a mapping of metric names to values/images."""

    metrics: list[Metric] = []
    for cfg in metric_configs:
        metric = init_metric(cfg=cfg, model=model, run_config=run_config, device=device)
        metrics.append(metric)
        if not _should_run_metric(cfg=cfg, slow_step=slow_step):
            continue

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
            sigmoid_type=run_config.sigmoid_type,
            detach_inputs=False,
            sampling=run_config.sampling,
        )

        for metric in metrics:
            metric.update(
                batch=batch,
                target_out=target_out,
                ci=ci,
                current_frac_of_training=current_frac_of_training,
                ci_upper_leaky=ci_upper_leaky,
                weight_deltas=weight_deltas,
            )

    outputs: MetricOutType = {}
    for metric in metrics:
        # Combine metric states across all data-parallel processes
        metric.sync_dist()
        computed_raw: Any = metric.compute()
        computed = clean_metric_output(metric_name=type(metric).__name__, computed_raw=computed_raw)
        outputs.update(computed)

    return outputs

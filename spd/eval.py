"""Evaluation utilities using the new Metric classes."""

from collections import defaultdict
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
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    Config,
    FaithfulnessLossConfig,
    IdentityCIErrorConfig,
    ImportanceMinimalityLossConfig,
    MetricConfigType,
    PermutedCIPlotsConfig,
    PGDArbHiddenActsReconLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StochasticArbHiddenActsReconLossConfig,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetCEAndKLConfig,
    StochasticReconSubsetLossConfig,
    UVPlotsConfig,
)
from spd.metrics import PGDArbHiddenActsReconLoss, StochasticArbHiddenActsReconLoss
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
from spd.metrics.pgd_masked_recon_layerwise_loss import PGDReconLayerwiseLoss
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.metrics.pgd_masked_recon_subset_loss import PGDReconSubsetLoss
from spd.metrics.stochastic_hidden_acts_recon_loss import StochasticHiddenActsReconLoss
from spd.metrics.stochastic_recon_layerwise_loss import StochasticReconLayerwiseLoss
from spd.metrics.stochastic_recon_loss import StochasticReconLoss
from spd.metrics.stochastic_recon_subset_ce_and_kl import StochasticReconSubsetCEAndKL
from spd.metrics.stochastic_recon_subset_loss import StochasticReconSubsetLoss
from spd.metrics.uv_plots import UVPlots
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.utils.distributed_utils import avg_metrics_across_ranks, is_distributed
from spd.utils.general_utils import extract_batch_data

MetricOutType = dict[str, str | Number | Image.Image | CustomChart]
DistMetricOutType = dict[str, str | float | Image.Image | CustomChart]


def clean_metric_output(
    section: str,
    metric_name: str,
    computed_raw: Any,
) -> MetricOutType:
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
        computed[f"{section}/{metric_name}"] = item
    else:
        for k, v in computed_raw.items():
            assert isinstance(k, str), f"Only supports string keys, got {type(k)}"
            assert isinstance(v, str | Number | Image.Image | CustomChart | Tensor), (
                f"{type(v)} not supported"
            )
            if isinstance(v, Tensor):
                v = v.item()

            computed[f"{section}/{k}"] = v
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
    cfg: MetricConfigType,
    model: ComponentModel,
    run_config: Config,
    device: str,
) -> tuple[Metric, MetricConfigType]:
    match cfg:
        case ImportanceMinimalityLossConfig():
            metric = ImportanceMinimalityLoss(
                model=model,
                device=device,
                pnorm=cfg.pnorm,
                p_anneal_start_frac=cfg.p_anneal_start_frac,
                p_anneal_final_p=cfg.p_anneal_final_p,
                p_anneal_end_frac=cfg.p_anneal_end_frac,
            )
        case FaithfulnessLossConfig():
            metric = FaithfulnessLoss(
                model=model,
                device=device,
            )
        case CEandKLLossesConfig():
            metric = CEandKLLosses(
                model=model,
                device=device,
                sampling=run_config.sampling,
                rounding_threshold=cfg.rounding_threshold,
            )
        case CIHistogramsConfig():
            metric = CIHistograms(model=model, n_batches_accum=cfg.n_batches_accum)
        case CI_L0Config():
            metric = CI_L0(
                model=model,
                device=device,
                ci_alive_threshold=run_config.ci_alive_threshold,
                groups=cfg.groups,
            )
        case CIMaskedReconSubsetLossConfig():
            metric = CIMaskedReconSubsetLoss(
                model=model,
                device=device,
                output_loss_type=run_config.output_loss_type,
                routing=cfg.subset_routing_cfg,
            )
        case CIMaskedReconLayerwiseLossConfig():
            metric = CIMaskedReconLayerwiseLoss(
                model=model, device=device, output_loss_type=run_config.output_loss_type
            )
        case CIMaskedReconLossConfig():
            metric = CIMaskedReconLoss(
                model=model, device=device, output_loss_type=run_config.output_loss_type
            )
        case CIMeanPerComponentConfig():
            metric = CIMeanPerComponent(model=model, device=device)
        case ComponentActivationDensityConfig():
            metric = ComponentActivationDensity(
                model=model, device=device, ci_alive_threshold=run_config.ci_alive_threshold
            )
        case IdentityCIErrorConfig():
            metric = IdentityCIError(
                model=model,
                sampling=run_config.sampling,
                identity_ci=cfg.identity_ci,
                dense_ci=cfg.dense_ci,
            )
        case PermutedCIPlotsConfig():
            metric = PermutedCIPlots(
                model=model,
                sampling=run_config.sampling,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
        case StochasticReconLayerwiseLossConfig():
            metric = StochasticReconLayerwiseLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
            )
        case StochasticReconLossConfig():
            metric = StochasticReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
            )
        case StochasticReconSubsetLossConfig():
            metric = StochasticReconSubsetLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                output_loss_type=run_config.output_loss_type,
                routing=cfg.subset_routing_cfg,
            )
        case PGDReconLossConfig():
            metric = PGDReconLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                output_loss_type=run_config.output_loss_type,
                pgd_config=cfg,
            )
        case PGDReconSubsetLossConfig():
            metric = PGDReconSubsetLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                output_loss_type=run_config.output_loss_type,
                pgd_config=cfg,
                routing=cfg.subset_routing_cfg,
            )
        case PGDReconLayerwiseLossConfig():
            metric = PGDReconLayerwiseLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                output_loss_type=run_config.output_loss_type,
                pgd_config=cfg,
            )
        case StochasticReconSubsetCEAndKLConfig():
            metric = StochasticReconSubsetCEAndKL(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )
        case StochasticHiddenActsReconLossConfig():
            metric = StochasticHiddenActsReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
            )
        case UVPlotsConfig():
            metric = UVPlots(
                model=model,
                sampling=run_config.sampling,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
        case StochasticArbHiddenActsReconLossConfig():
            metric = StochasticArbHiddenActsReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                pre_target_module_patterns=cfg.pre_target_module_patterns,
                post_target_module_patterns=cfg.post_target_module_patterns,
            )
        case PGDArbHiddenActsReconLossConfig():
            metric = PGDArbHiddenActsReconLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                post_target_module_path=cfg.post_target_module_path,
                pgd_config=cfg,
            )
    return metric, cfg


def safe_update(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    """Update a dictionary with another dictionary, but only if the keys are not already present."""
    for k, v in d2.items():
        assert k not in d1
        d1[k] = v


def find_differing_config_keys(configs: list[MetricConfigType]) -> list[str]:
    """Find the differing config keys between a list of configs."""
    differing_keys: list[str] = []
    model_dumps = [config.model_dump() for config in configs]
    for key in model_dumps[0]:
        if not all(dump[key] == model_dumps[0][key] for dump in model_dumps[1:]):
            differing_keys.append(key)
    return differing_keys


def merge_with_qualification(
    outputs: list[tuple[Metric, MetricConfigType, MetricOutType]],
) -> MetricOutType:
    """Merge metric outputs with their qualifications."""
    outs_and_configs_by_metric_type: dict[
        type[Metric], list[tuple[MetricOutType, MetricConfigType]]
    ] = defaultdict(list)
    for metric, cfg, out in outputs:
        outs_and_configs_by_metric_type[type(metric)].append((out, cfg))

    merged: MetricOutType = {}
    for metric_outs_and_configs in outs_and_configs_by_metric_type.values():
        if len(metric_outs_and_configs) == 1:
            out, _cfg = metric_outs_and_configs[0]
            safe_update(merged, out)
            continue

        configs = [config for _, config in metric_outs_and_configs]
        differing_config_keys = find_differing_config_keys(configs)

        for out, cfg in metric_outs_and_configs:
            cfg_dict = cfg.model_dump()
            cfg_suffix = "_".join(f"{k}={cfg_dict[k]}" for k in differing_config_keys)
            new_items = {f"{k}_{cfg_suffix}": v for k, v in out.items()}
            safe_update(merged, new_items)

    return merged


def evaluate(
    eval_metric_configs: list[MetricConfigType],
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str,
    run_config: Config,
    slow_step: bool,
    n_eval_steps: int,
    current_frac_of_training: float,
) -> MetricOutType:
    """Run evaluation and return a mapping of metric names to values/images."""

    metrics_and_configs: list[tuple[Metric, MetricConfigType]] = []
    for cfg in eval_metric_configs:
        metric, cfg = init_metric(cfg=cfg, model=model, run_config=run_config, device=device)
        if metric.slow and not slow_step:
            continue
        metrics_and_configs.append((metric, cfg))

    # Weight deltas can be computed once per eval since params are frozen
    weight_deltas = model.calc_weight_deltas()

    for _ in range(n_eval_steps):
        batch_raw = next(eval_iterator)
        batch = extract_batch_data(batch_raw).to(device)

        target_output: OutputWithCache = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=run_config.sampling,
        )

        for metric, _ in metrics_and_configs:
            metric.update(
                batch=batch,
                target_out=target_output.output,
                pre_weight_acts=target_output.cache,
                ci=ci,
                current_frac_of_training=current_frac_of_training,
                weight_deltas=weight_deltas,
            )

    outputs: list[tuple[Metric, MetricConfigType, MetricOutType]] = []
    for metric, cfg in metrics_and_configs:
        computed_raw: Any = metric.compute()
        computed = clean_metric_output(
            section=metric.metric_section,
            metric_name=type(metric).__name__,
            computed_raw=computed_raw,
        )
        outputs.append((metric, cfg, computed))

    return merge_with_qualification(outputs)


# if __name__ == "__main__":
#     # quick test of merge_with_qualification
#     c1 = FaithfulnessLossConfig(coeff=1.0)
#     c2 = FaithfulnessLossConfig(coeff=2.0)

#     l1 = FaithfulnessLoss(model=None, device=None)
#     l2 = FaithfulnessLoss(model=None, device=None)

#     outputs: list[tuple[Metric, MetricConfigType, MetricOutType]] = [
#         (l1, c1, {"a": 1, "b": 2}),
#         (l2, c2, {"a": 3, "b": 4}),
#     ]

#     merged = merge_with_qualification(outputs)
#     print(merged)

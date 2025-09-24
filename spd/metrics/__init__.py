"""Metrics package with one file per metric.

This package exposes:
- Individual Metric classes (TorchMetrics-style)
- A `METRICS` registry mapping class names to classes

"""

from .ce_and_kl_losses import CEandKLLosses
from .ci_histograms import CIHistograms
from .ci_l0 import CI_L0
from .ci_mean_per_component import CIMeanPerComponent
from .ci_recon_layerwise_loss import CIReconLayerwiseLoss
from .ci_recon_loss import CIReconLoss
from .component_activation_density import ComponentActivationDensity
from .faithfulness_loss import FaithfulnessLoss
from .identity_ci_error import IdentityCIError
from .importance_minimality_loss import ImportanceMinimalityLoss
from .permuted_ci_plots import PermutedCIPlots
from .stochastic_recon_layerwise_loss import StochasticReconLayerwiseLoss
from .stochastic_recon_loss import StochasticReconLoss
from .subset_reconstruction_loss import SubsetReconstructionLoss
from .uv_plots import UVPlots

METRICS = {
    cls.__name__: cls
    for cls in [
        CI_L0,  # TODO: Check distributed behavior
        CEandKLLosses,  # TODO: Check distributed behavior
        CIHistograms,  # TODO: Check distributed behavior
        ComponentActivationDensity,  # TODO: Check distributed behavior
        PermutedCIPlots,
        UVPlots,
        IdentityCIError,
        CIMeanPerComponent,  # TODO: Check distributed behavior
        SubsetReconstructionLoss,  # TODO: Fix up module and check distributed
        FaithfulnessLoss,
        CIReconLoss,  # TODO: Rename all of these to CIMaskedRecon
        StochasticReconLoss,
        CIReconLayerwiseLoss,
        StochasticReconLayerwiseLoss,
        ImportanceMinimalityLoss,
    ]
}

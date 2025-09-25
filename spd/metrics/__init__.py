"""Metrics package with one file per metric.

This package exposes:
- Individual Metric classes (TorchMetrics-style)
- A `METRICS` registry mapping class names to classes

"""

# Note that "... as ..." allows for these to be imported elsewhere (See PEP 484 on re-exporting)
from .ce_and_kl_losses import CEandKLLosses as CEandKLLosses
from .ci_histograms import CIHistograms as CIHistograms
from .ci_l0 import CI_L0 as CI_L0
from .ci_mean_per_component import CIMeanPerComponent as CIMeanPerComponent
from .ci_recon_layerwise_loss import CIMaskedReconLayerwiseLoss as CIMaskedReconLayerwiseLoss
from .ci_recon_loss import CIMaskedReconLoss as CIMaskedReconLoss
from .component_activation_density import ComponentActivationDensity as ComponentActivationDensity
from .faithfulness_loss import FaithfulnessLoss as FaithfulnessLoss
from .faithfulness_loss import faithfulness_loss as faithfulness_loss
from .identity_ci_error import IdentityCIError as IdentityCIError
from .importance_minimality_loss import ImportanceMinimalityLoss as ImportanceMinimalityLoss
from .importance_minimality_loss import importance_minimality_loss as importance_minimality_loss
from .permuted_ci_plots import PermutedCIPlots as PermutedCIPlots
from .stochastic_recon_layerwise_loss import (
    StochasticReconLayerwiseLoss as StochasticReconLayerwiseLoss,
)
from .stochastic_recon_layerwise_loss import (
    stochastic_recon_layerwise_loss as stochastic_recon_layerwise_loss,
)
from .stochastic_recon_loss import StochasticReconLoss as StochasticReconLoss
from .stochastic_recon_loss import stochastic_recon_loss as stochastic_recon_loss
from .stochastic_recon_subset_loss import StochasticReconSubsetLoss as StochasticReconSubsetLoss
from .stochastic_recon_subset_loss import (
    stochastic_recon_subset_loss as stochastic_recon_subset_loss,
)
from .subset_reconstruction_loss import SubsetReconCEAndKL as SubsetReconCEAndKL
from .uv_plots import UVPlots as UVPlots

METRICS = {
    cls.__name__: cls
    for cls in [
        CI_L0,
        CEandKLLosses,
        CIHistograms,
        ComponentActivationDensity,
        PermutedCIPlots,
        UVPlots,
        IdentityCIError,
        CIMeanPerComponent,
        SubsetReconCEAndKL,
        FaithfulnessLoss,
        CIMaskedReconLoss,
        StochasticReconLoss,
        CIMaskedReconLayerwiseLoss,
        StochasticReconLayerwiseLoss,
        StochasticReconSubsetLoss,
        ImportanceMinimalityLoss,
    ]
}

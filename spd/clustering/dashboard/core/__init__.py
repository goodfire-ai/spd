"""Dashboard core data structures - modular package.

This package provides all data structures for the clustering dashboard.
All symbols are re-exported from this __init__ for backward compatibility.
"""

from spd.clustering.dashboard.core.activations_accumulator import GlobalActivationsAccumulator
from spd.clustering.dashboard.core.base import (
    ACTIVATION_SAMPLE_BATCH_STATS,
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterId,
    ClusterIdHash,
    ClusterLabel,
    ComponentActivationData,
    ComponentInfo,
    Direction,
    TextSample,
    TextSampleHash,
    TrackingCriterionHash,
)
from spd.clustering.dashboard.core.batch_storage import BatchProcessingStorage
from spd.clustering.dashboard.core.cluster_data import (
    BinnedData,
    ClusterData,
    TrackingCriterion,
)
from spd.clustering.dashboard.core.compute_helpers import (
    ClusterActivations,
    compute_all_cluster_activations,
    compute_cluster_coactivations,
    compute_component_coactivations_in_cluster,
    compute_component_cosine_similarities,
)
from spd.clustering.dashboard.core.dashboard_data import DashboardData

__all__ = [
    # Type aliases from base
    "TextSampleHash",
    "ActivationSampleHash",
    "ClusterIdHash",
    "TrackingCriterionHash",
    "ClusterLabel",
    "Direction",
    # Classes from base
    "ComponentInfo",
    "ComponentActivationData",
    "ClusterId",
    "TextSample",
    "ActivationSampleBatch",
    # Constants from base
    "ACTIVATION_SAMPLE_BATCH_STATS",
    # Classes from cluster_data
    "TrackingCriterion",
    "BinnedData",
    "ClusterData",
    # Classes from dashboard_data
    "DashboardData",
    # Classes from batch_storage
    "BatchProcessingStorage",
    # Classes from activations_accumulator
    "GlobalActivationsAccumulator",
    # Classes and functions from compute_helpers
    "ClusterActivations",
    "compute_all_cluster_activations",
    "compute_cluster_coactivations",
    "compute_component_coactivations_in_cluster",
    "compute_component_cosine_similarities",
]

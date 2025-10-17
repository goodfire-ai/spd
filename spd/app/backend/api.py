from typing import Any

from pydantic import BaseModel


class ActivationContext(BaseModel):
    raw_text: str
    offset_mapping: list[tuple[int, int]]
    token_ci_values: list[float]
    active_position: int
    ci_value: float


class TokenDensity(BaseModel):
    token: str
    density: float


class SubcomponentActivationContexts(BaseModel):
    subcomponent_idx: int
    examples: list[ActivationContext]
    token_densities: list[TokenDensity]


class ModelActivationContexts(BaseModel):
    layers: dict[str, list[SubcomponentActivationContexts]]


class SparseVector(BaseModel):
    l0: int
    indices: list[int]
    values: list[float]


class MatrixCausalImportances(BaseModel):
    subcomponent_cis_sparse: SparseVector
    subcomponent_cis: list[float]
    """the CI values for each subcomponent"""

    component_agg_cis: list[float]
    """len: M. For each component, the assigned CI value as aggregated from the subcomponents (usually via
    max)"""


class LayerCIs(BaseModel):
    module: str
    token_cis: list[MatrixCausalImportances]


class OutputTokenLogit(BaseModel):
    token: str
    logit: float
    probability: float


class RunResponse(BaseModel):
    prompt_id: str
    prompt_tokens: list[str]
    layer_cis: list[LayerCIs]
    full_run_token_logits: list[list[OutputTokenLogit]]
    ci_masked_token_logits: list[list[OutputTokenLogit]]


class TokenLayerCosineSimilarityData(BaseModel):
    input_singular_vectors: list[list[float]]
    output_singular_vectors: list[list[float]]
    component_indices: list[int]


class RunRequest(BaseModel):
    prompt: str


class SubcomponentAblationRequest(BaseModel):
    prompt_id: str
    subcomponent_mask: dict[str, list[list[int]]]


class SubcomponentAblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]


class ComponentAblationRequest(BaseModel):
    prompt_id: str
    component_mask: dict[str, list[list[int]]]


class TokenAblationStats(BaseModel):
    original_active_count: int
    """Number of components that were active (above threshold) in the original run"""
    ablated_count: int
    """Number of originally-active components that were ablated"""
    ablated_magnitude: float
    """Total CI magnitude lost due to ablation"""


class LayerAblationStats(BaseModel):
    module: str
    token_stats: list[TokenAblationStats]


class AblationStats(BaseModel):
    layer_stats: list[LayerAblationStats]


class InterventionResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]
    ablation_stats: AblationStats


class CombineMasksRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]  # List of token indices (positions) to combine
    description: str | None = None


class MaskOverrideDTO(BaseModel):
    id: str
    layer: str
    combined_mask: SparseVector
    description: str | None


class CombineMasksResponse(BaseModel):
    mask_id: str
    mask_override: MaskOverrideDTO


class SimulateMergeRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]


class SimulateMergeResponse(BaseModel):
    l0: int
    jacc: float


class ClusteringShape(BaseModel):
    module_component_assignments: dict[str, list[int]]
    """For each module, a length C list of indices mapping its subcomponents to a component"""
    module_component_groups: dict[str, list[list[int]]]
    """For each module, the groups of subcomponents that are assigned to a component (basically the
    inverse of module_component_assignments)"""


class ClusterRunDTO(BaseModel):
    wandb_path: str
    iteration: int
    clustering_shape: ClusteringShape


class TrainRunDTO(BaseModel):
    wandb_path: str
    component_layers: list[str]
    available_cluster_runs: list[str]
    config: dict[str, Any]


class Status(BaseModel):
    train_run: TrainRunDTO | None
    cluster_run: ClusterRunDTO | None


class AvailablePrompt(BaseModel):
    index: int
    full_text: str


class Run(BaseModel):
    id: str
    url: str


class ClusterIdDTO(BaseModel):
    clustering_run: str
    iteration: int
    cluster_label: int
    hash: str


class HistogramDTO(BaseModel):
    bin_edges: list[float]
    bin_counts: list[int]


class TokenActivationStatDTO(BaseModel):
    token: str
    count: int


class TokenActivationsDTO(BaseModel):
    top_tokens: list[TokenActivationStatDTO]
    total_unique_tokens: int
    total_activations: int
    entropy: float
    concentration_ratio: float
    activation_threshold: float


class ClusterComponentDTO(BaseModel):
    module: str
    index: int
    label: str


class ClusterStatsDTO(BaseModel):
    all_activations: HistogramDTO
    max_activation_position: HistogramDTO
    n_samples: int
    n_tokens: int
    mean_activation: float
    min_activation: float
    max_activation: float
    median_activation: float
    token_activations: TokenActivationsDTO


class ClusterDataDTO(BaseModel):
    cluster_hash: str
    components: list[ClusterComponentDTO]
    criterion_samples: dict[str, list[str]]
    stats: ClusterStatsDTO


class TextSampleDTO(BaseModel):
    text_hash: str
    full_text: str
    tokens: list[str]


class ActivationBatchDTO(BaseModel):
    cluster_id: ClusterIdDTO
    text_hashes: list[str]
    activations: list[list[float]]


class ClusterDashboardResponse(BaseModel):
    clusters: list[ClusterDataDTO]
    text_samples: list[TextSampleDTO]
    activation_batch: ActivationBatchDTO
    activations_map: dict[str, int]
    model_info: dict[str, Any]
    iteration: int
    run_path: str


class ApplyMaskRequest(BaseModel):
    prompt_id: str
    mask_override_id: str

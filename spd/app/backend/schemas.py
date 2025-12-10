from typing import Literal

from pydantic import BaseModel

# =============================================================================
# API Response Models
# =============================================================================


class OutputProbability(BaseModel):
    """Output probability for a specific token at a specific position."""

    prob: float
    token: str


class EdgeData(BaseModel):
    """Edge in the attribution graph."""

    src: str  # "layer:seq:cIdx"
    tgt: str  # "layer:seq:cIdx"
    val: float
    is_cross_seq: bool = False


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str


class GraphData(BaseModel):
    """Full attribution graph data."""

    id: int
    tokens: list[str]
    edges: list[EdgeData]
    outputProbs: dict[str, OutputProbability]
    nodeImportance: dict[str, float]  # node key -> sum of squared edge values
    maxAbsAttr: float  # max absolute edge value
    l0_total: int  # total active components at current CI threshold


class OptimizationResult(BaseModel):
    """Results from optimized CI computation."""

    label_token: int
    label_str: str
    imp_min_coeff: float
    ce_loss_coeff: float
    steps: int
    label_prob: float


class GraphDataWithOptimization(GraphData):
    """Attribution graph data with optimization results."""

    optimization: OptimizationResult


class ComponentStats(BaseModel):
    """Statistics for a component across prompts."""

    prompt_count: int
    avg_max_ci: float
    prompt_ids: list[int]


class PromptSearchQuery(BaseModel):
    """Query parameters for prompt search."""

    components: list[str]
    mode: str


class PromptSearchResponse(BaseModel):
    """Response from prompt search endpoint."""

    query: PromptSearchQuery
    count: int
    results: list[PromptPreview]


class TokenizeResponse(BaseModel):
    """Response from tokenize endpoint."""

    token_ids: list[int]
    tokens: list[str]
    text: str


class TokenInfo(BaseModel):
    """A single token from the tokenizer vocabulary."""

    id: int
    string: str


class TokensResponse(BaseModel):
    """Response containing all tokens in the vocabulary."""

    tokens: list[TokenInfo]


# SSE streaming message types
class ProgressMessage(BaseModel):
    """Progress update during streaming computation."""

    type: Literal["progress"]
    current: int
    total: int
    stage: str


class ErrorMessage(BaseModel):
    """Error message during streaming computation."""

    type: Literal["error"]
    error: str


class CompleteMessage(BaseModel):
    """Completion message with result data."""

    type: Literal["complete"]
    data: GraphData


class CompleteMessageWithOptimization(BaseModel):
    """Completion message with optimization result data."""

    type: Literal["complete"]
    data: GraphDataWithOptimization


# =============================================================================
# Configuration Models
# =============================================================================


class ActivationContextsGenerationConfig(BaseModel):
    """Configuration for generating activation contexts."""

    importance_threshold: float = 0.01
    n_batches: int = 100
    batch_size: int = 32
    n_tokens_either_side: int = 5
    topk_examples: int = 20
    separation_tokens: int = 0


class PromptsGenerationConfig(BaseModel):
    """Configuration for generating prompts to store."""

    n_prompts: int = 1000
    seq_length: int | None = None  # None = use model's max_seq_len


class SubcomponentActivationContexts(BaseModel):
    """Activation context data for a single subcomponent, using columnar layout for efficiency.

    Note: Token P/R/lift stats are now computed by the batch job and served via the
    /token_stats endpoint, not stored here.
    """

    subcomponent_idx: int
    mean_ci: float

    # Examples - columnar arrays (n_examples ~ topk, window_size ~ 2*n_tokens_either_side+1)
    example_tokens: list[list[str]]  # [n_examples][window_size]
    example_ci: list[list[float]]  # [n_examples][window_size]
    example_active_pos: list[int]  # [n_examples] - index into window
    example_active_ci: list[float]  # [n_examples]


class ModelActivationContexts(BaseModel):
    layers: dict[str, list[SubcomponentActivationContexts]]


class LoadedRun(BaseModel):
    """Info about the currently loaded run."""

    id: int
    wandb_path: str
    config_yaml: str
    has_activation_contexts: bool
    has_prompts: bool
    prompt_count: int
    context_length: int
    backend_user: str


class SubcomponentMetadata(BaseModel):
    """Lightweight metadata for a subcomponent (without examples/token_prs)"""

    subcomponent_idx: int
    mean_ci: float


class HarvestMetadata(BaseModel):
    """Lightweight metadata returned after harvest, containing only indices and mean_ci values"""

    layers: dict[str, list[SubcomponentMetadata]]


class TokenPRLiftPMI(BaseModel):
    """Token precision, recall, lift, and PMI lists."""

    top_recall: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_precision: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_lift: list[tuple[str, float]]  # [(token, lift), ...] sorted desc
    top_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest positive association
    bottom_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest negative association


class TokenStatsResponse(BaseModel):
    """Token stats for a component (from batch job).

    Contains both input token stats (what tokens activate this component)
    and output token stats (what tokens this component predicts).
    """

    input: TokenPRLiftPMI  # Stats for input tokens
    output: TokenPRLiftPMI  # Stats for output (predicted) tokens


# =============================================================================
# Intervention Models
# =============================================================================


class InterventionNode(BaseModel):
    """A specific node to activate during intervention."""

    layer: str
    seq_pos: int
    component_idx: int


class InterventionRequest(BaseModel):
    """Request for intervention forward pass."""

    text: str
    nodes: list[InterventionNode]
    top_k: int = 10


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str
    token_id: int
    prob: float
    logit: float


class InterventionResponse(BaseModel):
    """Response from intervention forward pass."""

    input_tokens: list[str]
    predictions_per_position: list[list[TokenPrediction]]


# =============================================================================
# Intervention Persistence Models
# =============================================================================


class RunInterventionRequest(BaseModel):
    """Request to run and save an intervention."""

    graph_id: int
    text: str
    selected_nodes: list[str]  # node keys (layer:seq:cIdx)
    top_k: int = 10


class InterventionRunSummary(BaseModel):
    """Summary of a saved intervention run."""

    id: int
    selected_nodes: list[str]
    result: InterventionResponse
    created_at: str


# =============================================================================
# Component Probe Models
# =============================================================================


class ComponentProbeRequest(BaseModel):
    """Request to probe a component's CI on custom text."""

    text: str
    layer: str
    component_idx: int


class ComponentProbeResponse(BaseModel):
    """Response with CI values for a component on custom text."""

    tokens: list[str]
    ci_values: list[float]


# =============================================================================
# Component Correlation Models
# =============================================================================


class CorrelatedComponent(BaseModel):
    """A component correlated with a query component."""

    component_key: str
    score: float


class ComponentCorrelationsResponse(BaseModel):
    """Correlation data for a component across different metrics."""

    precision: list[CorrelatedComponent]
    recall: list[CorrelatedComponent]
    f1: list[CorrelatedComponent]
    jaccard: list[CorrelatedComponent]
    pmi: list[CorrelatedComponent]


# =============================================================================
# Dataset Search Models
# =============================================================================


class DatasetSearchResult(BaseModel):
    """A single search result from the SimpleStories dataset."""

    story: str
    occurrence_count: int
    topic: str | None = None
    theme: str | None = None


class DatasetSearchMetadata(BaseModel):
    """Metadata about a completed dataset search."""

    query: str
    split: str
    total_results: int
    search_time_seconds: float


class DatasetSearchPage(BaseModel):
    """Paginated results from a dataset search."""

    results: list[DatasetSearchResult]
    page: int
    page_size: int
    total_results: int
    total_pages: int

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

    src: str  # Format: "layer:component_idx:position"
    tgt: str
    val: float


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


class OptimizationResult(BaseModel):
    """Results from optimized CI computation."""

    label_token: int
    label_str: str
    imp_min_coeff: float
    ce_loss_coeff: float
    steps: int
    label_prob: float
    l0_total: float
    l0_per_layer: dict[str, float]


class GraphDataWithOptimization(BaseModel):
    """Attribution graph data with optimization results."""

    id: int
    tokens: list[str]
    edges: list[EdgeData]
    outputProbs: dict[str, OutputProbability]
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
    """Activation context data for a single subcomponent, using columnar layout for efficiency."""

    subcomponent_idx: int
    mean_ci: float

    # Examples - columnar arrays (n_examples ~ topk, window_size ~ 2*n_tokens_either_side+1)
    example_tokens: list[list[str]]  # [n_examples][window_size]
    example_ci: list[list[float]]  # [n_examples][window_size]
    example_active_pos: list[int]  # [n_examples] - index into window
    example_active_ci: list[float]  # [n_examples]

    # Token precision/recall (input tokens) - columnar arrays sorted by recall descending
    pr_tokens: list[str]  # [n_unique_tokens]
    pr_recalls: list[float]  # [n_unique_tokens]
    pr_precisions: list[float]  # [n_unique_tokens]

    # Predicted token stats - P(predicted_token | component fires)
    # Sorted by probability descending
    predicted_tokens: list[str]  # [n_unique_predicted]
    predicted_probs: list[float]  # [n_unique_predicted] - P(token predicted | component fires)


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


class RunInfo(BaseModel):
    """Info about a run in the database."""

    id: int
    wandb_path: str
    prompt_count: int
    has_activation_contexts: bool


class SubcomponentMetadata(BaseModel):
    """Lightweight metadata for a subcomponent (without examples/token_prs)"""

    subcomponent_idx: int
    mean_ci: float


class HarvestMetadata(BaseModel):
    """Lightweight metadata returned after harvest, containing only indices and mean_ci values"""

    layers: dict[str, list[SubcomponentMetadata]]

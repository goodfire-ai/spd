from typing import Any

from pydantic import BaseModel


class ActivationContext(BaseModel):
    # raw_text: str
    # offset_mapping: list[tuple[int, int]]
    token_strings: list[str]
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
    mean_ci: float


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


class RunRequest(BaseModel):
    prompt: str


class SubcomponentAblationRequest(BaseModel):
    prompt_id: str
    subcomponent_mask: dict[str, list[list[int]]]


class SubcomponentAblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]


class TokenAblationEffect(BaseModel):
    original_active_count: int
    """Number of components that were active (above threshold) in the original run"""
    ablated_count: int
    """Number of originally-active components that were ablated"""
    ablated_magnitude: float
    """Total CI magnitude lost due to ablation"""


class LayerAblationEffect(BaseModel):
    module: str
    token_abl_effect: list[TokenAblationEffect]


class AblationEffect(BaseModel):
    layer_abl_effect: list[LayerAblationEffect]


class AblationResponse(BaseModel):
    token_logits: list[list[OutputTokenLogit]]
    ablation_effect: AblationEffect


class CombineMasksRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]
    """List of token indices (positions) for which to combine active components"""
    description: str | None = None


class MaskDTO(BaseModel):
    id: str
    layer: str
    description: str | None
    combined_mask: SparseVector


class CombineMasksResponse(BaseModel):
    mask_id: str
    mask: MaskDTO


class SimulateMergeRequest(BaseModel):
    prompt_id: str
    layer: str
    token_indices: list[int]


class SimulateMergeResponse(BaseModel):
    l0: int
    jacc: float


class TrainRun(BaseModel):
    wandb_path: str
    config: dict[str, Any]


class Status(BaseModel):
    train_run: TrainRun | None


class AvailablePrompt(BaseModel):
    index: int
    full_text: str


class ApplyMaskRequest(BaseModel):
    prompt_id: str
    mask_id: str

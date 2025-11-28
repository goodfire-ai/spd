from pydantic import BaseModel


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


class TrainRun(BaseModel):
    wandb_path: str
    config_yaml: str


class Status(BaseModel):
    train_run: TrainRun | None


class SubcomponentMetadata(BaseModel):
    """Lightweight metadata for a subcomponent (without examples/token_prs)"""

    subcomponent_idx: int
    mean_ci: float


class HarvestMetadata(BaseModel):
    """Lightweight metadata returned after harvest, containing only indices and mean_ci values"""

    harvest_id: str
    layers: dict[str, list[SubcomponentMetadata]]

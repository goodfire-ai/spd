from pydantic import BaseModel


class ActivationContext(BaseModel):
    token_strings: list[str]
    token_ci_values: list[float]
    active_position: int
    ci_value: float


class TokenPR(BaseModel):
    """Token Precision and Recall for a given subcomponent"""

    token: str
    """Token string"""
    recall: float
    """Recall: P(token | subcomponent firing)"""
    precision: float
    """Precision: P(subcomponent firing | token)"""


class SubcomponentActivationContexts(BaseModel):
    subcomponent_idx: int
    examples: list[ActivationContext]
    token_prs: list[TokenPR]
    mean_ci: float


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

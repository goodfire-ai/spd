from pydantic import BaseModel


class ActivationContext(BaseModel):
    token_strings: list[str]
    token_ci_values: list[float]
    active_position: int
    ci_value: float


class TokenDensity(BaseModel):
    token: str
    recall: float
    precision: float


class SubcomponentActivationContexts(BaseModel):
    subcomponent_idx: int
    examples: list[ActivationContext]
    token_densities: list[TokenDensity]
    mean_ci: float


class ModelActivationContexts(BaseModel):
    layers: dict[str, list[SubcomponentActivationContexts]]


class TrainRun(BaseModel):
    wandb_path: str
    config_yaml: str


class Status(BaseModel):
    train_run: TrainRun | None

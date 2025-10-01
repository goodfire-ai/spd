import wandb
from pydantic import BaseModel

PROJECT = "spd"


class Run(BaseModel):
    id: str
    url: str


class WandBService:
    def __init__(self):
        self.api = wandb.Api()

    def get_runs(self) -> list[Run]:
        return [Run(id=run.id, url=run.url) for run in self.api.runs(PROJECT)]

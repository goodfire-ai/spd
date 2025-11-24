from dataclasses import dataclass


@dataclass
class Command:
    command: str
    env_vars: dict[str, str] | None = None

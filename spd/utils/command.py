from dataclasses import dataclass, field


@dataclass
class Command:
    command: str
    env_vars: dict[str, str] = field(default_factory=dict)

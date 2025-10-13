import json
from pathlib import Path
from typing import Any, ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Pydantic BaseModel suited for configs.

    Enforces extra="forbid" and frozen=True and adds loading and saving from/to YAML, JSON, and
    JSON string (these are prefixed with "json:").
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def load(cls, path_or_obj: Path | str | dict[str, Any]) -> Self:
        """Load config from path to a JSON or YAML file, or a raw dictionary or json string.

        Json strings are passed in with a "json:" prefix. This is to prevent Fire from
        inconveniently parsing the string as a dictionary and e.g. converting "null" to None.

        Args:
            path_or_obj: Path to config file, or JSON string prefixed with "json:", or dictionary

        Returns:
            The config object
        """
        if isinstance(path_or_obj, str) and path_or_obj.startswith("json:"):
            json_str = path_or_obj[5:]  # Remove "json:" prefix
            assert json_str[0] == "{", f"Json string starts with {json_str[0]} instead of {{"
            return cls.model_validate(json.loads(json_str))

        if isinstance(path_or_obj, str):
            path_or_obj = Path(path_or_obj)

        match path_or_obj:
            case Path() if path_or_obj.suffix == ".json":
                data = json.loads(path_or_obj.read_text())
            case Path() if path_or_obj.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(path_or_obj.read_text())
            case dict():
                data = path_or_obj
            case _:
                raise ValueError(f"Unsupported path_or_obj type: {type(path_or_obj)}")

        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Save config to file (format inferred from extension)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        match path.suffix:
            case ".json":
                path.write_text(self.model_dump_json(indent=2))
            case ".yaml" | ".yml":
                path.write_text(yaml.dump(self.model_dump(mode="json")))
            case _:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

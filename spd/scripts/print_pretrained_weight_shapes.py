"""Load a pretrained model specified in an SPD config and print weight matrix shapes.

This script reads a YAML config to find `pretrained_model_class` and
`pretrained_model_name` (supports W&B paths like "wandb:goodfire/spd/runs/<id>").
It then loads the model using the class's `from_pretrained` method and prints the
shapes of all parameters whose names end with "weight" and which have 2+ dimensions.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml


def resolve_class(path: str):
    """Resolve a class from a dotted path without importing large project deps."""
    import importlib

    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def read_pretrained_info(config_path: Path) -> tuple[str, str]:
    """Read the pretrained model class path and name from a YAML config file.

    Args:
        config_path: Absolute path to the YAML config file.

    Returns:
        Tuple of (pretrained_model_class, pretrained_model_name).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_class_path = cfg.get("pretrained_model_class")
    model_name = cfg.get("pretrained_model_name")

    if not model_class_path or not model_name:
        raise ValueError("Config must contain 'pretrained_model_class' and 'pretrained_model_name'")

    return str(model_class_path), str(model_name)


def load_model(model_class_path: str, model_name: str) -> torch.nn.Module:
    """Resolve the model class and load it via its from_pretrained method.

    Args:
        model_class_path: Dotted import path to the model class.
        model_name: Identifier passed to from_pretrained (e.g., a W&B path).

    Returns:
        Instantiated torch.nn.Module loaded with pretrained weights.
    """
    model_class = resolve_class(model_class_path)
    model: torch.nn.Module

    # Special-case for SimpleStories models when available
    if model_class_path.startswith("simple_stories_train"):
        try:
            from simple_stories_train.run_info import RunInfo as SSRunInfo  # type: ignore

            if hasattr(model_class, "from_run_info"):
                run_info = SSRunInfo.from_path(model_name)
                model = model_class.from_run_info(run_info)  # type: ignore[attr-defined]
            elif hasattr(model_class, "from_pretrained"):
                model = model_class.from_pretrained(model_name)  # type: ignore[attr-defined]
            else:
                raise AttributeError(
                    f"Model class {model_class} has neither 'from_run_info' nor 'from_pretrained'"
                )
        except ModuleNotFoundError:
            # Fallback to from_pretrained if simple_stories_train isn't installed
            if not hasattr(model_class, "from_pretrained"):
                raise AttributeError(
                    f"Model class {model_class} does not define a 'from_pretrained' method"
                )
            model = model_class.from_pretrained(model_name)  # type: ignore[attr-defined]
    else:
        if not hasattr(model_class, "from_pretrained"):
            raise AttributeError(
                f"Model class {model_class} does not define a 'from_pretrained' method"
            )
        model = model_class.from_pretrained(model_name)  # type: ignore[attr-defined]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Loaded object is not a torch.nn.Module")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def iter_weight_matrices(model: torch.nn.Module) -> Iterable[tuple[str, torch.Size]]:
    """Yield (name, shape) for all weight parameters with 2+ dimensions.

    Args:
        model: The loaded neural network model.

    Yields:
        Pairs of parameter name and tensor size for matrix-like weights.
    """
    for name, param in model.named_parameters():
        if name.split(".")[-1] == "weight" and param.ndim >= 2:
            yield name, param.size()


def format_shape(shape: torch.Size) -> str:
    return " x ".join(str(dim) for dim in shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Load pretrained model from config and print all weight matrix shapes.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/mnt/polished-lake/home/lucius/spd/spd/experiments/lm/ss_gpt2_simple_config.yaml"
        ),
        help="Absolute path to SPD YAML config containing pretrained model info.",
    )
    args = parser.parse_args()

    model_class_path, model_name = read_pretrained_info(args.config)
    print(f"Loading model class: {model_class_path}")
    print(f"from_pretrained identifier: {model_name}")

    model = load_model(model_class_path, model_name)

    weight_entries = sorted(iter_weight_matrices(model), key=lambda kv: kv[0])
    if not weight_entries:
        print("No 2D weight matrices found.")
        return

    name_padding = max(len(name) for name, _ in weight_entries)
    print("\nWeight matrix shapes:\n")
    for name, shape in weight_entries:
        print(f"{name.ljust(name_padding)} : {format_shape(shape)}")
    print(f"\nTotal weight matrices: {len(weight_entries)}")


if __name__ == "__main__":
    main()

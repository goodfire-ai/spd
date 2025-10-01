"""Utility to load a pretrained LM from a YAML config and print SVD of q_proj.

This script:
- Parses a SPD experiment YAML config
- Loads the specified pretrained model class and weights
- Extracts the attention query projection for layer 1: model.model.layers[1].self_attn.q_proj
- Computes and prints its singular values

Default config path points to the SimpleStories Llama config used in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from spd.utils.general_utils import resolve_class


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_pretrained_model(config: dict[str, Any]) -> torch.nn.Module:
    pretrained_model_class_path = config["pretrained_model_class"]
    pretrained_model_name: str | None = config.get("pretrained_model_name")
    pretrained_model_path: str | None = config.get("pretrained_model_path")

    model_class = resolve_class(pretrained_model_class_path)

    # Prefer an explicit path if provided; otherwise fall back to HF repo name
    load_id = pretrained_model_path or pretrained_model_name
    assert load_id is not None, (
        "Config must set either 'pretrained_model_path' or 'pretrained_model_name'"
    )

    # Mirror logic used in the repo's LM decomposition: standard classes use from_pretrained
    # Keep everything on CPU and in float32 for numerical stability of SVD
    model = model_class.from_pretrained(  # type: ignore[attr-defined]
        load_id,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def get_layer1_q_proj_matrix(model: torch.nn.Module) -> Tensor:
    # HF Llama structure: LlamaForCausalLM.model.layers[1].self_attn.q_proj
    q_proj = model.model.layers[1].mlp.gate_proj  # type: ignore[attr-defined]
    # q_proj is nn.Linear; weight shape: (out_features, in_features)
    weight: Tensor = q_proj.weight.detach().to(dtype=torch.float32, device="cpu")  # type: ignore[assignment]
    return weight


def compute_singular_values(weight: Tensor) -> Tensor:
    # Use torch.linalg.svdvals for direct singular values computation
    # Ensure on CPU float32
    weight = weight.to(dtype=torch.float32, device="cpu")
    svals = torch.linalg.svdvals(weight)
    return svals


def main() -> None:
    parser = argparse.ArgumentParser(description="Print singular values of q_proj for layer 1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/mnt/polished-lake/home/lucius/spd/spd/experiments/lm/ss_llama_config.yaml"),
        help="Path to SPD YAML config",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    config = load_yaml_config(args.config)
    model = load_pretrained_model(config)
    weight = get_layer1_q_proj_matrix(model)

    # Optional shape check for the expected SimpleStories-1.25M model
    if tuple(weight.shape) != (128, 128):
        print(
            f"Warning: unexpected q_proj weight shape {tuple(weight.shape)} (expected (128, 128))"
        )

    svals = compute_singular_values(weight)

    # Print as a simple Python list for readability
    print("Singular values (descending):")
    # torch returns sorted descending for svdvals
    vals = svals.cpu().tolist()
    indexed = ", ".join(f"{i + 1}: {v}" for i, v in enumerate(vals))
    print(indexed)


if __name__ == "__main__":
    main()

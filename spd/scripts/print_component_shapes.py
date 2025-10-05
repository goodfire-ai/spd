from __future__ import annotations

import argparse
from collections.abc import Iterable

from spd.models.component_model import ComponentModel


def _ensure_wandb_prefix(path: str) -> str:
    """Add the "wandb:" prefix if missing to comply with loader expectations."""
    return path if path.startswith("wandb:") else f"wandb:{path}"


def _print_component_shapes(model: ComponentModel, module_names: Iterable[str]) -> None:
    for module_name in module_names:
        component = model.components.get(module_name)
        if component is None:
            print(f"[WARN] Components not found for '{module_name}'")
            continue

        V_shape = tuple(component.V.shape)
        U_shape = tuple(component.U.shape)
        print(f"{module_name}: V.shape={V_shape}, U.shape={U_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SPD component matrix shapes for a run")
    parser.add_argument(
        "run_path",
        type=str,
        nargs="?",
        default="goodfire/spd/hb28dic0",
        help=(
            "W&B run path like 'goodfire/spd/<run_id>' or with prefix 'wandb:goodfire/spd/<run_id>'"
        ),
    )
    args = parser.parse_args()

    run_path = _ensure_wandb_prefix(args.run_path)

    model = ComponentModel.from_pretrained(run_path)
    model.eval()
    model.requires_grad_(False)

    targets = [
        # Identity insertion that runs before gate_proj
        "model.layers.0.mlp.gate_proj.pre_identity",
        # MLP projections in layer 0
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.up_proj",
    ]

    _print_component_shapes(model, targets)


if __name__ == "__main__":
    main()

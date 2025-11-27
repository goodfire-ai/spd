"""Export attribution data from PyTorch .pt format to JSON for web visualization."""

import json
from pathlib import Path

import torch


def export_attributions_to_json(
    pt_path: Path,
    output_path: Path | None = None,
    alive_indices: dict[str, list[int]] | None = None,
    n_blocks: int = 2,
) -> Path:
    """Convert global attributions from .pt to JSON format.

    Args:
        pt_path: Path to the global_attributions_*.pt file.
        output_path: Output JSON path. Defaults to same name with .json extension.
        alive_indices: Optional alive indices dict. If None, inferred from tensor shapes.
        n_blocks: Number of transformer blocks.

    Returns:
        Path to the created JSON file.
    """
    global_attributions = torch.load(pt_path, weights_only=False)

    # Convert attributions to JSON-serializable format
    attributions_json = {}
    inferred_alive: dict[str, int] = {}

    for (in_layer, out_layer), attr_tensor in global_attributions.items():
        key = f"('{in_layer}', '{out_layer}')"
        attributions_json[key] = attr_tensor.cpu().tolist()

        # Infer alive counts from tensor shapes
        n_in, n_out = attr_tensor.shape
        if in_layer not in inferred_alive:
            inferred_alive[in_layer] = n_in
        if out_layer not in inferred_alive:
            inferred_alive[out_layer] = n_out

    # Build alive_indices if not provided
    if alive_indices is None:
        alive_indices = {layer: list(range(n)) for layer, n in inferred_alive.items()}

    # Create output structure
    data = {
        "n_blocks": n_blocks,
        "attributions": attributions_json,
        "alive_indices": alive_indices,
    }

    # Determine output path
    if output_path is None:
        output_path = pt_path.with_suffix(".json")

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Exported attributions to {output_path}")
    print(f"  - {len(attributions_json)} layer pairs")
    print(f"  - {sum(len(v) for v in alive_indices.values())} total alive components")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export attribution data to JSON")
    parser.add_argument("pt_file", type=Path, help="Path to global_attributions_*.pt file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-n", "--n-blocks", type=int, default=2, help="Number of blocks")

    args = parser.parse_args()

    export_attributions_to_json(
        pt_path=args.pt_file,
        output_path=args.output,
        n_blocks=args.n_blocks,
    )

#!/usr/bin/env python3
"""Export attribution data in an optimized format for web visualization."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch


def optimize_attributions_for_web(
    pt_path: Path,
    output_path: Path | None = None,
    threshold: float = 0.0,
) -> Path:
    """Export attributions optimized for web visualization.

    Args:
        pt_path: Path to the .pt file containing global attributions
        output_path: Output path for JSON (defaults to same name with .json)
        threshold: Values below this are set to 0 to reduce file size (0 = keep all)

    Returns:
        Path to the created JSON file
    """
    # Load the PyTorch data
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Check if it's the full format or just attributions
    if isinstance(data, dict) and "n_blocks" in data:
        # Full format from calc_attributions.py
        global_attributions = data["attributions"]
        alive_indices = data.get("alive_indices", {})
        n_blocks = data.get("n_blocks", 2)
    else:
        # Just the attributions tensor dict
        global_attributions = data
        alive_indices = {}
        n_blocks = 2  # Default

    # Convert to optimized format
    print("Optimizing attributions for web...")
    attributions_sparse = {}
    total_values = 0
    kept_values = 0

    for (in_layer, out_layer), attr_tensor in global_attributions.items():
        key = f"('{in_layer}', '{out_layer}')"

        # Convert to numpy
        arr = attr_tensor.cpu().numpy() if torch.is_tensor(attr_tensor) else np.array(attr_tensor)
        total_values += arr.size

        # Store as sparse format: list of [i, j, value] for non-zero values
        sparse_data = []
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = float(arr[i, j])
                if abs(val) > threshold:
                    # Keep full precision - no rounding
                    sparse_data.append([i, j, val])
                    kept_values += 1

        # Store both sparse and shape info
        attributions_sparse[key] = {"shape": list(arr.shape), "data": sparse_data}

    # Infer alive indices if not provided
    if not alive_indices:
        print("Inferring alive indices from tensor shapes...")
        for (in_layer, out_layer), attr_tensor in global_attributions.items():
            shape = attr_tensor.shape if torch.is_tensor(attr_tensor) else attr_tensor.shape
            if in_layer not in alive_indices:
                alive_indices[in_layer] = list(range(shape[0]))
            if out_layer not in alive_indices:
                alive_indices[out_layer] = list(range(shape[1]))

    # Create output structure
    json_data = {
        "n_blocks": n_blocks,
        "attributions_sparse": attributions_sparse,
        "alive_indices": alive_indices,
        "format": "sparse",
        "stats": {
            "total_values": total_values,
            "kept_values": kept_values,
            "compression_ratio": f"{(1 - kept_values / total_values) * 100:.1f}%",
            "threshold": threshold,
        },
    }

    # Determine output path
    if output_path is None:
        output_path = pt_path.with_suffix(".web.json")

    # Save JSON
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(json_data, f, separators=(",", ":"))

    # Also save compressed version
    gz_path = output_path.with_suffix(".json.gz")
    with gzip.open(gz_path, "wt") as f:
        json.dump(json_data, f, separators=(",", ":"))

    print(f"✓ Saved optimized JSON to {output_path}")
    print(f"✓ Saved compressed version to {gz_path}")
    print(f"  Original values: {total_values:,}")
    print(f"  Kept values: {kept_values:,} ({kept_values / total_values * 100:.1f}%)")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {gz_path.stat().st_size / 1024:.2f} KB")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export attributions for web visualization")
    parser.add_argument("pt_file", type=Path, help="Path to .pt file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.0, help="Threshold for sparsity (0 = keep all)"
    )

    args = parser.parse_args()

    optimize_attributions_for_web(
        pt_path=args.pt_file,
        output_path=args.output,
        threshold=args.threshold,
    )

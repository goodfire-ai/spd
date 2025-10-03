#!/usr/bin/env python3
"""
Automatically visualize component weights and their interactions with embedding/unembedding matrices.

This script automatically detects common embedding/unembedding module names and creates heatmaps showing:
1. Rank-one matrices (V @ U) for each component
2. W_in^c @ W_embedding interactions (if embedding found)
3. W_unembedding @ W_out^c interactions (if unembedding found)

Usage:
    python spd/scripts/visualize_component_weights_auto.py <model_path> [--output-dir <dir>]
"""

import argparse
from pathlib import Path

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.plotting import plot_component_weight_heatmaps
from spd.utils.distributed_utils import get_device


def find_embedding_unembedding_modules(model: ComponentModel) -> tuple[str | None, str | None]:
    """Automatically find embedding and unembedding modules in the target model."""
    embedding_module = None
    unembedding_module = None
    
    # Get all parameter names in the target model
    all_param_names = [name for name, _ in model.target_model.named_parameters()]
    
    print("Available parameters in target model:")
    for name in sorted(all_param_names):
        print(f"  - {name}")
    
    # Look for W_E and W_U parameters directly
    if "W_E" in all_param_names:
        embedding_module = "W_E"
        print(f"Found embedding parameter: {embedding_module}")
    
    if "W_U" in all_param_names:
        unembedding_module = "W_U"
        print(f"Found unembedding parameter: {unembedding_module}")
    
    # If not found, try common module names
    if embedding_module is None:
        embedding_names = [
            "token_embed", "embed", "embedding", "embeddings", "token_embeddings",
            "wte", "word_embeddings", "token_embedding"
        ]
        
        all_module_names = [name for name, _ in model.target_model.named_modules()]
        
        for name in all_module_names:
            if any(embed_name in name.lower() for embed_name in embedding_names):
                try:
                    module = model.target_model.get_submodule(name)
                    if hasattr(module, 'weight') and hasattr(module, 'num_embeddings'):
                        embedding_module = name
                        print(f"Found embedding module: {name}")
                        break
                except Exception:
                    continue
    
    if unembedding_module is None:
        unembedding_names = [
            "unembed", "lm_head", "classifier", "head", "output_projection"
        ]
        
        all_module_names = [name for name, _ in model.target_model.named_modules()]
        
        for name in all_module_names:
            if any(unembed_name in name.lower() for unembed_name in unembedding_names):
                try:
                    module = model.target_model.get_submodule(name)
                    if hasattr(module, 'weight') and hasattr(module, 'in_features'):
                        unembedding_module = name
                        print(f"Found unembedding module: {name}")
                        break
                except Exception:
                    continue
    
    return embedding_module, unembedding_module


def main():
    parser = argparse.ArgumentParser(description="Automatically visualize component weights and embedding interactions")
    parser.add_argument("model_path", type=str, help="Path to the trained SPD model")
    parser.add_argument("--output-dir", type=str, default="component_weight_plots",
                       help="Directory to save the plots")
    parser.add_argument("--figsize", type=float, nargs=2, default=[4, 3],
                       help="Figure size per component (width height)")
    parser.add_argument("--dpi", type=int, default=150,
                       help="DPI for the plots")
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    device = get_device()
    run_info = SPDRunInfo.from_path(args.model_path)
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Components: {list(model.components.keys())}")
    print(f"Number of components per layer: {[comp.C for comp in model.components.values()]}")
    
    # Automatically find embedding and unembedding modules
    print("\nSearching for embedding and unembedding modules...")
    embedding_module, unembedding_module = find_embedding_unembedding_modules(model)
    
    if embedding_module is None:
        print("Warning: No embedding module found. Skipping input-embedding interaction plots.")
    if unembedding_module is None:
        print("Warning: No unembedding module found. Skipping output-unembedding interaction plots.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate the plots
    print("\nGenerating component weight heatmaps...")
    figures = plot_component_weight_heatmaps(
        model=model,
        embedding_module_name=embedding_module,
        unembedding_module_name=unembedding_module,
        figsize_per_component=tuple(args.figsize),
        dpi=args.dpi,
    )
    
    # Save the plots
    print(f"\nSaving plots to {output_dir}...")
    for name, fig_img in figures.items():
        filename = f"{name}.png"
        filepath = output_dir / filename
        fig_img.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"\nAll plots saved to {output_dir}!")
    print("\nGenerated plots:")
    for name in figures:
        print(f"  - {name}.png")


if __name__ == "__main__":
    main()


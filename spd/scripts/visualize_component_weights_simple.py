#!/usr/bin/env python3
"""
Simplified component weight visualization script that works around import issues.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the spd directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    parser = argparse.ArgumentParser(description="Visualize component weights")
    parser.add_argument("model_path", type=str, help="Path to the trained SPD model")
    parser.add_argument("--output-dir", type=str, default="component_analysis",
                       help="Directory to save the plots")
    
    args = parser.parse_args()
    
    try:
        # Try to import and run the visualization
        from spd.models.component_model import ComponentModel, SPDRunInfo
        from spd.plotting import plot_component_weight_heatmaps
        from spd.utils.distributed_utils import get_device
        
        print(f"Loading model from {args.model_path}...")
        device = get_device()
        run_info = SPDRunInfo.from_path(args.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        print(f"Components: {list(model.components.keys())}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate the plots
        print("Generating component weight heatmaps...")
        figures = plot_component_weight_heatmaps(
            model=model,
            embedding_module_name=None,  # Try without embedding first
            unembedding_module_name=None,
        )
        
        # Save the plots
        print(f"Saving plots to {output_dir}...")
        for name, fig_img in figures.items():
            filename = f"{name}.png"
            filepath = output_dir / filename
            fig_img.save(filepath)
            print(f"Saved: {filepath}")
        
        print(f"All plots saved to {output_dir}!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This might be due to a syntax error in the component_model.py file.")
        print("Please check the file for any syntax issues around line 599.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Test script to demonstrate the new multi-magnitude causal importance plots."""

import sys
from pathlib import Path

# Add the spd package to the path
sys.path.insert(0, str(Path(__file__).parent))

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.plotting import plot_multi_magnitude_causal_importance_vals
from spd.models.sigmoids import SigmoidTypes

def test_multi_magnitude_plots():
    """Test the new multi-magnitude plotting functionality."""
    
    # Load a model (using the same model as in the component visualization script)
    model_path = "wandb:goodfire/spd/in10bwoo"
    print(f"Loading model from: {model_path}")
    
    try:
        run_info = SPDRunInfo.from_path(model_path)
        model = ComponentModel.from_run_info(run_info)
        model.eval()
        print(f"Successfully loaded model with {len(model.components)} component modules")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Define test parameters
    batch_shape = (1, model.target_model.config.n_features)
    input_magnitudes = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Smaller set for testing
    sampling = "continuous"
    sigmoid_type = "leaky_hard"
    
    print(f"Testing with input magnitudes: {input_magnitudes}")
    print(f"Batch shape: {batch_shape}")
    
    try:
        # Generate the multi-magnitude plots
        figures, perm_indices = plot_multi_magnitude_causal_importance_vals(
            model=model,
            batch_shape=batch_shape,
            input_magnitudes=input_magnitudes,
            sampling=sampling,
            sigmoid_type=sigmoid_type,
            plot_raw_cis=False,  # Only plot upper leaky for now
        )
        
        print(f"Generated {len(figures)} figures:")
        for key in figures.keys():
            print(f"  - {key}")
        
        # Save the figures to test output
        output_dir = Path("test_multi_magnitude_output")
        output_dir.mkdir(exist_ok=True)
        
        for key, figure in figures.items():
            output_path = output_dir / f"{key}.png"
            figure.save(output_path)
            print(f"Saved {key} to {output_path}")
        
        print("✅ Multi-magnitude plotting test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_magnitude_plots()

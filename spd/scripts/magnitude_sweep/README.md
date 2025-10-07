# Magnitude Sweep Analysis

This script creates plots showing how individual neuron activations and causal importance values respond as we gradually increase the magnitude of a one-hot input vector from 0 to max_magnitude.

## Usage

### Using configuration file:
```bash
python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml
```


## Configuration

The script uses a YAML configuration file (`magnitude_sweep_config.yaml`) with the following parameters:

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `feature_idx`: Which feature to activate (default: 0)
- `n_steps`: Number of steps from 0 to max_magnitude (default: 100)
- `max_magnitude`: Maximum input magnitude (default: 2.0)
- `ci_threshold`: CI threshold for active gates (default: 0.1)
- `pre_activation`: Show pre-activation values (before ReLU) instead of post-activation (default: false)
- `figsize_per_subplot`: Figure size per subplot (width height) (default: [2, 1.5])
- `dpi`: DPI for figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (optional, defaults to 'out' directory relative to script location)

## Output

The script generates unified grid plots showing:
- Individual neuron activations in the ResidMLP layers
- Output responses
- Causal importance function values for gates that actually activate
- Pre-sigmoid gate outputs
- Gate inputs (inner acts)

Plots are saved as PNG files in model-specific subdirectories within the output directory:

**Directory Structure:**
```
out/
└── {model_id}/
    ├── unified_grid_feature_0_layers_0_mlp_in.png
    ├── unified_grid_feature_0_layers_0_mlp_out.png
    └── ...
```

**File Naming:**
- `unified_grid_feature_0_layers_0_mlp_in.png` - Feature 0, Layer 0 input component
- `unified_grid_feature_0_layers_0_mlp_out.png` - Feature 0, Layer 0 output component

**Model ID Examples:**
- Local files: `resid_mlp` (from `resid_mlp.pth`)
- Wandb URLs: `2ki9tfsx` (from `wandb:goodfire/spd/runs/2ki9tfsx`)

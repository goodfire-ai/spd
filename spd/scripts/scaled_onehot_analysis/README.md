# Scaled One-Hot Vector Analysis

This script analyzes how ResidMLP models respond to scaled one-hot vectors by creating plots showing the relationship between input scale and output dimensions.

## Overview

For a given ResidMLP model, the script:
1. Creates scaled one-hot vectors (one feature active at different scales)
2. Passes these through the model to get outputs
3. Creates plots showing:
   - X-axis: Scale of the one-hot vector
   - Y-axis: Output dimension values
   - Multiple lines: One per output dimension
   - Optional: Comparison to expected target function (e.g., ReLU(coeff*x) + x for resid_mlp models)

## Usage

### Basic Usage
```bash
python spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis.py spd/scripts/scaled_onehot_analysis/scaled_onehot_analysis_config.yaml
```


## Configuration

The script accepts a YAML configuration file with the following parameters:

- `model_path`: Path to the trained model (wandb:project/run_id or local path)
- `model_type`: Model type - "spd", "target", or "auto" (default: "auto")
  - "spd": Load as SPD model (ComponentModel)
  - "target": Load as target ResidMLP model directly
  - "auto": Auto-detect based on path (wandb: URLs → SPD, local paths → target)
- `min_scale`: Minimum scale for one-hot vectors (default: -2.0)
- `max_scale`: Maximum scale for one-hot vectors (default: 2.0)
- `n_steps`: Number of scale steps (default: 100)
- `n_features_to_plot`: Number of input features to analyze (default: 20)
- `subtract_inputs`: Whether to subtract inputs from outputs (default: true)
- `compare_to_target`: Whether to compare model output to expected target function (default: false)
- `figsize`: Figure size per plot [width, height] (default: [12, 8])
- `dpi`: DPI for figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (default: null, uses script directory/out)

## Target Function Comparison

When `compare_to_target: true`, the script will overlay the expected target function on the plots. For resid_mlp models, this is typically a residual function of the form:

**`y_i = ReLU(coeff_i * x_i) + x_i`**

Where:
- `coeff_i` is the label coefficient for dimension `i` (learned during target model training)
- This creates a residual connection where the output is the input plus a ReLU transformation

The comparison helps verify that the model is learning the correct target function rather than a simple ReLU.

## Output

The script generates individual plots for each input feature:

**One plot per input feature per layer**: Each plot shows how one specific input feature affects all output dimensions as the scale changes.

Plots are saved as PNG files in model-specific subdirectories within the output directory:

**Directory Structure:**
```
out/
└── {model_id}/
    ├── scaled_onehot_layers_0_mlp_in_input_0.png
    ├── scaled_onehot_layers_0_mlp_in_input_1.png
    ├── scaled_onehot_layers_0_mlp_out_input_0.png
    └── ...
```

**File Naming:**
- `scaled_onehot_layers_0_mlp_in_input_0.png` - Layer 0 input, input feature 0
- `scaled_onehot_layers_0_mlp_in_input_1.png` - Layer 0 input, input feature 1
- `scaled_onehot_layers_0_mlp_out_input_0.png` - Layer 0 output, input feature 0
- etc.

**Model ID Examples:**
- Local files: `resid_mlp` (from `resid_mlp.pth`)
- Wandb URLs: `2ki9tfsx` (from `wandb:goodfire/spd/runs/2ki9tfsx`)

## Example Output

Each plot shows:
- X-axis: Scale of the one-hot vector (from min_scale to max_scale)
- Y-axis: Output dimension values (or output - input if subtract_inputs=true)
- Multiple colored lines: Each representing one output dimension response to the single active input
- Grid and legend for clarity (or summary box for models with many outputs)
- Title indicating the specific input feature being analyzed

## Model Support

The script supports both SPD models and target ResidMLP models:

### SPD Models (ComponentModel)
- Contains a trained ResidMLP target model with component decompositions
- Loaded using `SPDRunInfo.from_path()` and `ComponentModel.from_run_info()`
- Automatically detected when using wandb: URLs
- Example: `wandb:goodfire/spd/runs/any9ekl9`

### Target Models (ResidMLP)
- Direct ResidMLP models without component decompositions
- Loaded using `ResidMLPTargetRunInfo.from_path()` and `run_info.load_model()`
- Automatically detected for local file paths
- Example: `/path/to/trained_residmlp.pt`

### Model Structure Requirements
Both model types must contain a ResidMLP with the following structure:
- Input embedding layer (W_E)
- Multiple MLP layers with `mlp_in` and `mlp_out` components
- Output unembedding layer (W_U)

Layer names should follow the pattern: `layers.{i}.mlp_in` and `layers.{i}.mlp_out`

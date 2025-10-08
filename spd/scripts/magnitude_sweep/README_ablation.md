# Ablation Sweep Analysis

This script extends the magnitude sweep analysis by performing component ablation studies. It identifies which components activate during a magnitude sweep and then creates ablation versions by setting the causal importance of each active component to zero, which directly masks their contribution.

## What it does

1. **Baseline Analysis**: Runs a magnitude sweep to identify which components activate (causal importance > threshold)
2. **Component Ablation**: For each active component, creates a version where that component's causal importance is set to zero
3. **Comparative Plots**: Generates unified grid plots showing the effect of ablating each component

## Key Features

- **Sparse Component Detection**: Identifies the typically 1-2 components that activate over the magnitude range
- **Individual Ablation**: When multiple components activate, ablates them one at a time (not simultaneously)
- **Comprehensive Visualization**: Shows neurons, outputs, causal importance, gate outputs, and gate inputs
- **Slope Analysis**: Fits separate lines to output features, neuron activations, and gate inputs for positive and negative magnitudes, displaying slopes in the plots
- **Organized Output**: Creates separate directories for each ablation scenario

## Usage

### Using configuration file:
```bash
python spd/scripts/magnitude_sweep/ablation_sweep.py spd/scripts/magnitude_sweep/ablation_sweep_config.yaml
```

## Configuration

The script uses a YAML configuration file (`ablation_sweep_config.yaml`) with the same parameters as the magnitude sweep:

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

## Output Structure

The script creates a hierarchical directory structure:

```
out/
└── {model_id}/
    ├── unified_grid_feature_{feature_idx}_layers_{layer}_mlp_in.png  # Baseline plots
    └── ablation_layers_{layer}_mlp_in_component_{comp_idx}/          # Ablation plots
        ├── unified_grid_feature_{feature_idx}_layers_0_mlp_in.png
        └── unified_grid_feature_{feature_idx}_layers_1_mlp_in.png
```

### Example Output

For a model with:
- Layer 0: Component 291 active
- Layer 1: Components 3 and 300 active

The output structure would be:
```
out/5q3xxmmy/
├── unified_grid_feature_1_layers_0_mlp_in.png          # Baseline
├── unified_grid_feature_1_layers_1_mlp_in.png          # Baseline
├── ablation_layers_0_mlp_in_component_291/             # Ablate comp 291
│   ├── unified_grid_feature_1_layers_0_mlp_in.png
│   └── unified_grid_feature_1_layers_1_mlp_in.png
├── ablation_layers_1_mlp_in_component_3/               # Ablate comp 3
│   ├── unified_grid_feature_1_layers_0_mlp_in.png
│   └── unified_grid_feature_1_layers_1_mlp_in.png
└── ablation_layers_1_mlp_in_component_300/             # Ablate comp 300
    ├── unified_grid_feature_1_layers_0_mlp_in.png
    └── unified_grid_feature_1_layers_1_mlp_in.png
```

## Interpretation

### Baseline Plots
Show the normal behavior with all components active.

### Ablation Plots
Show how the model behavior changes when a specific component is removed:
- **Neuron Activations**: How hidden layer neurons respond differently
- **Output Responses**: How the final output changes
- **Causal Importance**: How the remaining components' importance changes
- **Gate Outputs**: How the gating mechanisms are affected
- **Gate Inputs**: How the input to gating functions changes

### Key Insights
- **Component Role**: Compare baseline vs ablation to understand what each component does
- **Compensation**: See if other components compensate when one is removed
- **Causal Effects**: Understand the causal role of individual components
- **Sparse Activation**: Confirm that only a few components are actually important
- **Slope Analysis**: Compare slopes of output feature responses to understand how ablation affects the linearity of the response

## Technical Details

### Component Ablation
Components are ablated by setting their causal importance to zero, which then directly masks their contribution:
```python
ablation_ci[:, component_to_ablate] = 0.0  # Zero out the specified component's causal importance
mask_infos = make_mask_infos(component_masks=ablation_causal_importances)
```

### Active Component Detection
Components are considered active if their maximum causal importance across all magnitude steps exceeds the threshold:
```python
max_ci_per_component = torch.max(ci, dim=0)[0]
active_indices = torch.where(max_ci_per_component > ci_threshold)[0]
```

### Individual Ablation
When multiple components activate in the same layer, each is ablated separately to understand their individual contributions.

### Slope Analysis
The plots include slope analysis for output features, neuron activations, and gate inputs:
- **Data Splitting**: Data is split at magnitude = 0 into positive and negative regions
- **Line Fitting**: Separate linear fits are performed for each region using `np.polyfit`
- **Visualization**: Fitted lines are shown as dashed lines (blue for negative, green for positive)
- **Slope Display**: Slope values are displayed as text in the subplot with 3 decimal precision (positive slope first, then negative slope)
- **Interpretation**: Compare slopes between baseline and ablation to understand how component removal affects the linearity of the response

## Example Results

The script output shows:
```
Layer layers.0.mlp_in: 1 active components out of 400
  Active components: [291]
  Max CI values: [1.0]
Layer layers.1.mlp_in: 2 active components out of 400
  Active components: [3, 300]
  Max CI values: [0.9251130819320679, 1.0]
Created 3 ablation plots
```

This indicates that:
- Layer 0 has 1 active component (291) with perfect activation (CI = 1.0)
- Layer 1 has 2 active components (3, 300) with high activation
- 3 separate ablation experiments were created (1 for layer 0, 2 for layer 1)

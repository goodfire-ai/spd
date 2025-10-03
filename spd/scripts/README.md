# Component Matrix Visualization

This script analyzes and visualizes SPD component matrices to understand component behavior patterns.

## Features

- **Component Matrix Visualization**: Creates heatmaps of rank-one matrices (V @ U) for each component
- **Gate Activation Analysis**: Analyzes which components activate for which inputs (similar to causal importance analysis)
- **Activation Pattern Detection**: Identifies universal components that activate for many inputs
- **Input-Specific Analysis**: Shows how many components activate for each input

## Usage

```bash
# Basic usage
python visualize_component_matrices.py "wandb:goodfire/spd/byfsymhw"

# With custom threshold and output directory
python visualize_component_matrices.py "wandb:goodfire/spd/byfsymhw" --threshold 0.1 --figsize 10 8

# Help
python visualize_component_matrices.py --help
```

## Arguments

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `--output`: Output path for the combined plot (default: results/component_matrices.png)
- `--output-dir`: Directory to save individual component plots (default: results/component_plots)
- `--device`: Device to use (default: auto)
- `--threshold`: Threshold for considering a gate as 'active' (default: 0.1)
- `--figsize`: Figure size per component (width height) (default: 8 6)
- `--dpi`: DPI for the figures (default: 150)

## Output Files

The script creates a `results/` directory with:

- `component_matrices.png`: Combined plot showing all component matrices
- `component_plots/`: Individual plots for each layer's components
- `activation_plots/`: Heatmaps showing which components activate for which inputs
- `*_activation_matrix.pt`: PyTorch tensors containing binary activation matrices

## Analysis Features

### Component Analysis
- Identifies components that activate for many inputs (potential universal components)
- Shows activation ratios and statistics for each component
- Ranks components by activation frequency

### Input Analysis
- Shows how many components activate for each input
- Displays which specific components are active for each input
- Counts input activation patterns (e.g., how many inputs have exactly 1, 2, 3+ components active)

### Pattern Detection
- **Universal Components**: Components that activate for >80% of inputs (flagged with 🚨)
- **Broad Activation**: Components that activate for >50% of inputs (flagged with ⚠️)
- **Input-Specific Patterns**: Shows the ideal 1:1 mapping vs problematic patterns

## Example Output

```
layers.1.mlp_in:
  Top 10 most active components:
    Component 241: 98/100 inputs activate gate (0.980)
      🚨 BROAD ACTIVATION: Component 241 gate activates for 98.0% of inputs!
  
  Input activation patterns:
    98 inputs have exactly 2 components active
  
  🚨 UNIVERSAL COMPONENTS (activate for >80% of inputs):
    Component 241: 98.0% of inputs
```

This indicates that layer 1 has a universal component (241) that activates for almost all inputs, plus each input has one additional specific component - breaking the desired 1:1 sparsity pattern.

# SPD Scripts

This directory contains analysis and visualization scripts for SPD (Sparse Probabilistic Decomposition) models.

## Available Scripts

### 📊 **Analysis Scripts**

#### `magnitude_sweep/` - Magnitude Sweep Analysis
Analyzes how ResidMLP models respond to increasing input magnitudes, plotting both individual neuron activations and causal importance values.

**Usage:**
```bash
python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml
```

**Features:**
- Unified grid plots showing neurons, output, causal importance, gate outputs, and gate inputs
- Comprehensive analysis of model response to input magnitude changes
- Configurable parameters for different analysis scenarios

#### `component_visualization/` - Component Matrix Visualization
Creates heatmaps of rank-one matrices (V @ U) for each component and analyzes component behavior patterns.

**Usage:**
```bash
python spd/scripts/component_visualization/component_visualization.py spd/scripts/component_visualization/component_visualization_config.yaml
```

**Features:**
- Component matrix heatmaps with red/blue/white color coding
- Gate activation analysis showing which components activate for which inputs
- Activation pattern detection and universal component identification
- Input-specific analysis and pattern statistics

#### `causal_importance_sweep/` - Causal Importance Analysis
Plots how each component's causal importance responds as input magnitude increases.

**Usage:**
```bash
python spd/scripts/causal_importance_sweep/causal_importance_sweep.py spd/scripts/causal_importance_sweep/causal_importance_sweep_config.yaml
```

**Features:**
- Layer-based causal importance plots
- Component-specific response curves
- Analysis of which components are most responsive to input changes

#### `compare_models/` - Model Comparison
Compares two SPD models by computing geometric similarities between their learned subcomponents.

**Usage:**
```bash
python spd/scripts/compare_models/compare_models.py spd/scripts/compare_models/compare_models_config.yaml
```

**Features:**
- Geometric similarity analysis between model components
- Density-based component filtering
- Comprehensive similarity metrics

### 🛠️ **Utility Scripts**

#### `run.py` - Experiment Runner
Unified SPD runner for experiments with optional parameter sweeps.

#### Example Scripts
- `example_magnitude_sweep.py` - Examples for magnitude sweep analysis
- `example_visualize_components.py` - Examples for component visualization

## Script Structure

All analysis scripts follow a consistent structure:

```
script_name/
├── script_name.py          # Main script with fire.Fire() interface
├── script_name_config.yaml # Configuration file
├── README.md               # Detailed documentation
└── out/                    # Results directory (auto-created)
    └── generated_files...
```

## Configuration

Each script supports both:
1. **Configuration files** (recommended): YAML files with all parameters
2. **Command-line arguments**: Direct parameter specification

## Common Parameters

Most scripts share these common parameters:
- `model_path`: Path to trained SPD model (wandb:project/run-id or local path)
- `device`: Device to use (default: "auto")
- `output_dir`: Output directory (default: "out/" relative to script location)

## Getting Started

1. **Choose your analysis type** from the available scripts above
2. **Copy the example config** from the script's directory
3. **Modify the config** with your model path and desired parameters
4. **Run the script** using the config file or command-line arguments

## Examples

### Quick Analysis
```bash
# Magnitude sweep with default settings
python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml

# Component visualization
python spd/scripts/component_visualization/component_visualization.py spd/scripts/component_visualization/component_visualization_config.yaml
```

### Custom Analysis
```bash
# High-resolution magnitude sweep
python spd/scripts/magnitude_sweep/magnitude_sweep.py \
    --model_path "wandb:project/run-id" \
    --n_steps 500 \
    --dpi 300 \
    --figsize_per_subplot 24 16
```

## Output

All scripts save results to their respective `out/` directories with:
- **Plots**: PNG files with analysis visualizations
- **Data**: PT files with computed matrices and tensors
- **Logs**: Console output with analysis statistics

## Troubleshooting

- **Model loading errors**: Ensure the model path is correct and accessible
- **Memory errors**: Reduce step counts or other memory-intensive parameters
- **Empty plots**: Check that thresholds aren't too high
- **Device errors**: Verify CUDA availability or use `--device cpu`

For detailed usage instructions, see the README.md file in each script's directory.
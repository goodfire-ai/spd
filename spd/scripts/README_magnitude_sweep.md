# ResidMLP Magnitude Sweep Analysis

This directory contains scripts for analyzing how ResidMLP models respond to increasing input magnitudes, plotting both individual neuron activations and causal importance values.

## Scripts

### `plot_residmlp_magnitude_sweep.py`

The main script that:
1. Takes a trained ResidMLP model (via ComponentModel/SPD)
2. Incrementally increases the magnitude of a one-hot input vector
3. Plots individual neuron activations in the ResidMLP layers
4. Plots causal importance function values for gates that actually activate
5. Creates both detailed per-layer plots and summary plots

### `example_magnitude_sweep.py`

Example script showing different ways to use the main script with various parameters.

## Usage

### Basic Usage

```bash
python spd/scripts/plot_residmlp_magnitude_sweep.py <model_path>
```

### With Custom Parameters

```bash
python spd/scripts/plot_residmlp_magnitude_sweep.py \
    wandb:your-project/run-id \
    --output-dir results/my_analysis \
    --feature-idx 5 \
    --n-steps 200 \
    --max-magnitude 3.0 \
    --max-neurons 50 \
    --ci-threshold 0.1
```

### Parameters

- `model_path`: Path to trained SPD model (wandb:project/run-id or local path)
- `--output-dir`: Output directory for plots (default: spd/scripts/results/magnitude_sweep_plots)
- `--device`: Device to use (default: auto)
- `--feature-idx`: Which feature to activate (default: 0)
- `--n-steps`: Number of magnitude steps (default: 100)
- `--max-magnitude`: Maximum input magnitude (default: 2.0)
- `--figsize`: Figure size as width height (default: 15 10)
- `--dpi`: DPI for figures (default: 150)
- `--max-neurons`: Max neurons per plot (default: 50)
- `--ci-threshold`: CI threshold for active gates (default: 0.1)
- `--summary-only`: Only create summary plot

## Output

The script generates several types of plots:

### 1. Detailed Per-Layer Plots
- **File format**: `magnitude_sweep_feature_{idx}_{layer_name}.png`
- **Content**: 
  - Top subplot: Individual neuron activations vs input magnitude
  - Bottom subplot: Causal importance values vs input magnitude (only for active gates)
- **Purpose**: Detailed analysis of each layer's response

### 2. Summary Plot
- **File format**: `magnitude_sweep_summary_feature_{idx}.png`
- **Content**: Grid showing mean activations and mean causal importance for all layers
- **Purpose**: Overview of the entire model's response

## Understanding the Plots

### X-Axis: Input Magnitude
- Range: 0 to `max_magnitude`
- Represents the strength of the one-hot input vector
- Shows how the model responds to increasing input intensity

### Y-Axes:

#### Neuron Activations
- Shows the activation values of individual neurons in each ResidMLP layer
- Each line represents one neuron
- Colors are assigned using viridis colormap
- Helps understand which neurons are most responsive to input changes

#### Causal Importance
- Shows the causal importance values for SPD component gates
- Only displays gates that exceed the `ci_threshold` at some point
- Each line represents one component/gate
- Colors are assigned using plasma colormap
- Helps understand which components are most important for the input

## Example Workflows

### 1. Quick Analysis
```bash
# Basic analysis with default settings
python spd/scripts/plot_residmlp_magnitude_sweep.py wandb:project/run-id
```

### 2. High-Resolution Analysis
```bash
# High-quality plots for publications
python spd/scripts/plot_residmlp_magnitude_sweep.py wandb:project/run-id \
    --n-steps 500 \
    --dpi 300 \
    --figsize 24 16 \
    --max-neurons 100
```

### 3. Multiple Features
```bash
# Analyze different input features
for feature in 0 5 10 15 20; do
    python spd/scripts/plot_residmlp_magnitude_sweep.py wandb:project/run-id \
        --feature-idx $feature \
        --summary-only
done
```

### 4. Sensitivity Analysis
```bash
# Test different magnitude ranges
python spd/scripts/plot_residmlp_magnitude_sweep.py wandb:project/run-id \
    --max-magnitude 5.0 \
    --n-steps 1000 \
    --ci-threshold 0.01
```

## Interpreting Results

### Neuron Activation Patterns
- **Linear responses**: Neurons that scale linearly with input
- **Threshold responses**: Neurons that activate only above certain magnitudes
- **Saturation**: Neurons that plateau at high magnitudes
- **Non-monotonic**: Neurons with complex response patterns

### Causal Importance Patterns
- **Early activation**: Components that become important at low magnitudes
- **Late activation**: Components that only matter at high magnitudes
- **Peak importance**: Components with optimal magnitude ranges
- **Persistent importance**: Components that remain important across all magnitudes

## Technical Details

### Model Requirements
- The script expects a trained ComponentModel containing a ResidMLP target model
- The ResidMLP should have been trained and decomposed using SPD
- Model must be loadable via `SPDRunInfo.from_path()`

### Memory Considerations
- Large `n_steps` values require more memory
- High `max-neurons` values create more complex plots
- Consider using `--summary-only` for quick overviews

### Performance
- Computation time scales with `n_steps × n_features × n_components`
- GPU acceleration is used when available
- Consider reducing parameters for large models

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure the model path is correct and accessible
2. **Memory errors**: Reduce `n_steps` or `max-neurons`
3. **Empty plots**: Check that `ci_threshold` isn't too high
4. **Feature index errors**: Ensure `feature_idx < n_features`

### Getting Help

- Check the example script for usage patterns
- Use `--help` flag to see all available options
- Ensure you're running from the project root directory
- Verify that the model was trained with SPD decomposition

## Related Scripts

- `plot_causal_importance_sweep_main.py`: Similar analysis for general component models
- `resid_mlp_interp.py`: ResidMLP-specific interpretation tools
- `visualize_component_matrices.py`: Component visualization tools

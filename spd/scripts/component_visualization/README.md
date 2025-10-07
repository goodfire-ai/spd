# Component Visualization

This script loads a trained SPD model and creates heatmaps of the rank-one matrices (V @ U) for each component, using red for positive values, blue for negative values, and white for zero.

## Usage

### Using configuration file:
```bash
python spd/scripts/component_visualization/component_visualization.py spd/scripts/component_visualization/component_visualization_config.yaml
```


## Configuration

The script uses a YAML configuration file (`component_visualization_config.yaml`) with the following parameters:

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `threshold`: Threshold for considering a value as 'active' (default: 0.01)
- `figsize`: Figure size per component (width height) (default: [8, 6])
- `dpi`: DPI for the figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (optional, defaults to 'out' directory relative to script location)

## Output

The script generates:
- **Component matrix heatmaps**: Combined and individual plots showing the learned component weights
- **Activation pattern analysis**: Analysis of which components activate for which inputs
- **Activation matrices**: Saved as .pt files for further analysis
- **Activation pattern heatmaps**: Visualizations showing component activation patterns

Results are saved to the `out/` directory by default, or to a custom directory if specified in the configuration.

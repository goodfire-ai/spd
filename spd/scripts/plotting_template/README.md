# Plotting Template

This is a template plotting script that follows the same structure as other scripts in spd/scripts.

## Usage

### Using configuration file:
```bash
python spd/scripts/plotting_template/plotting_template.py spd/scripts/plotting_template/plotting_template_config.yaml
```

## Configuration

The script uses a YAML configuration file (`plotting_template_config.yaml`) with the following parameters:

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `figsize`: Figure size (width height) (default: [8, 6])
- `dpi`: DPI for the figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (optional, defaults to 'out' directory relative to script location)

## Output

The script generates plots in the `out/` directory by default, or to a custom directory if specified in the configuration.

## Customization

To customize this template for your specific plotting needs:

1. Modify the `PlottingTemplateConfig` class to add your specific parameters
2. Update the `create_plot` function with your actual plotting code
3. Update the configuration file with your default values
4. Update this README with your specific documentation

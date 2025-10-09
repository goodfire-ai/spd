"""Usage examples for the extended hook system."""

from typing import Any

import torch

from spd.models.component_model import ComponentModel
from spd.models.extended_hooks import HookConfig


def basic_usage_example():
    """Basic usage of extended hooks with ComponentModel."""

    # Assume we have a trained ComponentModel
    model: ComponentModel = load_trained_model()  # Your model loading code here

    # Configure what to capture
    hook_config = HookConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=True,
        capture_intermediates=True,  # For ResidMLP pre/post activations
        clone_tensors=True,
        detach_tensors=True,
    )

    # Single forward pass captures everything
    input_tensor = torch.randn(1, 100)  # Example input
    _output, captured_data = model.forward_with_extended_hooks(
        input_tensor, hook_config=hook_config
    )

    # Access captured quantities
    target_inputs = captured_data["target_inputs"]
    _target_outputs = captured_data["target_outputs"]
    component_inner_acts = captured_data["component_inner_acts"]
    ci_fn_outputs = captured_data["ci_fn_outputs"]
    _causal_importances = captured_data["causal_importances"]
    _residmlp_pre_activations = captured_data.get("residmlp_pre_activations", {})
    _final_output = captured_data["final_output"]

    print(f"Captured data keys: {list(captured_data.keys())}")
    print(f"Target inputs shape: {target_inputs['layers.0.mlp_in'].shape}")
    print(f"Component inner acts shape: {component_inner_acts['layers.0.mlp_in'].shape}")
    print(f"CI function outputs shape: {ci_fn_outputs['layers.0.mlp_in'].shape}")


def magnitude_sweep_replacement_example() -> dict[str, Any]:
    """Example showing how to replace magnitude sweep functionality."""

    model: ComponentModel = load_trained_model()

    # Configure for magnitude sweep analysis
    hook_config = HookConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=True,
        capture_intermediates=True,
        clone_tensors=True,
        detach_tensors=True,
        log_capture_stats=True,
    )

    # Magnitude sweep parameters
    n_features = 100
    feature_idx = 0
    n_steps = 50
    max_magnitude = 2.0

    # Storage for sweep data
    activations = {}
    causal_importances = {}
    gate_outputs = {}
    gate_inputs = {}
    output_responses = {}

    # Get layer names
    layer_names = list(model.components.keys())

    # Initialize storage
    for layer_name in layer_names:
        activations[layer_name] = torch.zeros(n_steps, model.components[layer_name].C)
        causal_importances[layer_name] = torch.zeros(n_steps, model.components[layer_name].C)
        gate_outputs[layer_name] = torch.zeros(n_steps, model.components[layer_name].C)
        gate_inputs[layer_name] = torch.zeros(n_steps, model.components[layer_name].C)
        output_responses[layer_name] = torch.zeros(n_steps, n_features)

    # Perform magnitude sweep
    for step_idx in range(n_steps):
        magnitude = max_magnitude * step_idx / (n_steps - 1)

        # Create one-hot input with specified magnitude
        input_tensor = torch.zeros(1, n_features)
        input_tensor[0, feature_idx] = magnitude

        # Single forward pass captures everything
        output, captured_data = model.forward_with_extended_hooks(
            input_tensor, hook_config=hook_config
        )

        # Store results
        for layer_name in layer_names:
            if layer_name in captured_data["target_inputs"]:
                activations[layer_name][step_idx] = captured_data["target_inputs"][layer_name][0]

            if layer_name in captured_data["causal_importances"]:
                causal_importances[layer_name][step_idx] = captured_data["causal_importances"][
                    layer_name
                ][0]

            if layer_name in captured_data["ci_fn_outputs"]:
                gate_outputs[layer_name][step_idx] = captured_data["ci_fn_outputs"][layer_name][0]

            if layer_name in captured_data["component_inner_acts"]:
                gate_inputs[layer_name][step_idx] = captured_data["component_inner_acts"][
                    layer_name
                ][0]

            output_responses[layer_name][step_idx] = output[0]

    return {
        "activations": activations,
        "causal_importances": causal_importances,
        "gate_outputs": gate_outputs,
        "gate_inputs": gate_inputs,
        "output_responses": output_responses,
    }


def manual_hook_management_example():
    """Example of manual hook management for custom workflows."""

    model: ComponentModel = load_trained_model()

    # Create custom hook configuration
    hook_config = HookConfig(
        capture_inputs=True,
        capture_outputs=False,  # Don't capture outputs to save memory
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=False,  # Calculate manually later
        clone_tensors=False,  # Use references to save memory
        detach_tensors=True,
    )

    # Get hook manager for manual control
    hook_manager = model.get_extended_hook_manager(hook_config)

    with hook_manager.hooks_active():
        # Perform multiple forward passes
        for i in range(5):
            input_tensor = torch.randn(1, 100)
            _output = model(input_tensor)

            # Access captured data after each forward pass
            captured_data = hook_manager.capture_data
            print(f"Pass {i}: Captured {len(captured_data.get('target_inputs', {}))} inputs")

        # Get summary of all captures
        summary = hook_manager.get_capture_summary()
        print(f"Total captures: {summary['total_captures']}")
        print(f"Data keys: {summary['data_keys']}")


def performance_optimized_example():
    """Example with performance optimizations."""

    model: ComponentModel = load_trained_model()

    # Optimized configuration for large-scale analysis
    hook_config = HookConfig(
        capture_inputs=True,
        capture_outputs=False,  # Skip outputs to save memory
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=True,
        capture_intermediates=False,  # Skip ResidMLP intermediates
        clone_tensors=False,  # Use references instead of clones
        detach_tensors=True,
        validate_shapes=False,  # Skip shape validation for speed
        log_capture_stats=False,  # Disable logging
    )

    # Batch processing
    batch_size = 32
    input_tensor = torch.randn(batch_size, 100)

    _output, captured_data = model.forward_with_extended_hooks(
        input_tensor, hook_config=hook_config
    )

    # Process captured data efficiently
    for layer_name, inputs in captured_data["target_inputs"].items():
        print(f"Layer {layer_name}: {inputs.shape}")


def error_handling_example():
    """Example showing error handling capabilities."""

    model: ComponentModel = load_trained_model()

    # Configuration that might cause issues
    hook_config = HookConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_component_inner_acts=True,
        capture_ci_fn_outputs=True,
        capture_causal_importances=True,
        capture_intermediates=True,
        validate_shapes=True,
        log_capture_stats=True,
    )

    try:
        # This will handle errors gracefully
        input_tensor = torch.randn(1, 100)
        _output, captured_data = model.forward_with_extended_hooks(
            input_tensor, hook_config=hook_config
        )

        print("Forward pass completed successfully")
        print(f"Captured data keys: {list(captured_data.keys())}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        # The hook system will have cleaned up automatically


def load_trained_model() -> ComponentModel:
    """Placeholder for model loading - replace with your actual loading code."""
    # This is just a placeholder - replace with your actual model loading
    raise NotImplementedError("Replace with your model loading code")


if __name__ == "__main__":
    # Run examples
    print("Basic usage example:")
    try:
        basic_usage_example()
    except NotImplementedError:
        print("Model loading not implemented - skipping example")

    print("\nMagnitude sweep replacement example:")
    try:
        magnitude_sweep_replacement_example()
    except NotImplementedError:
        print("Model loading not implemented - skipping example")

    print("\nManual hook management example:")
    try:
        manual_hook_management_example()
    except NotImplementedError:
        print("Model loading not implemented - skipping example")

    print("\nPerformance optimized example:")
    try:
        performance_optimized_example()
    except NotImplementedError:
        print("Model loading not implemented - skipping example")

    print("\nError handling example:")
    try:
        error_handling_example()
    except NotImplementedError:
        print("Model loading not implemented - skipping example")

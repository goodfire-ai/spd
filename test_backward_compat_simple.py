"""Test backward compatibility by loading a model WITHOUT lm_head decomposition."""

import torch

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device

print("Testing backward compatibility with models WITHOUT lm_head decomposition...")
print("=" * 80)

# Use a local TMS run that doesn't have lm_head
local_run_dir = "/mnt/polished-lake/spd/runs/local-0bico796"
checkpoint_path = f"{local_run_dir}/model_50.pth"

print(f"\nLoading model from checkpoint: {checkpoint_path}")
run_info = SPDRunInfo.from_path(checkpoint_path)

print(f"\nTarget module patterns: {run_info.config.target_module_patterns}")
assert "lm_head" not in run_info.config.target_module_patterns, (
    "Test setup error: run should not have lm_head"
)

# Load the model
model = ComponentModel.from_run_info(run_info)
device = get_device()
model = model.to(device)
model.eval()

# Verify lm_head is NOT in target_module_paths
print(f"\ntarget_module_paths: {model.target_module_paths}")
assert "lm_head" not in model.target_module_paths, "lm_head should not be in target_module_paths"

# Test that model can perform forward pass
print("\nTesting model forward pass...")
task_name = run_info.config.task_config.task_name
if task_name == "tms":
    # TMS model expects float inputs with shape [..., n_features]
    n_features = model.target_model.config.n_features
    test_batch = torch.randn(2, n_features, device=device)
    print(f"  TMS model with n_features={n_features}")
elif task_name == "lm":
    # LM model expects long inputs with shape [batch, seq]
    test_batch = torch.zeros(2, 3, dtype=torch.long, device=device)
    print("  LM model")
else:
    print(f"  Unknown task: {task_name}, skipping forward pass test")
    test_batch = None

if test_batch is not None:
    with torch.no_grad():
        output = model(test_batch)
    # Handle both Tensor and OutputWithCache
    output_shape = output.output.shape if hasattr(output, "output") else output.shape
    print(f"  Forward pass successful! Output shape: {output_shape}")

print("\n" + "=" * 80)
print("✅ BACKWARD COMPATIBILITY TEST PASSED!")
print("=" * 80)
print("- Model without lm_head loads successfully")
print("- Model can perform forward pass correctly")
print(f"- Checkpoint path: {checkpoint_path}")
print(f"- Target modules: {run_info.config.target_module_patterns}")

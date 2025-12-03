"""Minimal test to profile local attribution computation speed."""

import time
from pathlib import Path

import torch
from spd.attributions.compute import compute_local_attributions, get_sources_by_target
from spd.attributions.db import LocalAttrDB
from transformers import AutoTokenizer

from spd.models.component_model import ComponentModel, SPDRunInfo

# Load model
db = LocalAttrDB(Path("local_attr_new.db"))
wandb_info = db.get_meta("wandb_path")
n_blocks_info = db.get_meta("n_blocks")
assert wandb_info is not None and n_blocks_info is not None
wandb_path = wandb_info["path"]
n_blocks = n_blocks_info["n_blocks"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Loading model from {wandb_path}...")

t0 = time.time()
run_info = SPDRunInfo.from_path(wandb_path)
model = ComponentModel.from_run_info(run_info)
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(run_info.config.tokenizer_name)
print(f"Model loaded in {time.time() - t0:.1f}s")

# Get sources_by_target
sampling = run_info.config.sampling
print("Computing sources_by_target...")
t0 = time.time()
sources_by_target = get_sources_by_target(model, device, sampling, n_blocks)
print(f"sources_by_target computed in {time.time() - t0:.1f}s")

# Get a prompt
prompt = db.get_prompt(1)
assert prompt is not None
print(f"Prompt has {len(prompt.token_ids)} tokens")

# Compute attributions with different thresholds
tokens = torch.tensor([prompt.token_ids], device=device)

for output_prob_threshold in [0.1, 0.05, 0.01]:
    print(f"\n--- output_prob_threshold={output_prob_threshold} ---")
    t0 = time.time()
    result = compute_local_attributions(
        model=model,
        tokens=tokens,
        sources_by_target=sources_by_target,
        ci_threshold=1e-6,
        output_prob_threshold=output_prob_threshold,
        sampling=sampling,
        device=device,
        show_progress=True,  # Show tqdm
    )
    elapsed = time.time() - t0
    print(f"Computed {len(result.pairs)} pairs in {elapsed:.1f}s")

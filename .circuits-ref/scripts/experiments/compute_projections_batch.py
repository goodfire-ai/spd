#!/usr/bin/env python3
"""Compute output/input projections for neurons in batch.

Usage:
    # Single task (for SLURM array)
    python scripts/compute_projections_batch.py --task-id 0 --num-tasks 32 \
        --input data/fineweb_50k_edge_stats_enriched.json \
        --output-dir outputs/projections

    # Merge results
    python scripts/compute_projections_batch.py --merge \
        --input data/fineweb_50k_edge_stats_enriched.json \
        --output-dir outputs/projections \
        --output data/fineweb_50k_edge_stats_enriched.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_projections(model, tokenizer, layer, neuron, top_k=20):
    """Compute output and input projections for a single neuron."""
    # Output projection: down_proj column -> lm_head
    down_proj = model.model.layers[layer].mlp.down_proj.weight
    output_direction = down_proj[:, neuron].float()
    
    lm_head = model.lm_head.weight.float()
    logit_contributions = lm_head @ output_direction
    
    # Top promoted/suppressed
    top_vals, top_idx = logit_contributions.topk(k=top_k)
    bot_vals, bot_idx = logit_contributions.topk(k=top_k, largest=False)
    
    promoted = [{"token": tokenizer.decode([idx.item()]), "weight": round(val.item(), 4)} 
                for idx, val in zip(top_idx, top_vals)]
    suppressed = [{"token": tokenizer.decode([idx.item()]), "weight": round(val.item(), 4)}
                  for idx, val in zip(bot_idx, bot_vals)]
    
    # Input projection: up_proj row (what activates this neuron)
    up_proj = model.model.layers[layer].mlp.up_proj.weight
    input_direction = up_proj[neuron, :].float()
    
    # Project onto embedding matrix to see what tokens activate
    embed = model.model.embed_tokens.weight.float()
    input_contributions = embed @ input_direction
    
    top_input_vals, top_input_idx = input_contributions.topk(k=top_k)
    
    activates = [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(), 
                  "weight": round(val.item(), 6)}
                 for idx, val in zip(top_input_idx, top_input_vals)]
    
    return {
        "output_projection": {"promoted": promoted, "suppressed": suppressed},
        "input_projection": {"activates": activates}
    }


def run_task(task_id, num_tasks, input_path, output_dir, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Process a subset of neurons for this task."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load edge stats to get neuron list
    print(f"Task {task_id}/{num_tasks}: Loading neuron list...", flush=True)
    with open(input_path) as f:
        data = json.load(f)
    
    profiles = data["profiles"]
    
    # Split neurons across tasks
    neurons_per_task = (len(profiles) + num_tasks - 1) // num_tasks
    start_idx = task_id * neurons_per_task
    end_idx = min(start_idx + neurons_per_task, len(profiles))
    
    my_profiles = profiles[start_idx:end_idx]
    print(f"Task {task_id}: Processing neurons {start_idx}-{end_idx} ({len(my_profiles)} neurons)", flush=True)
    
    if not my_profiles:
        print(f"Task {task_id}: No neurons to process", flush=True)
        return
    
    # Load model
    print(f"Task {task_id}: Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Process neurons
    results = {}
    for i, p in enumerate(tqdm(my_profiles, desc=f"Task {task_id}")):
        layer, neuron = p["layer"], p["neuron"]
        neuron_id = f"L{layer}/N{neuron}"
        
        try:
            with torch.no_grad():
                proj = compute_projections(model, tokenizer, layer, neuron)
            results[neuron_id] = proj
        except Exception as e:
            print(f"Error processing {neuron_id}: {e}", file=sys.stderr)
            results[neuron_id] = {"error": str(e)}
    
    # Save results
    output_file = output_dir / f"projections_task_{task_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    print(f"Task {task_id}: Saved {len(results)} projections to {output_file}", flush=True)


def merge_results(input_path, output_dir, output_path):
    """Merge projection results back into edge stats."""
    output_dir = Path(output_dir)
    
    # Load all projection files
    print("Loading projection files...", flush=True)
    all_projections = {}
    for pf in sorted(output_dir.glob("projections_task_*.json")):
        with open(pf) as f:
            proj = json.load(f)
        all_projections.update(proj)
        print(f"  Loaded {len(proj)} from {pf.name}", flush=True)
    
    print(f"Total projections: {len(all_projections)}", flush=True)
    
    # Load edge stats
    print(f"Loading {input_path}...", flush=True)
    with open(input_path) as f:
        data = json.load(f)
    
    # Add projections to profiles
    added = 0
    for p in data["profiles"]:
        neuron_id = p["neuron_id"]
        if neuron_id in all_projections:
            proj = all_projections[neuron_id]
            if "error" not in proj:
                p["output_projection"] = proj["output_projection"]
                p["input_projection"] = proj["input_projection"]
                added += 1
    
    print(f"Added projections to {added}/{len(data['profiles'])} profiles", flush=True)
    
    # Update metadata
    data["metadata"]["projections_added"] = True
    data["metadata"]["projections_count"] = added
    
    # Save
    print(f"Saving to {output_path}...", flush=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Done! Size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, help="Task ID for parallel processing")
    parser.add_argument("--num-tasks", type=int, default=32, help="Total number of tasks")
    parser.add_argument("--input", type=Path, required=True, help="Input edge stats JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/projections"))
    parser.add_argument("--output", type=Path, help="Final output file (for merge)")
    parser.add_argument("--merge", action="store_true", help="Merge results instead of computing")
    
    args = parser.parse_args()
    
    if args.merge:
        merge_results(args.input, args.output_dir, args.output or args.input)
    else:
        if args.task_id is None:
            parser.error("--task-id required for computation")
        run_task(args.task_id, args.num_tasks, args.input, args.output_dir)


if __name__ == "__main__":
    main()

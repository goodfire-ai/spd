#!/usr/bin/env python3
"""Batch neuron investigation using SLURM.

Distributes neuron investigations across multiple GPUs with parallel agents per GPU.
Runs the full NeuronPI pipeline: Scientist → Skeptic → GPT Review.

Usage:
    # Investigate neurons from a circuit graph
    python scripts/batch_slurm_investigate.py \
        --graph graphs/aspirin-cox-target.json \
        --agents-per-gpu 4 \
        --max-gpus 24

    # Investigate specific neurons from a list
    python scripts/batch_slurm_investigate.py \
        --neurons-file /tmp/investigation_order.json \
        --agents-per-gpu 4 \
        --max-gpus 24

    # Single neuron investigation (full PI pipeline)
    python scripts/batch_slurm_investigate.py \
        --neuron L58/N13280 \
        --target-model qwen3-32b

    # Dry run to see job distribution
    python scripts/batch_slurm_investigate.py \
        --graph graphs/aspirin-cox-target.json \
        --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_neurons_from_graph(graph_path: str, v6_only: bool = True) -> list:
    """Extract neurons from a graph file, optionally filtering to v6 only."""
    with open(graph_path) as f:
        graph = json.load(f)

    # Load v6 neurons if filtering
    v6_neurons = set()
    if v6_only:
        v6_path = Path("data/medical_edge_stats_v6_enriched.json")
        if v6_path.exists():
            with open(v6_path) as f:
                v6_data = json.load(f)
            v6_neurons = set((p['layer'], p['neuron']) for p in v6_data['profiles'])

    # Get already investigated
    investigated = set()
    inv_dir = Path("outputs/investigations")
    if inv_dir.exists():
        for f in inv_dir.glob("*_dashboard.json"):
            parts = f.stem.replace("_dashboard", "").split("_")
            if len(parts) == 2:
                try:
                    layer = int(parts[0][1:])
                    neuron = int(parts[1][1:])
                    investigated.add((layer, neuron))
                except:
                    pass

    # Extract MLP neurons
    neurons = []
    for node in graph.get('nodes', []):
        if node.get('feature_type') == 'mlp_neuron':
            try:
                layer = int(node['layer'])
                neuron = node['feature']
                influence = node.get('influence') or 0

                # Skip if already investigated
                if (layer, neuron) in investigated:
                    continue

                # Skip if not in v6 (when filtering)
                if v6_only and v6_neurons and (layer, neuron) not in v6_neurons:
                    continue

                neurons.append({
                    'layer': layer,
                    'neuron': neuron,
                    'influence': influence
                })
            except:
                pass

    # Sort by layer (late to early), then influence
    neurons.sort(key=lambda x: (-x['layer'], -x['influence']))
    return neurons


def get_neurons_from_file(neurons_file: str) -> list:
    """Load neurons from a JSON file."""
    with open(neurons_file) as f:
        data = json.load(f)

    # Handle list of tuples or list of dicts
    neurons = []
    for item in data:
        if isinstance(item, list):
            neurons.append({'layer': item[0], 'neuron': item[1], 'influence': 0})
        elif isinstance(item, dict):
            neurons.append(item)
    return neurons


def create_worker_script(neurons: list, job_id: int, output_dir: Path,
                         edge_stats: str = None, max_concurrent: int = 4,
                         project_root: str = None, target_model: str = None,
                         skip_review: bool = False, skip_skeptic: bool = False) -> str:
    """Create a worker script for a single SLURM job."""

    if project_root is None:
        project_root = str(Path.cwd())

    # Model config setup
    model_config_code = ""
    if target_model:
        model_config_code = f'''
from neuron_scientist.tools import set_model_config
set_model_config("{target_model}")
print(f"Model config set to: {target_model}", flush=True)
'''

    script = f'''#!/usr/bin/env python3
"""Auto-generated worker script for SLURM job {job_id}."""

import asyncio
import json
import sys
from pathlib import Path

# Use absolute path to project root
sys.path.insert(0, "{project_root}")
{model_config_code}
from neuron_scientist import run_neuron_pi

NEURONS = {json.dumps(neurons)}
OUTPUT_DIR = "{Path(output_dir).absolute()}"
EDGE_STATS = {f'"{edge_stats}"' if edge_stats else 'None'}
MAX_CONCURRENT = {max_concurrent}
SKIP_REVIEW = {skip_review}

async def investigate_one(layer: int, neuron: int):
    """Investigate a single neuron with full PI pipeline."""
    neuron_id = f"L{{layer}}/N{{neuron}}"
    print(f"[Job {job_id}] Starting PI investigation: {{neuron_id}}", flush=True)

    try:
        result = await run_neuron_pi(
            neuron_id=neuron_id,
            initial_label="",
            initial_hypothesis="",
            edge_stats_path=EDGE_STATS,
            output_dir=OUTPUT_DIR,
            model="opus",  # PI orchestration model - upgraded for better reasoning
            scientist_model="opus",  # Scientist model - upgraded for predictor/detector distinction
            max_review_iterations=2,
            skip_review=SKIP_REVIEW,
        )
        verdict = result.final_verdict if result.final_verdict else "no-verdict"
        skeptic_verdict = result.skeptic_report.verdict if result.skeptic_report else "no-skeptic"
        print(f"[Job {job_id}] Completed: {{neuron_id}} (verdict={{verdict}}, skeptic={{skeptic_verdict}})", flush=True)
        return True
    except Exception as e:
        print(f"[Job {job_id}] FAILED: {{neuron_id}} - {{e}}", flush=True)
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run investigations with concurrency limit."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_investigate(n):
        async with semaphore:
            return await investigate_one(n['layer'], n['neuron'])

    tasks = [bounded_investigate(n) for n in NEURONS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if r is True)
    print(f"[Job {job_id}] Completed {{success}}/{{len(NEURONS)}} investigations", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
'''
    return script


def submit_slurm_job(
    worker_script: str,
    job_id: int,
    job_name: str,
    partition: str = "h200-reserved",
    output_dir: Path = None,
) -> str:
    """Submit a SLURM job and return the job ID."""

    # Write worker script
    script_path = output_dir / f"worker_{job_id}.py"
    with open(script_path, 'w') as f:
        f.write(worker_script)
    os.chmod(script_path, 0o755)

    # Get API key from .env
    api_key = ""
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break

    # Create SLURM submission script
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output={output_dir}/slurm_{job_id}_%j.out
#SBATCH --error={output_dir}/slurm_{job_id}_%j.err

cd {Path.cwd()}
source .venv/bin/activate
export ANTHROPIC_API_KEY="{api_key}"
python {script_path}
'''

    slurm_path = output_dir / f"slurm_{job_id}.sh"
    with open(slurm_path, 'w') as f:
        f.write(slurm_script)

    # Submit job
    result = subprocess.run(
        ['sbatch', str(slurm_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error submitting job {job_id}: {result.stderr}")
        return None

    # Extract job ID from "Submitted batch job 12345"
    slurm_job_id = result.stdout.strip().split()[-1]
    return slurm_job_id


def main():
    parser = argparse.ArgumentParser(description="Batch neuron investigation via SLURM")

    # Input sources
    parser.add_argument("--graph", help="Graph file to extract neurons from")
    parser.add_argument("--neurons-file", help="JSON file with neuron list")
    parser.add_argument("--neuron", help="Single neuron to investigate (e.g., L58/N13280)")

    # Model configuration
    parser.add_argument("--target-model", default=None,
                        choices=["llama-3.1-8b", "qwen3-32b", "qwen3-8b"],
                        help="Target model for activation testing (default: llama-3.1-8b)")

    # SLURM configuration
    parser.add_argument("--agents-per-gpu", type=int, default=4,
                        help="Number of parallel agents per GPU (default: 4)")
    parser.add_argument("--max-gpus", type=int, default=24,
                        help="Maximum number of GPUs to use (default: 24)")
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition (default: h200-reserved)")

    # Options
    parser.add_argument("--edge-stats", default="data/fineweb_50k_edge_stats_enriched.json",
                        help="Edge stats file for connectivity analysis (default: data/fineweb_50k_edge_stats_enriched.json)")
    parser.add_argument("--output-dir", default="neuron_reports/json",
                        help="Output directory for results (default: neuron_reports/json)")
    parser.add_argument("--v6-only", action="store_true", default=True,
                        help="Only investigate neurons in v6 (default: True)")
    parser.add_argument("--no-v6-filter", action="store_true",
                        help="Investigate all neurons, not just v6")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without submitting jobs")
    parser.add_argument("--job-prefix", default="neuron-pi",
                        help="Prefix for SLURM job names (default: neuron-pi)")
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip GPT review step (Scientist + Skeptic only)")

    args = parser.parse_args()

    # Get neurons to investigate
    if args.neuron:
        # Single neuron mode
        import re
        match = re.match(r'L(\d+)/N(\d+)', args.neuron)
        if not match:
            print(f"Error: Invalid neuron format '{args.neuron}'. Expected format: L58/N13280")
            sys.exit(1)
        layer, neuron = int(match.group(1)), int(match.group(2))
        neurons = [{'layer': layer, 'neuron': neuron, 'influence': 0}]
        source = f"single: {args.neuron}"
    elif args.graph:
        v6_only = not args.no_v6_filter
        neurons = get_neurons_from_graph(args.graph, v6_only=v6_only)
        source = f"graph: {args.graph}"
    elif args.neurons_file:
        neurons = get_neurons_from_file(args.neurons_file)
        source = f"file: {args.neurons_file}"
    else:
        print("Error: Must specify --neuron, --graph, or --neurons-file")
        sys.exit(1)

    if not neurons:
        print("No neurons to investigate!")
        sys.exit(0)

    # Calculate job distribution
    agents_per_gpu = args.agents_per_gpu
    max_gpus = args.max_gpus
    total_neurons = len(neurons)

    # Distribute neurons across GPUs
    neurons_per_gpu = agents_per_gpu  # Each GPU handles this many in parallel
    num_gpus_needed = min((total_neurons + neurons_per_gpu - 1) // neurons_per_gpu, max_gpus)

    # Create batches
    batches = []
    for i in range(0, total_neurons, neurons_per_gpu):
        batch = neurons[i:i + neurons_per_gpu]
        batches.append(batch)

    # Limit to max GPUs - redistribute while keeping layers contiguous
    if len(batches) > max_gpus:
        # Flatten and re-batch with larger batch sizes
        all_neurons = [n for batch in batches for n in batch]
        neurons_per_job = (len(all_neurons) + max_gpus - 1) // max_gpus
        batches = []
        for i in range(0, len(all_neurons), neurons_per_job):
            batches.append(all_neurons[i:i + neurons_per_job])

    print("=" * 60)
    print("SLURM BATCH PI INVESTIGATION")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Target model: {args.target_model or 'llama-3.1-8b (default)'}")
    print(f"Total neurons: {total_neurons}")
    print(f"Agents per GPU: {agents_per_gpu}")
    print(f"Max GPUs: {max_gpus}")
    print(f"Jobs to submit: {len(batches)}")
    print(f"Partition: {args.partition}")
    print(f"Pipeline: Scientist → Skeptic → {'GPT Review' if not args.skip_review else 'SKIP REVIEW'}")
    print("=" * 60)

    # Show distribution
    print("\nJob distribution:")
    for i, batch in enumerate(batches):
        layers = sorted(set(n['layer'] for n in batch), reverse=True)
        layer_str = f"L{max(layers)}-L{min(layers)}" if len(layers) > 1 else f"L{layers[0]}"
        print(f"  Job {i+1}: {len(batch)} neurons ({layer_str})")

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        print("\nFirst batch neurons:")
        for n in batches[0][:10]:
            print(f"  L{n['layer']}/N{n['neuron']} (influence={n['influence']:.3f})")
        return

    # Create output directory for SLURM scripts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_dir = Path(args.output_dir) / f"slurm_{timestamp}"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    print(f"\nSubmitting {len(batches)} jobs...")
    job_ids = []
    for i, batch in enumerate(batches):
        worker_script = create_worker_script(
            neurons=batch,
            job_id=i,
            output_dir=Path(args.output_dir),
            edge_stats=args.edge_stats,
            max_concurrent=args.agents_per_gpu,
            project_root=str(Path.cwd()),
            target_model=args.target_model,
            skip_review=args.skip_review,
        )

        slurm_job_id = submit_slurm_job(
            worker_script=worker_script,
            job_id=i,
            job_name=f"{args.job_prefix}-{i}",
            partition=args.partition,
            output_dir=slurm_dir,
        )

        if slurm_job_id:
            job_ids.append(slurm_job_id)
            print(f"  Submitted job {i+1}/{len(batches)}: SLURM ID {slurm_job_id}")

    print(f"\nSubmitted {len(job_ids)} jobs.")
    print(f"SLURM scripts: {slurm_dir}")
    print(f"Results will be in: {args.output_dir}")
    print("\nMonitor with: squeue -u $USER")

    # Save job manifest
    manifest = {
        'timestamp': timestamp,
        'source': source,
        'target_model': args.target_model or 'llama-3.1-8b',
        'total_neurons': total_neurons,
        'jobs': len(batches),
        'slurm_job_ids': job_ids,
        'output_dir': str(args.output_dir),
        'skip_review': args.skip_review,
        'neurons': [(n['layer'], n['neuron']) for n in neurons]
    }
    manifest_path = slurm_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

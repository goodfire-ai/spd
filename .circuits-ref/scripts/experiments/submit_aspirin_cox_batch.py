#!/usr/bin/env python3
"""Submit NeuronPI batch investigation for Aspirin-Cox circuit neurons.

Optimized configuration:
- 4 agents per GPU (to maximize API parallelism while sharing GPU)
- Up to 36 GPUs
- All 134 neurons in parallel
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
AGENTS_PER_GPU = 4
MAX_GPUS = 36
PARTITION = "h200-reserved"
TIME_LIMIT = "4:00:00"
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# NeuronPI settings
MODEL = "sonnet"  # PI orchestrator model
SCIENTIST_MODEL = "opus"  # NeuronScientist model
MAX_REVIEW_ITERATIONS = 2
SKIP_REVIEW = False
SKIP_DASHBOARD = False


def load_neurons(neurons_file: str) -> list:
    """Load neurons from the aspirin-cox circuit file."""
    with open(neurons_file) as f:
        data = json.load(f)

    neurons = []
    for n in data.get('neurons', []):
        # Parse neuron_id like "L0/N10585"
        neuron_id = n.get('neuron_id', '')
        if '/' in neuron_id:
            layer_part, neuron_part = neuron_id.split('/')
            layer = int(layer_part[1:])  # Remove 'L'
            neuron = int(neuron_part[1:])  # Remove 'N'
            neurons.append({
                'layer': layer,
                'neuron': neuron,
                'title': n.get('title', ''),
            })

    return neurons


def create_worker_script(neurons: list, job_id: int, output_dir: Path) -> str:
    """Create a worker script for a single SLURM job."""

    neurons_json = json.dumps([{'layer': n['layer'], 'neuron': n['neuron']} for n in neurons])

    script = f'''#!/usr/bin/env python3
"""Auto-generated NeuronPI worker script for SLURM job {job_id}."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "{PROJECT_ROOT}")

from neuron_scientist.pi_agent import run_neuron_pi

NEURONS = {neurons_json}
OUTPUT_DIR = Path("{PROJECT_ROOT}/outputs/investigations")
HTML_OUTPUT_DIR = Path("{PROJECT_ROOT}/neuron_reports/html")
JSON_OUTPUT_DIR = Path("{PROJECT_ROOT}/neuron_reports/json")
EDGE_STATS = None
MAX_CONCURRENT = {AGENTS_PER_GPU}
MAX_REVIEW_ITERATIONS = {MAX_REVIEW_ITERATIONS}

async def investigate_one(layer: int, neuron: int):
    """Investigate a single neuron with NeuronPI."""
    neuron_id = f"L{{layer}}/N{{neuron}}"
    print(f"[Job {job_id}] Starting NeuronPI: {{neuron_id}}", flush=True)

    try:
        result = await run_neuron_pi(
            neuron_id=neuron_id,
            initial_label="",
            initial_hypothesis="",
            edge_stats_path=EDGE_STATS,
            output_dir=OUTPUT_DIR,
            model="{MODEL}",
            scientist_model="{SCIENTIST_MODEL}",
            max_review_iterations=MAX_REVIEW_ITERATIONS,
            skip_review={SKIP_REVIEW},
            skip_dashboard={SKIP_DASHBOARD},
        )

        # Copy outputs to canonical locations
        if result.dashboard_path:
            html_src = Path(result.dashboard_path)
            if html_src.exists():
                html_name = f"L{{layer}}_N{{neuron}}.html"
                html_dest = HTML_OUTPUT_DIR / html_name
                HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(html_src, html_dest)
                print(f"[Job {job_id}] HTML: {{html_dest}}", flush=True)

        verdict = result.final_verdict[:50] if result.final_verdict else "completed"
        print(f"[Job {job_id}] Completed: {{neuron_id}} ({{verdict}}...)", flush=True)
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
    failed = sum(1 for r in results if r is not True)
    print(f"[Job {job_id}] Completed {{success}}/{{len(NEURONS)}} ({{failed}} failed)", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
'''
    return script


def submit_slurm_job(worker_script: str, job_id: int, job_name: str, output_dir: Path) -> str:
    """Submit a SLURM job and return the job ID."""

    # Write worker script
    script_path = output_dir / f"worker_{job_id}.py"
    with open(script_path, 'w') as f:
        f.write(worker_script)
    os.chmod(script_path, 0o755)

    # Get API key from .env
    api_key = ""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break

    # Create SLURM submission script
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time={TIME_LIMIT}
#SBATCH --output={output_dir}/slurm_{job_id}_%j.out
#SBATCH --error={output_dir}/slurm_{job_id}_%j.err

cd {PROJECT_ROOT}
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

    slurm_job_id = result.stdout.strip().split()[-1]
    return slurm_job_id


def main():
    # Load neurons
    neurons_file = PROJECT_ROOT / "data" / "aspirin_cox_circuit_neurons.json"
    neurons = load_neurons(neurons_file)

    print("=" * 70)
    print("ASPIRIN-COX CIRCUIT NEURONPI BATCH INVESTIGATION")
    print("=" * 70)
    print(f"Total neurons: {len(neurons)}")
    print(f"Agents per GPU: {AGENTS_PER_GPU}")
    print(f"Max GPUs: {MAX_GPUS}")
    print(f"Partition: {PARTITION}")
    print(f"PI Model: {MODEL}, Scientist Model: {SCIENTIST_MODEL}")
    print("=" * 70)

    # Calculate job distribution
    # Each job handles AGENTS_PER_GPU neurons (one per agent)
    neurons_per_job = AGENTS_PER_GPU
    num_jobs = min((len(neurons) + neurons_per_job - 1) // neurons_per_job, MAX_GPUS)

    # Redistribute if we have more neurons than slots
    if len(neurons) > num_jobs * neurons_per_job:
        neurons_per_job = (len(neurons) + num_jobs - 1) // num_jobs

    # Create batches
    batches = []
    for i in range(0, len(neurons), neurons_per_job):
        batch = neurons[i:i + neurons_per_job]
        batches.append(batch)

    print("\nJob distribution:")
    print(f"  Jobs to submit: {len(batches)}")
    print(f"  Neurons per job: {neurons_per_job}")

    for i, batch in enumerate(batches[:5]):
        layer_range = f"L{min(n['layer'] for n in batch)}-L{max(n['layer'] for n in batch)}"
        print(f"  Job {i}: {len(batch)} neurons ({layer_range})")
    if len(batches) > 5:
        print(f"  ... and {len(batches) - 5} more jobs")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "investigations" / f"slurm_aspirin_cox_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Confirm before submitting
    response = input("\nSubmit jobs? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Submit jobs
    print(f"\nSubmitting {len(batches)} jobs...")
    job_ids = []
    for i, batch in enumerate(batches):
        worker_script = create_worker_script(
            neurons=batch,
            job_id=i,
            output_dir=output_dir,
        )

        slurm_job_id = submit_slurm_job(
            worker_script=worker_script,
            job_id=i,
            job_name=f"aspirin-cox-{i}",
            output_dir=output_dir,
        )

        if slurm_job_id:
            job_ids.append(slurm_job_id)
            print(f"  Submitted job {i+1}/{len(batches)}: SLURM ID {slurm_job_id}")

    print(f"\n{'=' * 70}")
    print(f"SUBMITTED {len(job_ids)} JOBS")
    print(f"{'=' * 70}")
    print("Monitor with: squeue -u $USER")
    print(f"Logs in: {output_dir}")
    print(f"Results in: {PROJECT_ROOT}/outputs/investigations/")
    print(f"HTML reports: {PROJECT_ROOT}/neuron_reports/html/")

    # Save manifest
    manifest = {
        'timestamp': timestamp,
        'circuit': 'aspirin-cox',
        'total_neurons': len(neurons),
        'agents_per_gpu': AGENTS_PER_GPU,
        'jobs': len(batches),
        'slurm_job_ids': job_ids,
        'neurons': [(n['layer'], n['neuron']) for n in neurons],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

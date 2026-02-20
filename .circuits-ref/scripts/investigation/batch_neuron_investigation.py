#!/usr/bin/env python3
"""Batch investigation of multiple neurons using SLURM.

Usage:
    # Investigate selected neurons in parallel
    python scripts/batch_neuron_investigation.py \
        --neurons data/interesting_neurons.json \
        --n-gpus 10 \
        --output-dir outputs/investigations

    # Dry run
    python scripts/batch_neuron_investigation.py \
        --neurons data/interesting_neurons.json \
        --dry-run

    # Worker mode (called by SLURM)
    python scripts/batch_neuron_investigation.py \
        --worker --job-dir <path> --task-id <N>
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_slurm_script(
    job_dir: Path,
    n_tasks: int,
    partition: str,
    time_limit: str,
    mem: str,
    repo_dir: Path,
) -> str:
    """Generate SLURM array job script."""
    return f"""#!/bin/bash
#SBATCH --job-name=neuron-investigate
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
#SBATCH --output={job_dir}/logs/task_%a_%j.out
#SBATCH --error={job_dir}/logs/task_%a_%j.err

echo "=== Neuron Investigation Worker ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================="

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

.venv/bin/python scripts/batch_neuron_investigation.py \\
    --worker \\
    --job-dir {job_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "=== Worker Complete ==="
echo "End time: $(date)"
"""


def generate_finalize_script(job_dir: Path, repo_dir: Path, partition: str) -> str:
    """Generate finalize script."""
    return f"""#!/bin/bash
#SBATCH --job-name=investigate-finalize
#SBATCH --partition={partition}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output={job_dir}/logs/finalize_%j.out
#SBATCH --error={job_dir}/logs/finalize_%j.err

echo "=== Investigation Finalize ==="
cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

.venv/bin/python scripts/batch_neuron_investigation.py \\
    --finalize \\
    --job-dir {job_dir}

echo "=== Complete ==="
"""


def create_job(
    neurons: list[dict],
    n_gpus: int,
    output_dir: Path,
    job_output_dir: Path,
    partition: str,
    time_limit: str,
    mem: str,
    edge_stats_path: Path | None,
    labels_path: Path | None,
    graph_dir: Path | None,
    max_experiments: int,
    llm_model: str,
) -> Path:
    """Create job directory and files."""
    job_id = f"investigate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = job_output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir()
    (job_dir / "partial").mkdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribute neurons across workers
    neurons_per_worker = len(neurons) // n_gpus
    extra = len(neurons) % n_gpus

    for task_id in range(n_gpus):
        start = task_id * neurons_per_worker + min(task_id, extra)
        end = start + neurons_per_worker + (1 if task_id < extra else 0)
        task_neurons = neurons[start:end]

        task_file = job_dir / f"neurons_task_{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_neurons, f)

    # Save config
    config = {
        "n_tasks": n_gpus,
        "total_neurons": len(neurons),
        "output_dir": str(output_dir),
        "edge_stats_path": str(edge_stats_path) if edge_stats_path else None,
        "labels_path": str(labels_path) if labels_path else None,
        "graph_dir": str(graph_dir) if graph_dir else None,
        "max_experiments": max_experiments,
        "llm_model": llm_model,
    }
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate SLURM scripts
    repo_dir = Path(__file__).parent.parent.resolve()
    script = generate_slurm_script(job_dir, n_gpus, partition, time_limit, mem, repo_dir)
    with open(job_dir / "submit.sh", "w") as f:
        f.write(script)

    finalize_script = generate_finalize_script(job_dir, repo_dir, partition)
    with open(job_dir / "finalize.sh", "w") as f:
        f.write(finalize_script)

    return job_dir


def submit_job(job_dir: Path) -> tuple[int | None, int | None]:
    """Submit the SLURM job."""
    try:
        result = subprocess.run(
            ["sbatch", str(job_dir / "submit.sh")],
            capture_output=True,
            text=True,
            check=True,
        )
        array_job_id = int(result.stdout.strip().split()[-1])

        result = subprocess.run(
            ["sbatch", f"--dependency=afterok:{array_job_id}", str(job_dir / "finalize.sh")],
            capture_output=True,
            text=True,
            check=True,
        )
        finalize_job_id = int(result.stdout.strip().split()[-1])

        return array_job_id, finalize_job_id

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
        print("sbatch not found. Is SLURM available?", file=sys.stderr)
        return None, None


async def run_worker_async(job_dir: Path, task_id: int):
    """Execute worker for a single SLURM task."""
    from neuron_scientist import NeuronScientist

    print(f"Worker {task_id} starting...")

    # Load config
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    output_dir = Path(config["output_dir"])
    max_experiments = config.get("max_experiments", 100)
    llm_model = config.get("llm_model", "gpt-5")

    # Load resources
    edge_stats = None
    if config.get("edge_stats_path"):
        edge_stats_path = Path(config["edge_stats_path"])
        if edge_stats_path.exists():
            with open(edge_stats_path) as f:
                edge_stats = json.load(f)

    labels = {}
    if config.get("labels_path"):
        labels_path = Path(config["labels_path"])
        if labels_path.exists():
            with open(labels_path) as f:
                data = json.load(f)
            for lbl in data.get("labels", []):
                labels[lbl.get("neuron_id", "")] = lbl.get("output_label", "")

    neuron_index = {}
    graph_dir = None
    if config.get("graph_dir"):
        graph_dir = Path(config["graph_dir"])
        index_path = graph_dir / "neuron_index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            neuron_index = data.get("index", {})

    # Load neurons for this task
    neurons_file = job_dir / f"neurons_task_{task_id}.json"
    with open(neurons_file) as f:
        neurons = json.load(f)

    print(f"Task {task_id}: Processing {len(neurons)} neurons")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Process each neuron
    results = []
    processed = 0
    failed = 0

    for neuron in neurons:
        neuron_id = neuron.get("neuron_id", "")
        if not neuron_id:
            continue

        try:
            print(f"  Investigating {neuron_id}...")

            scientist = NeuronScientist(
                model=model,
                tokenizer=tokenizer,
                neuron_id=neuron_id,
                initial_label=neuron,
                edge_stats=edge_stats,
                labels=labels,
                graph_dir=graph_dir,
                neuron_index=neuron_index,
                llm_model=llm_model,
            )

            investigation = await scientist.investigate(max_experiments=max_experiments)

            # Save individual investigation
            safe_id = neuron_id.replace("/", "_")
            output_path = output_dir / f"{safe_id}.json"
            scientist.save_investigation(output_path)

            results.append({
                "neuron_id": neuron_id,
                "confidence": investigation.confidence,
                "experiments": investigation.total_experiments,
                "output_file": str(output_path),
            })
            processed += 1

        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            failed += 1
            continue

    # Save partial results
    partial_file = job_dir / "partial" / f"task_{task_id}.json"
    with open(partial_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "processed": processed,
            "failed": failed,
            "results": results,
        }, f, indent=2)

    print(f"Task {task_id}: Complete. Processed {processed}, failed {failed}")


def run_worker(job_dir: Path, task_id: int):
    """Sync wrapper for worker."""
    asyncio.run(run_worker_async(job_dir, task_id))


def run_finalize(job_dir: Path):
    """Combine results from all workers."""
    print("Finalizing results...")

    with open(job_dir / "config.json") as f:
        config = json.load(f)

    n_tasks = config["n_tasks"]
    output_dir = Path(config["output_dir"])

    # Collect all results
    all_results = []
    total_processed = 0
    total_failed = 0

    for task_id in range(n_tasks):
        partial_file = job_dir / "partial" / f"task_{task_id}.json"
        if not partial_file.exists():
            print(f"Warning: Missing partial file for task {task_id}", file=sys.stderr)
            continue

        with open(partial_file) as f:
            partial = json.load(f)

        total_processed += partial["processed"]
        total_failed += partial["failed"]
        all_results.extend(partial["results"])

        print(f"Task {task_id}: {partial['processed']} processed, {partial['failed']} failed")

    # Save index
    index_file = output_dir / "investigation_index.json"
    with open(index_file, "w") as f:
        json.dump({
            "metadata": {
                "total_neurons": len(all_results),
                "processed": total_processed,
                "failed": total_failed,
                "timestamp": datetime.now().isoformat(),
            },
            "investigations": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Investigation complete!")
    print(f"  Total: {len(all_results)}")
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Index: {index_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch neuron investigation using SLURM"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--worker", action="store_true",
                           help="Run as worker (called by SLURM)")
    mode_group.add_argument("--finalize", action="store_true",
                           help="Finalize and combine results")

    # Launcher arguments
    parser.add_argument("--neurons", type=Path,
                        help="Path to interesting neurons JSON")
    parser.add_argument("--n-gpus", type=int, default=10,
                        help="Number of GPUs/workers")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/investigations"),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create job but don't submit")

    # Resource paths
    parser.add_argument("--edge-stats", type=Path, default=None,
                        help="Path to edge statistics")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to neuron labels")
    parser.add_argument("--graph-dir", type=Path, default=None,
                        help="Path to graph directory")

    # SLURM options
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition")
    parser.add_argument("--time", default="4:00:00",
                        help="Time limit per task")
    parser.add_argument("--mem", default="64G",
                        help="Memory per task")

    # Investigation options
    parser.add_argument("--max-experiments", type=int, default=100,
                        help="Max experiments per neuron")
    parser.add_argument("--llm-model", default="gpt-5",
                        help="LLM model for hypotheses")

    # Worker/finalize arguments
    parser.add_argument("--job-dir", type=Path,
                        help="Job directory")
    parser.add_argument("--task-id", type=int,
                        help="Task ID")

    # Job directory
    parser.add_argument("--job-output-dir", type=Path, default=Path("outputs/investigation_jobs"),
                        help="Directory for job files")

    args = parser.parse_args()

    # Worker mode
    if args.worker:
        if not args.job_dir or args.task_id is None:
            print("Worker mode requires --job-dir and --task-id", file=sys.stderr)
            sys.exit(1)
        run_worker(args.job_dir, args.task_id)
        return

    # Finalize mode
    if args.finalize:
        if not args.job_dir:
            print("Finalize mode requires --job-dir", file=sys.stderr)
            sys.exit(1)
        run_finalize(args.job_dir)
        return

    # Launcher mode
    if not args.neurons:
        print("Launcher mode requires --neurons", file=sys.stderr)
        sys.exit(1)

    # Load neurons
    with open(args.neurons) as f:
        data = json.load(f)
    neurons = data.get("neurons", [])

    print(f"Loaded {len(neurons)} neurons for investigation")

    # Create job
    job_dir = create_job(
        neurons=neurons,
        n_gpus=args.n_gpus,
        output_dir=args.output_dir,
        job_output_dir=args.job_output_dir,
        partition=args.partition,
        time_limit=args.time,
        mem=args.mem,
        edge_stats_path=args.edge_stats,
        labels_path=args.labels,
        graph_dir=args.graph_dir,
        max_experiments=args.max_experiments,
        llm_model=args.llm_model,
    )

    print(f"\nCreated job: {job_dir}")
    print(f"  Neurons: {len(neurons)}")
    print(f"  Workers: {args.n_gpus}")
    print(f"  Neurons/worker: ~{len(neurons) // args.n_gpus}")

    if args.dry_run:
        print("\nDry run - job not submitted")
        print(f"To submit manually: sbatch {job_dir}/submit.sh")
        return

    array_id, finalize_id = submit_job(job_dir)
    if array_id:
        print("\nSubmitted to SLURM:")
        print(f"  Array job ID: {array_id}")
        print(f"  Finalize job ID: {finalize_id}")
        print("\nMonitor with: squeue -u $USER")


if __name__ == "__main__":
    main()

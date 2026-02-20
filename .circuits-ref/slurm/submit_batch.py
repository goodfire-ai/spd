"""Submit batch jobs to SLURM for parallel attribution graph generation.

Usage:
    # From CSV file
    python -m slurm.submit_batch --input prompts.csv --n-tasks 100

    # From YAML config
    python -m slurm.submit_batch --input batch_config.yaml --n-tasks 100

    # Resume a failed job
    python -m slurm.submit_batch --resume outputs/jobs/job_20250112_143052

    # Dry run (don't submit, just create job directory)
    python -m slurm.submit_batch --input prompts.csv --n-tasks 10 --dry-run
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .manifest import JobManifest

# Default SLURM settings
DEFAULT_PARTITION = "h200-reserved"
DEFAULT_TIME = "4:00:00"
DEFAULT_MEM = "64G"
DEFAULT_CPUS = 8
DEFAULT_OUTPUT_DIR = Path("outputs/jobs")


def load_prompts_from_csv(path: Path) -> list[dict[str, str]]:
    """Load prompts from a CSV file.

    Expected columns: prompt, answer_prefix (optional)
    """
    prompts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "prompt" not in row:
                raise ValueError("CSV must have a 'prompt' column")
            prompts.append({
                "prompt": row["prompt"],
                "answer_prefix": row.get("answer_prefix", ""),
            })
    return prompts


def load_prompts_from_yaml(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load prompts and config from a YAML file.

    Expected format:
        config:
          k: 5
          tau: 0.005
          ...
        sequences:
          - prompt: "What is the capital of France?"
          - prompt: "The Eiffel Tower is in"
            answer_prefix: " Paris"
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    config = data.get("config", {})
    sequences = data.get("sequences", [])

    if not sequences:
        raise ValueError("YAML must have a 'sequences' list")

    prompts = []
    for seq in sequences:
        if "prompt" not in seq:
            raise ValueError("Each sequence must have a 'prompt' field")
        prompts.append({
            "prompt": seq["prompt"],
            "answer_prefix": seq.get("answer_prefix", ""),
        })

    return prompts, config


def load_config_file(path: Path | None) -> dict[str, Any]:
    """Load pipeline configuration from a YAML file."""
    if path is None:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("config", data)


def generate_slurm_script(
    job_dir: Path,
    n_tasks: int,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    repo_dir: Path,
) -> str:
    """Generate SLURM array job script content."""
    return f"""#!/bin/bash
#SBATCH --job-name=attrib-graphs
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --output={job_dir}/arrays/task_%a/slurm_%j.out
#SBATCH --error={job_dir}/arrays/task_%a/slurm_%j.err

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "====================="

# Activate environment and load API keys
cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

# Load environment variables (API keys, etc.)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Run worker
.venv/bin/python -m slurm.worker \\
    --job-dir {job_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "=== Job Complete ==="
echo "End time: $(date)"
"""


def generate_aggregator_script(
    job_dir: Path,
    partition: str,
    repo_dir: Path,
) -> str:
    """Generate SLURM script for aggregation job (runs after array tasks complete)."""
    return f"""#!/bin/bash
#SBATCH --job-name=attrib-aggregate
#SBATCH --partition={partition}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output={job_dir}/merged/aggregator_%j.out
#SBATCH --error={job_dir}/merged/aggregator_%j.err

echo "=== Aggregator Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "==========================="

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

# Run aggregator
.venv/bin/python -m slurm.aggregator {job_dir}

echo "=== Aggregation Complete ==="
echo "End time: $(date)"
"""


def create_job_directory(
    output_dir: Path,
    prompts: list[dict[str, str]],
    config: dict[str, Any],
    n_tasks: int,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
) -> tuple[Path, str]:
    """Create job directory structure and manifest.

    Returns:
        Tuple of (job_dir, job_id)
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create task output directories
    for i in range(n_tasks):
        (job_dir / "arrays" / f"task_{i}").mkdir(parents=True, exist_ok=True)

    # Create merged output directory
    (job_dir / "merged").mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = JobManifest(job_dir)
    manifest.create(prompts, config, n_tasks, job_id)

    # Save config separately for easy access
    config_path = job_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"config": config}, f, default_flow_style=False)

    # Save SLURM scripts
    repo_dir = Path(__file__).parent.parent.resolve()

    # Main array job script
    script_content = generate_slurm_script(
        job_dir=job_dir,
        n_tasks=n_tasks,
        partition=partition,
        time=time,
        mem=mem,
        cpus=cpus,
        repo_dir=repo_dir,
    )
    script_path = job_dir / "submit.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    # Aggregator job script (runs after array tasks complete)
    aggregator_content = generate_aggregator_script(
        job_dir=job_dir,
        partition=partition,
        repo_dir=repo_dir,
    )
    aggregator_path = job_dir / "aggregate.sh"
    with open(aggregator_path, "w") as f:
        f.write(aggregator_content)

    return job_dir, job_id


def submit_slurm_job(job_dir: Path, skip_aggregator: bool = False) -> tuple[int | None, int | None]:
    """Submit the SLURM job and return the job IDs.

    Returns:
        Tuple of (array_job_id, aggregator_job_id). Either may be None on failure.
    """
    script_path = job_dir / "submit.sh"
    aggregator_path = job_dir / "aggregate.sh"

    try:
        # Submit main array job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" not in output:
            print(f"Unexpected sbatch output: {output}", file=sys.stderr)
            return None, None

        array_job_id = int(output.split()[-1])

        # Update manifest with SLURM job ID
        manifest = JobManifest(job_dir)
        manifest.mark_job_status("submitted")

        # Submit aggregator job with dependency on array job completion
        aggregator_job_id = None
        if not skip_aggregator and aggregator_path.exists():
            try:
                agg_result = subprocess.run(
                    ["sbatch", f"--dependency=afterany:{array_job_id}", str(aggregator_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                agg_output = agg_result.stdout.strip()
                if "Submitted batch job" in agg_output:
                    aggregator_job_id = int(agg_output.split()[-1])
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to submit aggregator job: {e.stderr}", file=sys.stderr)
                print("You can run aggregation manually after job completes.", file=sys.stderr)

        return array_job_id, aggregator_job_id

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
        print("sbatch command not found. Is SLURM available?", file=sys.stderr)
        return None, None


def resume_job(job_dir: Path) -> tuple[int | None, int | None]:
    """Resume a failed/incomplete job by resubmitting.

    Returns:
        Tuple of (array_job_id, aggregator_job_id). Either may be None on failure.
    """
    manifest = JobManifest(job_dir)
    if not manifest.exists():
        print(f"No manifest found in {job_dir}", file=sys.stderr)
        return None, None

    progress = manifest.get_progress()
    pending = progress["pending"] + progress["failed"]

    if pending == 0:
        print("Job already complete, nothing to resume")
        return None, None

    print(f"Resuming job: {progress['completed']}/{progress['total']} completed")
    print(f"Will retry: {pending} prompts ({progress['pending']} pending, {progress['failed']} failed)")

    # Reset failed prompts to pending so they get retried
    manifest._atomic_update(lambda data: [
        p.update({"status": "pending", "error": None})
        for p in data["prompts"]
        if p["status"] == "failed"
    ])

    return submit_slurm_job(job_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch attribution graph generation jobs to SLURM"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Input file (CSV with prompts or YAML with config+sequences)",
    )
    input_group.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing job directory",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Pipeline config YAML file (overrides config in input YAML)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=100,
        help="Number of SLURM array tasks (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for job files (default: {DEFAULT_OUTPUT_DIR})",
    )

    # SLURM options
    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help=f"SLURM partition (default: {DEFAULT_PARTITION})",
    )
    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help=f"Time limit (default: {DEFAULT_TIME})",
    )
    parser.add_argument(
        "--mem",
        default=DEFAULT_MEM,
        help=f"Memory per task (default: {DEFAULT_MEM})",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_CPUS,
        help=f"CPUs per task (default: {DEFAULT_CPUS})",
    )

    # LLM options
    parser.add_argument(
        "--llm-rate-limit",
        type=int,
        default=3000,
        help="Global LLM API rate limit (requests/min) shared across all workers (default: 3000)",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=64,
        help="Max concurrent LLM requests per worker (default: 64)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create job directory but don't submit to SLURM",
    )
    parser.add_argument(
        "--no-auto-aggregate",
        action="store_true",
        help="Don't automatically run aggregator after job completes",
    )

    args = parser.parse_args()

    # Handle resume
    if args.resume:
        array_id, agg_id = resume_job(args.resume)
        if array_id:
            print(f"Job resubmitted: SLURM array job ID {array_id}")
            if agg_id:
                print(f"Aggregator job ID: {agg_id} (runs after array completes)")
        return

    # Load prompts
    input_path = args.input
    if input_path.suffix == ".csv":
        prompts = load_prompts_from_csv(input_path)
        config = load_config_file(args.config) if args.config else {}
    elif input_path.suffix in (".yaml", ".yml"):
        prompts, yaml_config = load_prompts_from_yaml(input_path)
        # CLI config overrides YAML config
        config = yaml_config
        if args.config:
            config.update(load_config_file(args.config))
    else:
        print(f"Unsupported input format: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Validate
    if not prompts:
        print("No prompts found in input file", file=sys.stderr)
        sys.exit(1)

    n_tasks = min(args.n_tasks, len(prompts))  # Don't create more tasks than prompts

    # Add LLM settings to config
    config["llm_rate_limit"] = args.llm_rate_limit
    config["llm_concurrency"] = args.llm_concurrency

    print(f"Loaded {len(prompts)} prompts")
    print(f"Creating job with {n_tasks} SLURM tasks")
    print(f"Prompts per task: ~{len(prompts) // n_tasks}")
    print(f"LLM rate limit: {args.llm_rate_limit}/min global, {args.llm_rate_limit // n_tasks}/min per worker")

    # Create job directory
    job_dir, job_id = create_job_directory(
        output_dir=args.output_dir,
        prompts=prompts,
        config=config,
        n_tasks=n_tasks,
        partition=args.partition,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
    )

    print(f"Created job: {job_id}")
    print(f"Job directory: {job_dir}")

    if args.dry_run:
        print("Dry run: job not submitted")
        print(f"To submit manually: sbatch {job_dir}/submit.sh")
        print(f"To run aggregator after: sbatch --dependency=afterany:<JOB_ID> {job_dir}/aggregate.sh")
        return

    # Submit to SLURM
    array_id, agg_id = submit_slurm_job(job_dir, skip_aggregator=args.no_auto_aggregate)
    if array_id:
        print(f"Submitted to SLURM: array job ID {array_id}")
        if agg_id:
            print(f"Aggregator job ID: {agg_id} (runs automatically after array completes)")
        elif not args.no_auto_aggregate:
            print(f"Run aggregator manually after completion: python -m slurm.aggregator {job_dir}")
        print(f"Monitor progress: python -m slurm.monitor {job_dir}")
    else:
        print("Failed to submit job to SLURM")
        print(f"Submit manually: sbatch {job_dir}/submit.sh")


if __name__ == "__main__":
    main()

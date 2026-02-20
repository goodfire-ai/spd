#!/usr/bin/env python3
"""Build SQLite index mapping neurons to RelP graphs.

This script indexes ~40k RelP graph files to enable fast lookup of which
graphs contain a specific neuron.

Usage:
    # Single-threaded build (for smaller datasets or testing)
    python scripts/build_neuron_index.py \
        --graphs-dir graphs/fabric_fineweb_50k \
        -o data/neuron_graph_index.db

    # Parallel build with SLURM array job
    python scripts/build_neuron_index.py \
        --graphs-dir graphs/fabric_fineweb_50k \
        --parallel --n-tasks 20 \
        -o data/neuron_graph_index.db

    # Worker mode (called by SLURM, not directly)
    python scripts/build_neuron_index.py \
        --worker --task-id 0 --total-tasks 20 \
        --graphs-dir graphs/fabric_fineweb_50k \
        --output-dir outputs/index_job/partial

    # Merge partial results
    python scripts/build_neuron_index.py \
        --merge --partial-dir outputs/index_job/partial \
        -o data/neuron_graph_index.db
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron_scientist.graph_index import GraphIndexDB


def parse_graph_file(
    graph_path: Path,
    graphs_base_dir: Path,
) -> tuple[str, list[tuple[int, int, float, list[int]]], dict[str, Any]]:
    """Parse a single graph file and extract neuron information.

    Args:
        graph_path: Path to graph JSON file
        graphs_base_dir: Base directory for relative path calculation

    Returns:
        (relative_path, neuron_entries, metadata)
        where neuron_entries is list of (layer, neuron_idx, max_influence, ctx_positions)
    """
    with open(graph_path) as f:
        data = json.load(f)

    # Calculate relative path from graphs base dir
    rel_path = str(graph_path.relative_to(graphs_base_dir))

    # Group MLP neurons by (layer, neuron_idx)
    neuron_data: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"influences": [], "positions": []}
    )

    for node in data.get("nodes", []):
        if node.get("feature_type") == "mlp_neuron":
            layer = node["layer"]
            neuron_idx = node["feature"]
            influence = node.get("influence", 0.0)
            ctx_idx = node.get("ctx_idx", 0)

            key = (layer, neuron_idx)
            neuron_data[key]["influences"].append(influence or 0.0)
            neuron_data[key]["positions"].append(ctx_idx)

    # Build entries: (layer, neuron_idx, max_influence, ctx_positions)
    entries = []
    for (layer, neuron_idx), info in neuron_data.items():
        max_influence = max(info["influences"]) if info["influences"] else 0.0
        positions = sorted(set(info["positions"]))
        entries.append((layer, neuron_idx, max_influence, positions))

    # Extract metadata
    meta = data.get("metadata", {})
    metadata = {
        "num_nodes": len(data.get("nodes", [])),
        "num_mlp_neurons": len(entries),
        "prompt_preview": (meta.get("original_prompt") or meta.get("prompt", ""))[:200],
        "source": meta.get("source", ""),
    }

    return rel_path, entries, metadata


def build_index_single(
    graphs_dir: Path,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """Build index in single-threaded mode."""
    graphs_dir = graphs_dir.resolve()
    output_path = output_path.resolve()

    # Find all graph files
    graph_files = sorted(graphs_dir.glob("*.json"))
    total = len(graph_files)

    if verbose:
        print(f"Found {total} graph files in {graphs_dir}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    db = GraphIndexDB(output_path)
    db.create_schema()

    # Process files
    start_time = time.time()
    batch_records = []
    batch_size = 10000

    with db.connection() as conn:
        for i, graph_path in enumerate(graph_files):
            try:
                rel_path, entries, metadata = parse_graph_file(graph_path, graphs_dir)

                # Add neuron entries
                for layer, neuron_idx, influence, positions in entries:
                    batch_records.append((
                        layer,
                        neuron_idx,
                        rel_path,
                        influence,
                        json.dumps(positions),
                    ))

                # Add metadata
                db.insert_graph_metadata(
                    rel_path,
                    metadata["num_nodes"],
                    metadata["num_mlp_neurons"],
                    metadata["prompt_preview"],
                    metadata["source"],
                    conn=conn,
                )

                # Batch insert
                if len(batch_records) >= batch_size:
                    db.insert_batch(batch_records, conn=conn)
                    conn.commit()
                    batch_records = []

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    print(f"  Processed {i+1}/{total} ({rate:.1f}/s, ETA: {eta:.0f}s)")

            except Exception as e:
                print(f"Error processing {graph_path}: {e}", file=sys.stderr)
                continue

        # Insert remaining records
        if batch_records:
            db.insert_batch(batch_records, conn=conn)
            conn.commit()

    # Create indexes
    if verbose:
        print("Creating indexes...")
    db.create_indexes()

    # Record build info
    db.set_build_info("build_time", datetime.now().isoformat())
    db.set_build_info("graphs_dir", str(graphs_dir))
    db.set_build_info("total_graphs", str(total))

    elapsed = time.time() - start_time
    if verbose:
        print(f"Done! Built index in {elapsed:.1f}s")
        print(f"  Total graphs: {db.get_total_graphs()}")
        print(f"  Total entries: {db.get_total_entries()}")
        print(f"  Unique neurons: {db.get_unique_neurons()}")
        print(f"  Output: {output_path}")


def worker_mode(
    task_id: int,
    total_tasks: int,
    graphs_dir: Path,
    output_dir: Path,
) -> None:
    """Process a subset of graphs as a SLURM worker."""
    graphs_dir = graphs_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all graph files and select this worker's subset
    graph_files = sorted(graphs_dir.glob("*.json"))
    total = len(graph_files)

    # Calculate this worker's slice
    per_task = total // total_tasks
    start_idx = task_id * per_task
    end_idx = start_idx + per_task if task_id < total_tasks - 1 else total

    my_files = graph_files[start_idx:end_idx]
    print(f"Worker {task_id}/{total_tasks}: processing {len(my_files)} files ({start_idx}-{end_idx})")

    # Create partial database
    partial_db_path = output_dir / f"partial_{task_id:04d}.db"
    db = GraphIndexDB(partial_db_path)
    db.create_schema()

    # Process files
    start_time = time.time()
    batch_records = []
    batch_size = 5000

    with db.connection() as conn:
        for i, graph_path in enumerate(my_files):
            try:
                rel_path, entries, metadata = parse_graph_file(graph_path, graphs_dir)

                for layer, neuron_idx, influence, positions in entries:
                    batch_records.append((
                        layer,
                        neuron_idx,
                        rel_path,
                        influence,
                        json.dumps(positions),
                    ))

                db.insert_graph_metadata(
                    rel_path,
                    metadata["num_nodes"],
                    metadata["num_mlp_neurons"],
                    metadata["prompt_preview"],
                    metadata["source"],
                    conn=conn,
                )

                if len(batch_records) >= batch_size:
                    db.insert_batch(batch_records, conn=conn)
                    conn.commit()
                    batch_records = []

                if (i + 1) % 500 == 0:
                    print(f"  Worker {task_id}: {i+1}/{len(my_files)}")

            except Exception as e:
                print(f"Worker {task_id} error on {graph_path}: {e}", file=sys.stderr)

        if batch_records:
            db.insert_batch(batch_records, conn=conn)
            conn.commit()

    elapsed = time.time() - start_time
    print(f"Worker {task_id} done: {len(my_files)} files in {elapsed:.1f}s")


def merge_partials(
    partial_dir: Path,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """Merge partial SQLite databases into final database."""
    partial_dir = partial_dir.resolve()
    output_path = output_path.resolve()

    # Find partial databases
    partial_files = sorted(partial_dir.glob("partial_*.db"))
    if not partial_files:
        print(f"No partial files found in {partial_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Merging {len(partial_files)} partial databases...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output if present
    if output_path.exists():
        output_path.unlink()

    # Initialize output database
    db = GraphIndexDB(output_path)
    db.create_schema()

    start_time = time.time()
    total_entries = 0
    total_graphs = 0

    with db.connection() as out_conn:
        for i, partial_path in enumerate(partial_files):
            if verbose:
                print(f"  Merging {partial_path.name}...")

            # Attach partial database
            out_conn.execute(f"ATTACH DATABASE '{partial_path}' AS partial")

            # Copy neuron_graph_index entries
            out_conn.execute("""
                INSERT INTO neuron_graph_index
                    (layer, neuron_idx, graph_path, influence_score, ctx_positions)
                SELECT layer, neuron_idx, graph_path, influence_score, ctx_positions
                FROM partial.neuron_graph_index
            """)

            # Copy graph_metadata entries
            out_conn.execute("""
                INSERT OR IGNORE INTO graph_metadata
                    (graph_path, num_nodes, num_mlp_neurons, prompt_preview, source)
                SELECT graph_path, num_nodes, num_mlp_neurons, prompt_preview, source
                FROM partial.graph_metadata
            """)

            # Get counts for progress
            row = out_conn.execute(
                "SELECT COUNT(*) FROM partial.neuron_graph_index"
            ).fetchone()
            total_entries += row[0]

            row = out_conn.execute(
                "SELECT COUNT(*) FROM partial.graph_metadata"
            ).fetchone()
            total_graphs += row[0]

            out_conn.execute("DETACH DATABASE partial")
            out_conn.commit()

    # Create indexes
    if verbose:
        print("Creating indexes...")
    db.create_indexes()

    # Record build info
    db.set_build_info("build_time", datetime.now().isoformat())
    db.set_build_info("merged_from", str(len(partial_files)))

    elapsed = time.time() - start_time
    if verbose:
        print(f"Done! Merged in {elapsed:.1f}s")
        print(f"  Total graphs: {db.get_total_graphs()}")
        print(f"  Total entries: {db.get_total_entries()}")
        print(f"  Unique neurons: {db.get_unique_neurons()}")
        print(f"  Output: {output_path}")


def generate_slurm_script(
    job_dir: Path,
    n_tasks: int,
    graphs_dir: Path,
    repo_dir: Path,
    partition: str = "h200-reserved",
    time_limit: str = "2:00:00",
) -> str:
    """Generate SLURM array job script."""
    return f"""#!/bin/bash
#SBATCH --job-name=neuron-index
#SBATCH --partition={partition}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --time={time_limit}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output={job_dir}/logs/worker_%a.out
#SBATCH --error={job_dir}/logs/worker_%a.err

cd {repo_dir}

# Activate environment
source .venv/bin/activate

# Run worker
python scripts/build_neuron_index.py \\
    --worker \\
    --task-id $SLURM_ARRAY_TASK_ID \\
    --total-tasks {n_tasks} \\
    --graphs-dir {graphs_dir} \\
    --output-dir {job_dir}/partial

echo "Worker $SLURM_ARRAY_TASK_ID done"
"""


def generate_merge_script(
    job_dir: Path,
    output_path: Path,
    repo_dir: Path,
    partition: str = "h200-reserved",
) -> str:
    """Generate SLURM merge script."""
    return f"""#!/bin/bash
#SBATCH --job-name=neuron-index-merge
#SBATCH --partition={partition}
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output={job_dir}/logs/merge.out
#SBATCH --error={job_dir}/logs/merge.err

cd {repo_dir}

# Activate environment
source .venv/bin/activate

# Run merge
python scripts/build_neuron_index.py \\
    --merge \\
    --partial-dir {job_dir}/partial \\
    -o {output_path}

echo "Merge done"
"""


def launch_parallel(
    graphs_dir: Path,
    output_path: Path,
    n_tasks: int,
    partition: str,
    dry_run: bool,
) -> None:
    """Launch parallel SLURM array job."""
    repo_dir = Path(__file__).parent.parent.resolve()
    graphs_dir = graphs_dir.resolve()
    output_path = output_path.resolve()

    # Create job directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = repo_dir / "outputs" / "index_jobs" / f"job_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir()
    (job_dir / "partial").mkdir()

    # Write config
    config = {
        "graphs_dir": str(graphs_dir),
        "output_path": str(output_path),
        "n_tasks": n_tasks,
        "partition": partition,
        "created": datetime.now().isoformat(),
    }
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate scripts
    submit_script = generate_slurm_script(
        job_dir, n_tasks, graphs_dir, repo_dir, partition
    )
    merge_script = generate_merge_script(job_dir, output_path, repo_dir, partition)

    submit_path = job_dir / "submit.sh"
    merge_path = job_dir / "merge.sh"

    with open(submit_path, "w") as f:
        f.write(submit_script)
    with open(merge_path, "w") as f:
        f.write(merge_script)

    print(f"Job directory: {job_dir}")
    print(f"Submit script: {submit_path}")
    print(f"Merge script: {merge_path}")

    if dry_run:
        print("\nDry run - not submitting. To submit manually:")
        print(f"  sbatch {submit_path}")
        print("  # After array job completes:")
        print(f"  sbatch {merge_path}")
        return

    # Submit array job
    print("\nSubmitting array job...")
    result = subprocess.run(
        ["sbatch", str(submit_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Extract job ID
    job_id = result.stdout.strip().split()[-1]
    print(f"Array job submitted: {job_id}")

    # Submit merge job with dependency
    print("\nSubmitting merge job (will wait for array job)...")
    result = subprocess.run(
        ["sbatch", f"--dependency=afterok:{job_id}", str(merge_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build SQLite index mapping neurons to RelP graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--worker",
        action="store_true",
        help="Run as SLURM worker (internal use)",
    )
    mode_group.add_argument(
        "--merge",
        action="store_true",
        help="Merge partial databases",
    )
    mode_group.add_argument(
        "--parallel",
        action="store_true",
        help="Launch parallel SLURM array job",
    )

    # Common arguments
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("graphs/fabric_fineweb_50k"),
        help="Directory containing graph JSON files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/neuron_graph_index.db"),
        help="Output database path",
    )

    # Worker mode arguments
    parser.add_argument("--task-id", type=int, help="SLURM array task ID")
    parser.add_argument("--total-tasks", type=int, help="Total number of tasks")
    parser.add_argument("--output-dir", type=Path, help="Output directory for partial results")

    # Merge mode arguments
    parser.add_argument("--partial-dir", type=Path, help="Directory with partial databases")

    # Parallel mode arguments
    parser.add_argument("--n-tasks", type=int, default=20, help="Number of parallel tasks")
    parser.add_argument("--partition", default="h200-reserved", help="SLURM partition")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but don't submit")

    args = parser.parse_args()

    if args.worker:
        if args.task_id is None or args.total_tasks is None or args.output_dir is None:
            parser.error("--worker requires --task-id, --total-tasks, and --output-dir")
        worker_mode(args.task_id, args.total_tasks, args.graphs_dir, args.output_dir)

    elif args.merge:
        if args.partial_dir is None:
            parser.error("--merge requires --partial-dir")
        merge_partials(args.partial_dir, args.output)

    elif args.parallel:
        launch_parallel(
            args.graphs_dir,
            args.output,
            args.n_tasks,
            args.partition,
            args.dry_run,
        )

    else:
        # Single-threaded mode
        build_index_single(args.graphs_dir, args.output)


if __name__ == "__main__":
    main()

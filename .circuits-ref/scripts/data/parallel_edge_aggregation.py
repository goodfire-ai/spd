#!/usr/bin/env python3
"""Parallel edge statistics aggregation using SLURM array jobs.

Splits prompts across multiple GPUs to efficiently aggregate edge statistics.

Usage:
    # Launch with 10 GPUs
    python scripts/parallel_edge_aggregation.py \
        --prompts data/medical_corpus_1000.yaml \
        --n-gpus 10 \
        -o data/medical_edge_stats.json

    # Also add FineWeb baseline samples
    python scripts/parallel_edge_aggregation.py \
        --prompts data/medical_corpus_1000.yaml \
        --fineweb-samples 500 \
        --n-gpus 20 \
        -o data/medical_edge_stats.json

    # Dry run (create job but don't submit)
    python scripts/parallel_edge_aggregation.py \
        --prompts data/medical_corpus_1000.yaml \
        --n-gpus 10 \
        --dry-run

    # Worker mode (called by SLURM, not directly)
    python scripts/parallel_edge_aggregation.py --worker --job-dir <path> --task-id <N>

    # Merge results (called by SLURM, or manually after job completes)
    python scripts/parallel_edge_aggregation.py --merge --job-dir <path>
"""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment from .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# =============================================================================
# Data structures for edge aggregation
# =============================================================================

def normalize_node_id(node_id: str) -> str:
    """Normalize a node_id by stripping position, keeping only layer and neuron.

    This allows aggregation across positions since RelP tracks attribution
    across positions (attention moves vectors between positions).

    Examples:
        "15_1816_26" -> "15_1816" (MLP neuron)
        "E_128000_0" -> "E_128000" (embedding)
        "L_12345_10" -> "L_12345" (logit)

    Returns the original node_id if format is unexpected.
    """
    parts = node_id.split("_")
    if len(parts) >= 2:
        # Keep layer/type and neuron/token, drop position
        return f"{parts[0]}_{parts[1]}"
    return node_id


@dataclass
class NeuronEdgeProfile:
    """Aggregated edge statistics for a single neuron."""
    layer: int
    neuron: int
    appearance_count: int = 0
    domain_appearance_count: int = 0
    baseline_appearance_count: int = 0

    # node_id -> {count, total_weight, weights}
    upstream_sources: dict[str, dict] = field(default_factory=dict)
    downstream_targets: dict[str, dict] = field(default_factory=dict)
    output_token_counts: dict[str, int] = field(default_factory=dict)

    # Co-occurrence: neuron_id -> count (neurons appearing in same graph)
    cooccurrence_counts: dict[str, int] = field(default_factory=dict)


def merge_edge_profiles(
    profiles: dict[tuple[int, int], NeuronEdgeProfile],
    other: dict[tuple[int, int], NeuronEdgeProfile],
):
    """Merge edge statistics from 'other' into 'profiles'."""
    for key, other_profile in other.items():
        if key not in profiles:
            profiles[key] = NeuronEdgeProfile(
                layer=other_profile.layer,
                neuron=other_profile.neuron,
            )

        profile = profiles[key]
        profile.appearance_count += other_profile.appearance_count
        profile.domain_appearance_count += other_profile.domain_appearance_count
        profile.baseline_appearance_count += other_profile.baseline_appearance_count

        # Merge upstream sources
        for source_id, stats in other_profile.upstream_sources.items():
            if source_id not in profile.upstream_sources:
                profile.upstream_sources[source_id] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "weights": [],
                }
            profile.upstream_sources[source_id]["count"] += stats["count"]
            profile.upstream_sources[source_id]["total_weight"] += stats["total_weight"]
            profile.upstream_sources[source_id]["weights"].extend(stats.get("weights", []))

        # Merge downstream targets
        for target_id, stats in other_profile.downstream_targets.items():
            if target_id not in profile.downstream_targets:
                profile.downstream_targets[target_id] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "weights": [],
                }
            profile.downstream_targets[target_id]["count"] += stats["count"]
            profile.downstream_targets[target_id]["total_weight"] += stats["total_weight"]
            profile.downstream_targets[target_id]["weights"].extend(stats.get("weights", []))

        # Merge output token counts
        for token, count in other_profile.output_token_counts.items():
            profile.output_token_counts[token] = (
                profile.output_token_counts.get(token, 0) + count
            )

        # Merge co-occurrence counts
        for other_id, count in other_profile.cooccurrence_counts.items():
            profile.cooccurrence_counts[other_id] = (
                profile.cooccurrence_counts.get(other_id, 0) + count
            )


def profile_to_dict(profile: NeuronEdgeProfile) -> dict:
    """Convert profile to JSON-serializable dict."""
    # Compute top upstream sources by frequency
    upstream_list = []
    for source_id, stats in profile.upstream_sources.items():
        freq = stats["count"] / profile.appearance_count if profile.appearance_count > 0 else 0
        avg_weight = stats["total_weight"] / stats["count"] if stats["count"] > 0 else 0
        upstream_list.append({
            "source": source_id,
            "count": stats["count"],
            "frequency": freq,
            "avg_weight": avg_weight,
        })
    upstream_list.sort(key=lambda x: x["frequency"], reverse=True)

    # Compute top downstream targets
    downstream_list = []
    for target_id, stats in profile.downstream_targets.items():
        freq = stats["count"] / profile.appearance_count if profile.appearance_count > 0 else 0
        avg_weight = stats["total_weight"] / stats["count"] if stats["count"] > 0 else 0
        downstream_list.append({
            "target": target_id,
            "count": stats["count"],
            "frequency": freq,
            "avg_weight": avg_weight,
        })
    downstream_list.sort(key=lambda x: x["frequency"], reverse=True)

    # Output token associations
    token_list = []
    for token, count in profile.output_token_counts.items():
        freq = count / profile.appearance_count if profile.appearance_count > 0 else 0
        token_list.append({"token": token, "count": count, "frequency": freq})
    token_list.sort(key=lambda x: x["frequency"], reverse=True)

    domain_ratio = (
        profile.domain_appearance_count / profile.appearance_count
        if profile.appearance_count > 0
        else 0
    )

    # Compute top co-occurring neurons by frequency
    cooccurrence_list = []
    for other_id, count in profile.cooccurrence_counts.items():
        freq = count / profile.appearance_count if profile.appearance_count > 0 else 0
        cooccurrence_list.append({
            "neuron_id": other_id,
            "count": count,
            "frequency": freq,
        })
    cooccurrence_list.sort(key=lambda x: x["frequency"], reverse=True)

    return {
        "layer": profile.layer,
        "neuron": profile.neuron,
        "neuron_id": f"L{profile.layer}/N{profile.neuron}",
        "appearance_count": profile.appearance_count,
        "domain_appearance_count": profile.domain_appearance_count,
        "baseline_appearance_count": profile.baseline_appearance_count,
        "domain_specificity": domain_ratio,
        "top_upstream_sources": upstream_list[:20],
        "top_downstream_targets": downstream_list[:20],
        "output_token_associations": token_list[:20],
        "top_cooccurring_neurons": cooccurrence_list[:50],  # Top 50 co-occurring neurons
    }


def dict_to_profile(d: dict) -> NeuronEdgeProfile:
    """Reconstruct profile from dict (for merging partial results)."""
    profile = NeuronEdgeProfile(
        layer=d["layer"],
        neuron=d["neuron"],
        appearance_count=d["appearance_count"],
        domain_appearance_count=d["domain_appearance_count"],
        baseline_appearance_count=d["baseline_appearance_count"],
    )

    # Reconstruct upstream sources
    for item in d.get("top_upstream_sources", []):
        source_id = item["source"]
        profile.upstream_sources[source_id] = {
            "count": item["count"],
            "total_weight": item["avg_weight"] * item["count"],
            "weights": [],
        }

    # Reconstruct downstream targets
    for item in d.get("top_downstream_targets", []):
        target_id = item["target"]
        profile.downstream_targets[target_id] = {
            "count": item["count"],
            "total_weight": item["avg_weight"] * item["count"],
            "weights": [],
        }

    # Reconstruct output tokens
    for item in d.get("output_token_associations", []):
        profile.output_token_counts[item["token"]] = item["count"]

    # Reconstruct co-occurrence counts
    for item in d.get("top_cooccurring_neurons", []):
        profile.cooccurrence_counts[item["neuron_id"]] = item["count"]

    return profile


# =============================================================================
# Prompt loading
# =============================================================================

def load_prompts_from_yaml(path: Path) -> list[dict[str, str]]:
    """Load prompts from YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    prompts = []
    for seq in data.get("sequences", []):
        prompts.append({
            "prompt": seq.get("prompt", ""),
            "answer_prefix": seq.get("answer_prefix", ""),
            "is_domain": True,
        })

    return prompts


def load_prompts_from_json(path: Path) -> list[dict[str, str]]:
    """Load prompts from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Check for global answer_prefix in metadata
    global_prefix = data.get("metadata", {}).get("answer_prefix", "")

    prompts = []
    for p in data.get("prompts", []):
        prompts.append({
            "prompt": p.get("prompt", ""),
            "answer_prefix": p.get("answer_prefix", global_prefix),
            "is_domain": p.get("is_domain", True),
        })

    return prompts


def sample_fineweb_sentences(n_samples: int, seed: int = 42) -> list[dict[str, str]]:
    """Sample random sentences from FineWeb dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets not installed, skipping FineWeb samples", file=sys.stderr)
        return []

    random.seed(seed)

    print("Loading FineWeb samples...", file=sys.stderr)

    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        buffer = []
        for i, example in enumerate(dataset):
            if i >= n_samples * 10:
                break
            text = example.get("text", "")
            if text:
                chunk = text[:500].split(".")[0].strip()
                if 20 < len(chunk) < 200:
                    buffer.append(chunk)

        sentences = random.sample(buffer, min(n_samples, len(buffer)))
        print(f"Sampled {len(sentences)} FineWeb sentences", file=sys.stderr)

        return [{"prompt": s, "answer_prefix": "", "is_domain": False} for s in sentences]

    except Exception as e:
        print(f"Warning: Could not load FineWeb: {e}", file=sys.stderr)
        return []


# =============================================================================
# SLURM job management
# =============================================================================

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
#SBATCH --job-name=edge-agg
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
#SBATCH --output={job_dir}/logs/task_%a_%j.out
#SBATCH --error={job_dir}/logs/task_%a_%j.err

echo "=== Edge Aggregation Worker ==="
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

.venv/bin/python scripts/parallel_edge_aggregation.py \\
    --worker \\
    --job-dir {job_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "=== Worker Complete ==="
echo "End time: $(date)"
"""


def generate_merge_script(job_dir: Path, repo_dir: Path, output_path: Path, partition: str) -> str:
    """Generate SLURM script for merge job."""
    return f"""#!/bin/bash
#SBATCH --job-name=edge-merge
#SBATCH --partition={partition}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output={job_dir}/logs/merge_%j.out
#SBATCH --error={job_dir}/logs/merge_%j.err

echo "=== Edge Stats Merge ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

.venv/bin/python scripts/parallel_edge_aggregation.py \\
    --merge \\
    --job-dir {job_dir} \\
    -o {output_path}

echo "=== Merge Complete ==="
echo "End time: $(date)"
"""


def create_job(
    prompts: list[dict[str, str]],
    n_gpus: int,
    output_dir: Path,
    output_path: Path,
    partition: str,
    time_limit: str,
    mem: str,
    tau: float,
    k: int,
) -> Path:
    """Create job directory and files."""
    job_id = f"edge_agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir()
    (job_dir / "partial").mkdir()

    # Save prompts with task assignments
    prompts_per_task = len(prompts) // n_gpus
    extra = len(prompts) % n_gpus

    for task_id in range(n_gpus):
        start = task_id * prompts_per_task + min(task_id, extra)
        end = start + prompts_per_task + (1 if task_id < extra else 0)
        task_prompts = prompts[start:end]

        task_file = job_dir / f"prompts_task_{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_prompts, f)

    # Save config
    config = {
        "n_tasks": n_gpus,
        "total_prompts": len(prompts),
        "tau": tau,
        "k": k,
        "output_path": str(output_path),
    }
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate SLURM scripts
    repo_dir = Path(__file__).parent.parent.resolve()

    script = generate_slurm_script(
        job_dir=job_dir,
        n_tasks=n_gpus,
        partition=partition,
        time_limit=time_limit,
        mem=mem,
        repo_dir=repo_dir,
    )
    with open(job_dir / "submit.sh", "w") as f:
        f.write(script)

    merge_script = generate_merge_script(job_dir, repo_dir, output_path, partition)
    with open(job_dir / "merge.sh", "w") as f:
        f.write(merge_script)

    return job_dir


def submit_job(job_dir: Path) -> tuple[int | None, int | None]:
    """Submit the SLURM job."""
    try:
        # Submit array job
        result = subprocess.run(
            ["sbatch", str(job_dir / "submit.sh")],
            capture_output=True,
            text=True,
            check=True,
        )
        array_job_id = int(result.stdout.strip().split()[-1])

        # Submit merge job with dependency
        result = subprocess.run(
            ["sbatch", f"--dependency=afterok:{array_job_id}", str(job_dir / "merge.sh")],
            capture_output=True,
            text=True,
            check=True,
        )
        merge_job_id = int(result.stdout.strip().split()[-1])

        return array_job_id, merge_job_id

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
        print("sbatch not found. Is SLURM available?", file=sys.stderr)
        return None, None


# =============================================================================
# Worker execution
# =============================================================================

def run_worker(job_dir: Path, task_id: int):
    """Execute worker for a single SLURM task."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Add parent for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from circuits.relp import RelPAttributor, RelPConfig

    print(f"Worker {task_id} starting...")

    # Load config
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    tau = config.get("tau", 0.005)
    k = config.get("k", 5)

    # Load prompts for this task
    prompts_file = job_dir / f"prompts_task_{task_id}.json"
    with open(prompts_file) as f:
        prompts = json.load(f)

    print(f"Task {task_id}: Processing {len(prompts)} prompts")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Initialize attributor
    relp_config = RelPConfig(k=k, tau=tau, compute_edges=True, linearize=True)
    attributor = RelPAttributor(model, tokenizer, config=relp_config)

    # Chat template
    CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer_prefix}"""

    # Process prompts and aggregate
    profiles: dict[tuple[int, int], NeuronEdgeProfile] = {}
    processed = 0
    failed = 0

    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        answer_prefix = p.get("answer_prefix", "")
        is_domain = p.get("is_domain", True)

        formatted = CHAT_TEMPLATE.format(prompt=prompt_text, answer_prefix=answer_prefix)

        try:
            graph = attributor.compute_attributions(formatted, k=k, tau=tau, compute_edges=True)
            aggregate_graph_edges(graph, profiles, is_domain)
            processed += 1

            if (i + 1) % 10 == 0:
                print(f"Task {task_id}: Processed {i+1}/{len(prompts)} prompts, {len(profiles)} neurons")

        except Exception as e:
            print(f"Task {task_id}: Error on prompt {i}: {e}", file=sys.stderr)
            failed += 1

    # Save partial results
    output = {
        "task_id": task_id,
        "processed": processed,
        "failed": failed,
        "n_neurons": len(profiles),
        "profiles": [profile_to_dict(p) for p in profiles.values()],
    }

    partial_file = job_dir / "partial" / f"task_{task_id}.json"
    with open(partial_file, "w") as f:
        json.dump(output, f)

    print(f"Task {task_id}: Complete. Processed {processed}, failed {failed}, neurons {len(profiles)}")


def aggregate_graph_edges(
    graph: dict,
    profiles: dict[tuple[int, int], NeuronEdgeProfile],
    is_domain: bool,
):
    """Aggregate edge statistics from a single graph into profiles."""
    # Build edge maps
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    logit_tokens = {}

    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        # Check for logit nodes: either isLogit=True or layer='logit'
        if node.get("isLogit") or node.get("layer") == "logit":
            clerp = node.get("clerp", "")
            # Handle multiple formats:
            # - "Answer: token (p=X)"
            # - "token (p=X)"
            # - "Logit: token (p=X)"
            if clerp.startswith("Answer:"):
                token = clerp.split("(p=")[0].replace("Answer:", "").strip()
            elif clerp.startswith("Logit:"):
                token = clerp.split("(p=")[0].replace("Logit:", "").strip()
            elif "(p=" in clerp:
                token = clerp.split("(p=")[0].strip()
            else:
                token = clerp.strip()
            if token:
                logit_tokens[node_id] = token

    for link in graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        weight = link.get("weight", 0.0)
        if source and target:
            incoming[target].append((source, weight))
            outgoing[source].append((target, weight))

    # First pass: collect all MLP neurons in this graph (for co-occurrence tracking)
    neurons_in_graph: set[tuple[int, int]] = set()
    for node in graph.get("nodes", []):
        layer = node.get("layer")
        if layer in ("E", "32", None):
            continue
        try:
            layer_int = int(layer)
            if layer_int < 0 or layer_int > 31:
                continue
            neuron = node.get("feature")
            if neuron is not None:
                neurons_in_graph.add((layer_int, neuron))
        except ValueError:
            continue

    # Second pass: process MLP neurons
    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        layer = node.get("layer")

        if layer in ("E", "32", None):
            continue

        try:
            layer_int = int(layer)
            if layer_int < 0 or layer_int > 31:
                continue
            neuron = node.get("feature")
            if neuron is None:
                continue
        except ValueError:
            continue

        key = (layer_int, neuron)
        if key not in profiles:
            profiles[key] = NeuronEdgeProfile(layer=layer_int, neuron=neuron)

        profile = profiles[key]
        profile.appearance_count += 1
        if is_domain:
            profile.domain_appearance_count += 1
        else:
            profile.baseline_appearance_count += 1

        # Update co-occurrence counts with all other neurons in this graph
        for other_key in neurons_in_graph:
            if other_key != key:  # Don't count self
                other_layer, other_neuron = other_key
                other_id = f"L{other_layer}/N{other_neuron}"
                profile.cooccurrence_counts[other_id] = (
                    profile.cooccurrence_counts.get(other_id, 0) + 1
                )

        # Upstream sources (normalized to remove position)
        for source_id, weight in incoming.get(node_id, []):
            # Normalize to aggregate across positions
            norm_source = normalize_node_id(source_id)
            if norm_source not in profile.upstream_sources:
                profile.upstream_sources[norm_source] = {
                    "count": 0,
                    "total_weight": 0.0,
                }
            profile.upstream_sources[norm_source]["count"] += 1
            profile.upstream_sources[norm_source]["total_weight"] += weight

        # Downstream targets (normalized to remove position)
        for target_id, weight in outgoing.get(node_id, []):
            # Normalize to aggregate across positions
            norm_target = normalize_node_id(target_id)
            if norm_target not in profile.downstream_targets:
                profile.downstream_targets[norm_target] = {
                    "count": 0,
                    "total_weight": 0.0,
                }
            profile.downstream_targets[norm_target]["count"] += 1
            profile.downstream_targets[norm_target]["total_weight"] += weight

            if target_id in logit_tokens:
                token = logit_tokens[target_id]
                profile.output_token_counts[token] = (
                    profile.output_token_counts.get(token, 0) + 1
                )


# =============================================================================
# Merge results
# =============================================================================

def run_merge(job_dir: Path, output_path: Path):
    """Merge partial results from all workers."""
    print("Merging partial results...")

    # Load config
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    n_tasks = config["n_tasks"]
    total_prompts = config["total_prompts"]

    # Load and merge all partial results
    profiles: dict[tuple[int, int], NeuronEdgeProfile] = {}
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

        # Reconstruct profiles and merge
        for p_dict in partial["profiles"]:
            p = dict_to_profile(p_dict)
            key = (p.layer, p.neuron)
            if key not in profiles:
                profiles[key] = NeuronEdgeProfile(layer=p.layer, neuron=p.neuron)
            merge_edge_profiles({key: profiles[key]}, {key: p})

        print(f"Merged task {task_id}: {partial['processed']} prompts, {partial['n_neurons']} neurons")

    # Compute summary statistics
    domain_specific = sum(
        1 for p in profiles.values()
        if p.domain_appearance_count / p.appearance_count > 0.7
        if p.appearance_count > 0
    )
    baseline_specific = sum(
        1 for p in profiles.values()
        if p.domain_appearance_count / p.appearance_count < 0.3
        if p.appearance_count > 0
    )
    general = len(profiles) - domain_specific - baseline_specific

    # Save final output
    output = {
        "metadata": {
            "total_prompts": total_prompts,
            "processed": total_processed,
            "failed": total_failed,
            "n_tasks": n_tasks,
            "tau": config["tau"],
            "k": config["k"],
        },
        "summary": {
            "total_neurons": len(profiles),
            "domain_specific_neurons": domain_specific,
            "baseline_specific_neurons": baseline_specific,
            "general_neurons": general,
        },
        "profiles": sorted(
            [profile_to_dict(p) for p in profiles.values()],
            key=lambda x: x["appearance_count"],
            reverse=True,
        ),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"  Processed: {total_processed}/{total_prompts}")
    print(f"  Failed: {total_failed}")
    print(f"  Total neurons: {len(profiles)}")
    print(f"  Domain-specific: {domain_specific}")
    print(f"  Baseline-specific: {baseline_specific}")
    print(f"  General: {general}")
    print(f"  Output: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel edge statistics aggregation using SLURM"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--worker", action="store_true",
                           help="Run as worker (called by SLURM)")
    mode_group.add_argument("--merge", action="store_true",
                           help="Merge partial results")

    # Launcher arguments
    parser.add_argument("--prompts", type=Path,
                        help="YAML or JSON file with prompts")
    parser.add_argument("--fineweb-samples", type=int, default=0,
                        help="Number of FineWeb baseline samples")
    parser.add_argument("--n-gpus", type=int, default=10,
                        help="Number of GPUs to use (default: 10)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output JSON file for merged results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create job but don't submit")

    # SLURM options
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition")
    parser.add_argument("--time", default="4:00:00",
                        help="Time limit per task")
    parser.add_argument("--mem", default="64G",
                        help="Memory per task")

    # Graph generation options
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Node threshold")
    parser.add_argument("--k", type=int, default=5,
                        help="Top logits to trace")

    # Worker/merge arguments
    parser.add_argument("--job-dir", type=Path,
                        help="Job directory (for worker/merge modes)")
    parser.add_argument("--task-id", type=int,
                        help="Task ID (for worker mode)")

    # Output directory
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/edge_jobs"),
                        help="Directory for job files")

    args = parser.parse_args()

    # Worker mode
    if args.worker:
        if not args.job_dir or args.task_id is None:
            print("Worker mode requires --job-dir and --task-id", file=sys.stderr)
            sys.exit(1)
        run_worker(args.job_dir, args.task_id)
        return

    # Merge mode
    if args.merge:
        if not args.job_dir:
            print("Merge mode requires --job-dir", file=sys.stderr)
            sys.exit(1)
        output_path = args.output
        if not output_path:
            with open(args.job_dir / "config.json") as f:
                config = json.load(f)
            output_path = Path(config["output_path"])
        run_merge(args.job_dir, output_path)
        return

    # Launcher mode
    if not args.prompts:
        print("Launcher mode requires --prompts", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Launcher mode requires --output", file=sys.stderr)
        sys.exit(1)

    # Load prompts
    if args.prompts.suffix == ".yaml":
        prompts = load_prompts_from_yaml(args.prompts)
    else:
        prompts = load_prompts_from_json(args.prompts)

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Add FineWeb samples
    if args.fineweb_samples > 0:
        fineweb = sample_fineweb_sentences(args.fineweb_samples)
        prompts.extend(fineweb)
        print(f"Added {len(fineweb)} FineWeb baseline samples")

    # Shuffle for better load balancing
    random.shuffle(prompts)

    # Create job
    job_dir = create_job(
        prompts=prompts,
        n_gpus=args.n_gpus,
        output_dir=args.output_dir,
        output_path=args.output.resolve(),
        partition=args.partition,
        time_limit=args.time,
        mem=args.mem,
        tau=args.tau,
        k=args.k,
    )

    print(f"\nCreated job: {job_dir}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Prompts per GPU: ~{len(prompts) // args.n_gpus}")

    if args.dry_run:
        print("\nDry run - job not submitted")
        print("To submit manually:")
        print(f"  sbatch {job_dir}/submit.sh")
        print("  # Then after completion:")
        print(f"  sbatch {job_dir}/merge.sh")
        return

    # Submit job
    array_id, merge_id = submit_job(job_dir)
    if array_id:
        print("\nSubmitted to SLURM:")
        print(f"  Array job ID: {array_id}")
        print(f"  Merge job ID: {merge_id} (runs after array completes)")
        print("\nMonitor with: squeue -u $USER")
        print(f"Output will be saved to: {args.output}")
    else:
        print("\nFailed to submit to SLURM")
        print(f"Submit manually: sbatch {job_dir}/submit.sh")


if __name__ == "__main__":
    main()

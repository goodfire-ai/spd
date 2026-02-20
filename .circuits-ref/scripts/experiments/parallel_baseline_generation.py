#!/usr/bin/env python3
"""Parallel baseline graph generation using SLURM array jobs.

Generates attribution graphs with random target positions across multiple GPUs.
Supports FineWeb, chat datasets (WildChat, ShareGPT), or mixed mode.

Usage:
    # Launch with 10 GPUs, generating 500 graphs from FineWeb
    python scripts/parallel_baseline_generation.py \
        --n-samples 500 \
        --n-gpus 10 \
        --dataset fineweb \
        --output-dir graphs/fineweb_baseline

    # Mixed mode: 60% FineWeb, 40% Chat
    python scripts/parallel_baseline_generation.py \
        --n-samples 50000 \
        --n-gpus 80 \
        --dataset mixed \
        --output-dir graphs/fabric_v1

    # Chat only (WildChat)
    python scripts/parallel_baseline_generation.py \
        --n-samples 10000 \
        --n-gpus 20 \
        --dataset chat \
        --output-dir graphs/chat_baseline

    # Dry run (create job but don't submit)
    python scripts/parallel_baseline_generation.py \
        --n-samples 500 \
        --n-gpus 10 \
        --dry-run

    # Worker mode (called by SLURM, not directly)
    python scripts/parallel_baseline_generation.py --worker --job-dir <path> --task-id <N>

    # Finalize (called after all workers complete)
    python scripts/parallel_baseline_generation.py --finalize --job-dir <path>
"""

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment from .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# Llama 3.1 chat template
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def sample_fineweb_sentences(n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Sample random sentences from FineWeb dataset.

    Returns list of {"prompt": str, "source": "fineweb"}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

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
                # Extract first sentence-like chunk (20-200 chars)
                chunk = text[:500].split(".")[0].strip()
                if 20 < len(chunk) < 200:
                    buffer.append(chunk)

        samples = []
        if buffer:
            selected = random.sample(buffer, min(n_samples, len(buffer)))
            samples = [{"prompt": s, "source": "fineweb"} for s in selected]

        print(f"Sampled {len(samples)} FineWeb sentences", file=sys.stderr)
        return samples

    except Exception as e:
        print(f"Error loading FineWeb: {e}", file=sys.stderr)
        sys.exit(1)


def sample_chat_conversations(n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Sample conversations from chat datasets (WildChat, ShareGPT).

    Returns list of {"prompt": str, "source": "wildchat" | "sharegpt"}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    random.seed(seed)
    samples = []

    # Try WildChat first (primary)
    print("Loading WildChat samples...", file=sys.stderr)
    try:
        dataset = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
        )

        buffer = []
        for i, example in enumerate(dataset):
            if i >= n_samples * 5:
                break
            # Extract first user message from conversation
            conversation = example.get("conversation", [])
            if conversation and len(conversation) > 0:
                first_msg = conversation[0]
                if first_msg.get("role") == "user":
                    content = first_msg.get("content", "")
                    # Filter for reasonable length prompts
                    if 20 < len(content) < 500:
                        buffer.append(content)

        if buffer:
            selected = random.sample(buffer, min(n_samples, len(buffer)))
            samples.extend([{"prompt": s, "source": "wildchat"} for s in selected])
            print(f"Sampled {len(samples)} WildChat conversations", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Could not load WildChat: {e}", file=sys.stderr)

    # If we need more, try ShareGPT
    if len(samples) < n_samples:
        remaining = n_samples - len(samples)
        print(f"Loading ShareGPT samples for remaining {remaining}...", file=sys.stderr)
        try:
            dataset = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train",
                streaming=True,
            )

            buffer = []
            for i, example in enumerate(dataset):
                if i >= remaining * 5:
                    break
                conversations = example.get("conversations", [])
                if conversations and len(conversations) > 0:
                    first_msg = conversations[0]
                    if first_msg.get("from") == "human":
                        content = first_msg.get("value", "")
                        if 20 < len(content) < 500:
                            buffer.append(content)

            if buffer:
                selected = random.sample(buffer, min(remaining, len(buffer)))
                samples.extend([{"prompt": s, "source": "sharegpt"} for s in selected])
                print(f"Added {len(selected)} ShareGPT conversations", file=sys.stderr)

        except Exception as e:
            print(f"Warning: Could not load ShareGPT: {e}", file=sys.stderr)

    print(f"Total chat samples: {len(samples)}", file=sys.stderr)
    return samples


def sample_mixed_dataset(n_samples: int, seed: int = 42, fineweb_ratio: float = 0.6) -> list[dict[str, Any]]:
    """Sample from both FineWeb and chat datasets.

    Args:
        n_samples: Total number of samples
        seed: Random seed
        fineweb_ratio: Fraction from FineWeb (default 0.6 = 60%)

    Returns list of {"prompt": str, "source": str}
    """
    n_fineweb = int(n_samples * fineweb_ratio)
    n_chat = n_samples - n_fineweb

    print(f"Sampling mixed dataset: {n_fineweb} FineWeb + {n_chat} Chat", file=sys.stderr)

    fineweb_samples = sample_fineweb_sentences(n_fineweb, seed)
    chat_samples = sample_chat_conversations(n_chat, seed + 1000)  # Different seed for variety

    # Combine and shuffle
    all_samples = fineweb_samples + chat_samples
    random.seed(seed + 2000)
    random.shuffle(all_samples)

    print(f"Total mixed samples: {len(all_samples)}", file=sys.stderr)
    return all_samples


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
#SBATCH --job-name=baseline-gen
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
#SBATCH --output={job_dir}/logs/task_%a_%j.out
#SBATCH --error={job_dir}/logs/task_%a_%j.err

echo "=== Baseline Graph Generation Worker ==="
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

.venv/bin/python scripts/parallel_baseline_generation.py \\
    --worker \\
    --job-dir {job_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "=== Worker Complete ==="
echo "End time: $(date)"
"""


def generate_finalize_script(job_dir: Path, repo_dir: Path, partition: str) -> str:
    """Generate SLURM script for finalize job."""
    return f"""#!/bin/bash
#SBATCH --job-name=baseline-finalize
#SBATCH --partition={partition}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output={job_dir}/logs/finalize_%j.out
#SBATCH --error={job_dir}/logs/finalize_%j.err

echo "=== Baseline Generation Finalize ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

.venv/bin/python scripts/parallel_baseline_generation.py \\
    --finalize \\
    --job-dir {job_dir}

echo "=== Finalize Complete ==="
echo "End time: $(date)"
"""


def create_job(
    samples: list[dict[str, Any]],
    n_gpus: int,
    output_dir: Path,
    job_output_dir: Path,
    partition: str,
    time_limit: str,
    mem: str,
    tau: float,
    k: int,
    max_new_tokens: int,
    seed: int,
    dataset_type: str = "fineweb",
) -> Path:
    """Create job directory and files.

    Args:
        samples: List of {"prompt": str, "source": str}
        n_gpus: Number of GPUs/tasks
        output_dir: Where to save graphs
        job_output_dir: Where to save job metadata
        partition: SLURM partition
        time_limit: Time limit per task
        mem: Memory per task
        tau: Node threshold
        k: Top logits to trace
        max_new_tokens: Max tokens to generate
        seed: Random seed
        dataset_type: "fineweb", "chat", or "mixed"
    """
    job_id = f"fabric_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = job_output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir()
    (job_dir / "partial").mkdir()

    # Create output directory for graphs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save samples with task assignments
    samples_per_task = len(samples) // n_gpus
    extra = len(samples) % n_gpus

    for task_id in range(n_gpus):
        start = task_id * samples_per_task + min(task_id, extra)
        end = start + samples_per_task + (1 if task_id < extra else 0)
        task_samples = samples[start:end]

        # Include original indices and source info
        task_data = [
            {
                "index": start + i,
                "prompt": s["prompt"],
                "source": s.get("source", "unknown"),
            }
            for i, s in enumerate(task_samples)
        ]

        task_file = job_dir / f"samples_task_{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)

    # Save config
    config = {
        "n_tasks": n_gpus,
        "total_samples": len(samples),
        "dataset_type": dataset_type,
        "tau": tau,
        "k": k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "output_dir": str(output_dir),
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

    finalize_script = generate_finalize_script(job_dir, repo_dir, partition)
    with open(job_dir / "finalize.sh", "w") as f:
        f.write(finalize_script)

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

        # Submit finalize job with dependency
        result = subprocess.run(
            ["sbatch", f"--dependency=afterany:{array_job_id}", str(job_dir / "finalize.sh")],
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


# =============================================================================
# Worker execution
# =============================================================================

def find_assistant_response_range(
    tokenizer,
    full_text: str,
    prompt: str
) -> tuple[int, int]:
    """Find the token range of the assistant's response."""
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    full_token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Find the third <|end_header_id|> (system, user, assistant)
    header_count = 0
    assistant_start = None
    for i, tok in enumerate(full_token_strs):
        if '<|end_header_id|>' in tok or tok == '<|end_header_id|>':
            header_count += 1
            if header_count == 3:
                assistant_start = i + 1
                while assistant_start < len(tokens) and full_token_strs[assistant_start] in ['Ċ', 'ĊĊ', '\n', '\n\n']:
                    assistant_start += 1
                break

    if assistant_start is None:
        prompt_formatted = CHAT_TEMPLATE.format(prompt=prompt)
        prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=False)
        assistant_start = len(prompt_tokens)

    return assistant_start, len(tokens) - 1


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Generate a completion for a prompt."""
    import torch

    formatted = CHAT_TEMPLATE.format(prompt=prompt)
    device = next(model.parameters()).device
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_text


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
    max_new_tokens = config.get("max_new_tokens", 50)
    seed = config.get("seed", 42)
    output_dir = Path(config["output_dir"])

    # Set seed with task-specific offset for different random positions
    random.seed(seed + task_id * 1000)
    torch.manual_seed(seed + task_id * 1000)

    # Load samples for this task
    samples_file = job_dir / f"samples_task_{task_id}.json"
    with open(samples_file) as f:
        task_data = json.load(f)

    print(f"Task {task_id}: Processing {len(task_data)} samples")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # PHASE 1: Generate all completions
    print(f"Task {task_id}: Phase 1 - Generating completions...")
    completions = []
    for item in task_data:
        idx = item["index"]
        prompt = item["prompt"]
        source = item.get("source", "unknown")

        try:
            full_text = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens
            )

            start_pos, end_pos = find_assistant_response_range(
                tokenizer, full_text, prompt
            )

            if end_pos - start_pos < 3:
                print(f"  Task {task_id}: Skipping sample {idx}: response too short", file=sys.stderr)
                continue

            # Randomly select a position within the assistant's response
            target_pos = random.randint(start_pos + 1, end_pos)

            completions.append({
                "index": idx,
                "prompt": prompt,
                "source": source,
                "full_text": full_text,
                "target_pos": target_pos,
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
        except Exception as e:
            print(f"  Task {task_id}: Error generating completion {idx}: {e}", file=sys.stderr)
            continue

    print(f"Task {task_id}: Generated {len(completions)} completions")

    # Clear CUDA cache before graph generation
    torch.cuda.empty_cache()

    # Initialize attributor
    relp_config = RelPConfig(k=k, tau=tau, compute_edges=True, linearize=True)
    attributor = RelPAttributor(model, tokenizer, config=relp_config)

    # PHASE 2: Generate graphs
    print(f"Task {task_id}: Phase 2 - Generating graphs...")
    metadata_list = []
    processed = 0
    failed = 0

    # Track timing and node count stats
    generation_times = []
    node_counts = []
    edge_counts = []

    for i, comp in enumerate(completions):
        idx = comp["index"]
        try:
            # Time the graph generation
            start_time = time.time()
            graph = attributor.compute_attributions(
                comp["full_text"],
                k=k,
                tau=tau,
                compute_edges=True,
                target_position=comp["target_pos"]
            )
            gen_time = time.time() - start_time
            generation_times.append(gen_time)

            # Count nodes and edges
            num_nodes = len(graph.get("nodes", []))
            num_edges = len(graph.get("links", []))
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)

            # Add additional metadata
            graph["metadata"]["original_prompt"] = comp["prompt"]
            graph["metadata"]["source"] = comp["source"]
            graph["metadata"]["assistant_response_start"] = comp["start_pos"]
            graph["metadata"]["assistant_response_end"] = comp["end_pos"]
            graph["metadata"]["is_baseline"] = True
            graph["metadata"]["sample_index"] = idx
            graph["metadata"]["generation_time_sec"] = gen_time
            graph["metadata"]["num_nodes"] = num_nodes
            graph["metadata"]["num_edges"] = num_edges

            # Extract neurons from graph for indexing
            neurons_in_graph = []
            for node in graph.get("nodes", []):
                layer = node.get("layer")
                if layer in ("E", "32", None) or node.get("isLogit"):
                    continue
                try:
                    layer_int = int(layer)
                    neuron = node.get("feature")
                    if neuron is not None and 0 <= layer_int <= 31:
                        neuron_id = f"L{layer_int}/N{neuron}"
                        if neuron_id not in neurons_in_graph:
                            neurons_in_graph.append(neuron_id)
                except (ValueError, TypeError):
                    continue

            # Save graph
            graph_filename = f"graph_{idx:05d}.json"
            output_path = output_dir / graph_filename
            with open(output_path, "w") as f:
                json.dump(graph, f)

            metadata_list.append({
                "sample_index": idx,
                "prompt": comp["prompt"],
                "source": comp["source"],
                "target_position": comp["target_pos"],
                "assistant_start": comp["start_pos"],
                "assistant_end": comp["end_pos"],
                "graph_file": graph_filename,
                "neurons": neurons_in_graph,
                "generation_time_sec": gen_time,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
            })
            processed += 1

            if (i + 1) % 5 == 0:
                avg_time = sum(generation_times) / len(generation_times)
                avg_nodes = sum(node_counts) / len(node_counts)
                print(f"Task {task_id}: Generated {i+1}/{len(completions)} graphs | avg {avg_time:.1f}s/graph, {avg_nodes:.0f} nodes")

        except Exception as e:
            print(f"  Task {task_id}: Error processing sample {idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    # Compute task-level stats
    task_stats = {}
    if generation_times:
        task_stats = {
            "avg_generation_time_sec": sum(generation_times) / len(generation_times),
            "min_generation_time_sec": min(generation_times),
            "max_generation_time_sec": max(generation_times),
            "total_generation_time_sec": sum(generation_times),
            "avg_node_count": sum(node_counts) / len(node_counts),
            "min_node_count": min(node_counts),
            "max_node_count": max(node_counts),
            "avg_edge_count": sum(edge_counts) / len(edge_counts),
            "min_edge_count": min(edge_counts),
            "max_edge_count": max(edge_counts),
        }

    # Save partial metadata
    partial_file = job_dir / "partial" / f"task_{task_id}.json"
    with open(partial_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "processed": processed,
            "failed": failed,
            "skipped": len(task_data) - len(completions),
            "stats": task_stats,
            "metadata": metadata_list,
        }, f, indent=2)

    if task_stats:
        print(f"Task {task_id}: Complete. Processed {processed}, failed {failed}")
        print(f"  Avg time: {task_stats['avg_generation_time_sec']:.1f}s, Avg nodes: {task_stats['avg_node_count']:.0f}, Avg edges: {task_stats['avg_edge_count']:.0f}")
    else:
        print(f"Task {task_id}: Complete. Processed {processed}, failed {failed}")


# =============================================================================
# Finalize
# =============================================================================

def run_finalize(job_dir: Path):
    """Combine metadata from all workers and build neuron index."""
    print("Finalizing results...")

    # Load config
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    n_tasks = config["n_tasks"]
    output_dir = Path(config["output_dir"])
    dataset_type = config.get("dataset_type", "fineweb")

    # Collect all metadata and stats
    all_metadata = []
    total_processed = 0
    total_failed = 0
    total_skipped = 0

    # Aggregate stats across all tasks
    all_generation_times = []
    all_node_counts = []
    all_edge_counts = []

    for task_id in range(n_tasks):
        partial_file = job_dir / "partial" / f"task_{task_id}.json"
        if not partial_file.exists():
            print(f"Warning: Missing partial file for task {task_id}", file=sys.stderr)
            continue

        with open(partial_file) as f:
            partial = json.load(f)

        total_processed += partial["processed"]
        total_failed += partial["failed"]
        total_skipped += partial["skipped"]
        all_metadata.extend(partial["metadata"])

        # Collect per-sample stats
        for sample in partial["metadata"]:
            if "generation_time_sec" in sample:
                all_generation_times.append(sample["generation_time_sec"])
            if "num_nodes" in sample:
                all_node_counts.append(sample["num_nodes"])
            if "num_edges" in sample:
                all_edge_counts.append(sample["num_edges"])

        task_stats = partial.get("stats", {})
        if task_stats:
            print(f"Task {task_id}: {partial['processed']} processed, {partial['failed']} failed | "
                  f"avg {task_stats.get('avg_generation_time_sec', 0):.1f}s, "
                  f"{task_stats.get('avg_node_count', 0):.0f} nodes")
        else:
            print(f"Task {task_id}: {partial['processed']} processed, {partial['failed']} failed")

    # Sort by sample index
    all_metadata.sort(key=lambda x: x["sample_index"])

    # Build neuron index: neuron_id -> list of graph files
    print("Building neuron index...")
    neuron_index = defaultdict(list)
    source_counts = defaultdict(int)

    for sample in all_metadata:
        graph_file = sample.get("graph_file", "")
        source = sample.get("source", "unknown")
        source_counts[source] += 1

        for neuron_id in sample.get("neurons", []):
            neuron_index[neuron_id].append(graph_file)

    # Convert defaultdict to regular dict for JSON serialization
    neuron_index = dict(neuron_index)

    print(f"  Indexed {len(neuron_index)} unique neurons across {total_processed} graphs")
    print(f"  Source breakdown: {dict(source_counts)}")

    # Determine source description based on dataset type
    if dataset_type == "mixed":
        source_desc = f"Mixed: FineWeb ({source_counts.get('fineweb', 0)}) + Chat ({source_counts.get('wildchat', 0) + source_counts.get('sharegpt', 0)})"
    elif dataset_type == "chat":
        source_desc = f"Chat: WildChat ({source_counts.get('wildchat', 0)}) + ShareGPT ({source_counts.get('sharegpt', 0)})"
    else:
        source_desc = "FineWeb-edu sample-10BT"

    # Compute aggregate statistics
    aggregate_stats = {}
    if all_generation_times:
        aggregate_stats = {
            "generation_time": {
                "avg_sec": sum(all_generation_times) / len(all_generation_times),
                "min_sec": min(all_generation_times),
                "max_sec": max(all_generation_times),
                "total_sec": sum(all_generation_times),
                "total_hours": sum(all_generation_times) / 3600,
            },
            "node_count": {
                "avg": sum(all_node_counts) / len(all_node_counts) if all_node_counts else 0,
                "min": min(all_node_counts) if all_node_counts else 0,
                "max": max(all_node_counts) if all_node_counts else 0,
            },
            "edge_count": {
                "avg": sum(all_edge_counts) / len(all_edge_counts) if all_edge_counts else 0,
                "min": min(all_edge_counts) if all_edge_counts else 0,
                "max": max(all_edge_counts) if all_edge_counts else 0,
            },
        }

    # Save combined metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "metadata": {
                "source": source_desc,
                "dataset_type": dataset_type,
                "total_samples": config["total_samples"],
                "processed": total_processed,
                "failed": total_failed,
                "skipped": total_skipped,
                "source_counts": dict(source_counts),
                "seed": config["seed"],
                "tau": config["tau"],
                "k": config["k"],
                "max_new_tokens": config["max_new_tokens"],
            },
            "aggregate_stats": aggregate_stats,
            "samples": all_metadata,
        }, f, indent=2)

    # Save neuron index separately (can be large)
    neuron_index_file = output_dir / "neuron_index.json"
    with open(neuron_index_file, "w") as f:
        json.dump({
            "description": "Maps neuron IDs to list of graph files containing that neuron",
            "total_neurons": len(neuron_index),
            "total_graphs": total_processed,
            "index": neuron_index,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Total samples: {config['total_samples']}")
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Unique neurons: {len(neuron_index)}")
    if aggregate_stats:
        print("\n  === Aggregate Statistics ===")
        gen_stats = aggregate_stats.get("generation_time", {})
        node_stats = aggregate_stats.get("node_count", {})
        edge_stats = aggregate_stats.get("edge_count", {})
        print(f"  Generation time: avg {gen_stats.get('avg_sec', 0):.1f}s, "
              f"min {gen_stats.get('min_sec', 0):.1f}s, max {gen_stats.get('max_sec', 0):.1f}s")
        print(f"  Total compute time: {gen_stats.get('total_hours', 0):.2f} hours")
        print(f"  Node count: avg {node_stats.get('avg', 0):.0f}, "
              f"min {node_stats.get('min', 0)}, max {node_stats.get('max', 0)}")
        print(f"  Edge count: avg {edge_stats.get('avg', 0):.0f}, "
              f"min {edge_stats.get('min', 0)}, max {edge_stats.get('max', 0)}")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Metadata: {metadata_file}")
    print(f"  Neuron index: {neuron_index_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel baseline graph generation using SLURM"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--worker", action="store_true",
                           help="Run as worker (called by SLURM)")
    mode_group.add_argument("--finalize", action="store_true",
                           help="Finalize and combine results")

    # Launcher arguments
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples (default: 500)")
    parser.add_argument("--n-gpus", type=int, default=10,
                        help="Number of GPUs to use (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("graphs/fabric_baseline"),
                        help="Output directory for graphs")
    parser.add_argument("--dataset", type=str, default="fineweb",
                        choices=["fineweb", "chat", "mixed"],
                        help="Dataset type: fineweb, chat, or mixed (default: fineweb)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create job but don't submit")

    # SLURM options
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition")
    parser.add_argument("--time", default="2:00:00",
                        help="Time limit per task")
    parser.add_argument("--mem", default="64G",
                        help="Memory per task")

    # Graph generation options
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Node threshold")
    parser.add_argument("--k", type=int, default=5,
                        help="Top logits to trace")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Worker/finalize arguments
    parser.add_argument("--job-dir", type=Path,
                        help="Job directory (for worker/finalize modes)")
    parser.add_argument("--task-id", type=int,
                        help="Task ID (for worker mode)")

    # Job directory
    parser.add_argument("--job-output-dir", type=Path, default=Path("outputs/baseline_jobs"),
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

    # Launcher mode - sample from selected dataset
    print(f"Dataset type: {args.dataset}")

    if args.dataset == "fineweb":
        samples = sample_fineweb_sentences(args.n_samples, args.seed)
    elif args.dataset == "chat":
        samples = sample_chat_conversations(args.n_samples, args.seed)
    elif args.dataset == "mixed":
        samples = sample_mixed_dataset(args.n_samples, args.seed)
    else:
        print(f"Unknown dataset type: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("Failed to sample any prompts", file=sys.stderr)
        sys.exit(1)

    # Create job
    job_dir = create_job(
        samples=samples,
        n_gpus=args.n_gpus,
        output_dir=args.output_dir,
        job_output_dir=args.job_output_dir,
        partition=args.partition,
        time_limit=args.time,
        mem=args.mem,
        tau=args.tau,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dataset_type=args.dataset,
    )

    print(f"\nCreated job: {job_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {len(samples)}")
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Samples per GPU: ~{len(samples) // args.n_gpus}")
    print(f"  Output directory: {args.output_dir}")

    if args.dry_run:
        print("\nDry run - job not submitted")
        print("To submit manually:")
        print(f"  sbatch {job_dir}/submit.sh")
        print("  # Then after completion:")
        print(f"  sbatch {job_dir}/finalize.sh")
        return

    # Submit job
    array_id, finalize_id = submit_job(job_dir)
    if array_id:
        print("\nSubmitted to SLURM:")
        print(f"  Array job ID: {array_id}")
        print(f"  Finalize job ID: {finalize_id} (runs after array completes)")
        print("\nMonitor with: squeue -u $USER")
        print(f"Graphs will be saved to: {args.output_dir}")
    else:
        print("\nFailed to submit to SLURM")
        print(f"Submit manually: sbatch {job_dir}/submit.sh")


if __name__ == "__main__":
    main()

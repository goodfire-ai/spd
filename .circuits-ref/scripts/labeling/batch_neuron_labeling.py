#!/usr/bin/env python3
"""Batch neuron labeling using LLM for large-scale automated labeling.

Fully automated, SLURM-aware labeling for thousands of neurons.
Supports distributed execution across multiple nodes with checkpoint/resume.

Usage:
    # Launch batch labeling job (10 workers, ~5000 neurons)
    python scripts/batch_neuron_labeling.py \
        --edge-stats data/fabric_edge_stats.json \
        --n-workers 10 \
        --output-dir data/fabric_labels

    # Worker mode (called by SLURM)
    python scripts/batch_neuron_labeling.py \
        --worker --job-dir <path> --task-id <N>

    # Finalize and merge results
    python scripts/batch_neuron_labeling.py \
        --finalize --job-dir <path>

    # Single-node mode (no SLURM)
    python scripts/batch_neuron_labeling.py \
        --edge-stats data/fabric_edge_stats.json \
        --local --batch-size 64
"""

import argparse
import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# Label Schemas
# =============================================================================

@dataclass
class NeuronLabel:
    """Complete label for a neuron."""
    neuron_id: str
    layer: int
    neuron_idx: int

    # Output function (Pass 1)
    output_label: str = ""
    output_description: str = ""
    output_type: str = ""  # semantic, formatting, routing, lexical, etc.
    output_interpretability: str = "medium"  # low, medium, high

    # Input function (Pass 2)
    input_label: str = ""
    input_description: str = ""
    input_type: str = ""  # token-pattern, context, position, upstream-gated, etc.
    input_interpretability: str = "medium"

    # Interestingness scoring
    interestingness_score: int = 5  # 1-10
    interestingness_reason: str = ""

    # Metadata
    confidence: str = "llm-auto"
    appearance_count: int = 0
    domain_specificity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "layer": self.layer,
            "neuron_idx": self.neuron_idx,
            "output_label": self.output_label,
            "output_description": self.output_description,
            "output_type": self.output_type,
            "output_interpretability": self.output_interpretability,
            "input_label": self.input_label,
            "input_description": self.input_description,
            "input_type": self.input_type,
            "input_interpretability": self.input_interpretability,
            "interestingness_score": self.interestingness_score,
            "interestingness_reason": self.interestingness_reason,
            "confidence": self.confidence,
            "appearance_count": self.appearance_count,
            "domain_specificity": self.domain_specificity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NeuronLabel":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Prompt Building
# =============================================================================

def build_output_prompt(profile: dict[str, Any], downstream_labels: dict[str, str]) -> str:
    """Build prompt for OUTPUT function labeling (Pass 1).

    Includes interestingness scoring request.
    """
    neuron_id = profile["neuron_id"]
    layer = profile["layer"]

    parts = []
    parts.append(f"Neuron: {neuron_id} (Layer {layer} of 31)")
    parts.append(f"Appearances: {profile.get('appearance_count', 0)}")

    # Direct effect ratio
    der = profile.get("direct_effect_ratio", {})
    if der:
        effect_type = der.get("effect_type", "unknown")
        mean = der.get("mean", 0)
        parts.append(f"Direct effect ratio: {mean:.1%} ({effect_type})")

    # Transluce labels
    transluce_pos = profile.get("transluce_label_positive", "")
    transluce_neg = profile.get("transluce_label_negative", "")
    if transluce_pos:
        parts.append(f"Activates on: {transluce_pos}")
    if transluce_neg:
        parts.append(f"Suppressed by: {transluce_neg}")

    # Output projection
    output_proj = profile.get("output_projection", {})
    promotes = output_proj.get("promotes", output_proj.get("promoted", []))
    suppresses = output_proj.get("suppresses", output_proj.get("suppressed", []))

    if promotes or suppresses:
        parts.append("\n=== OUTPUT PROJECTION ===")
        if promotes:
            parts.append("PROMOTES:")
            for t in promotes[:5]:
                token = t.get("token", "?")
                weight = t.get("logit_contribution", t.get("weight", 0))
                parts.append(f"  {repr(token):25s} ({weight:+.4f})")
        if suppresses:
            parts.append("SUPPRESSES:")
            for t in suppresses[:5]:
                token = t.get("token", "?")
                weight = t.get("logit_contribution", t.get("weight", 0))
                parts.append(f"  {repr(token):25s} ({weight:+.4f})")

    # Downstream neurons with labels
    downstream = profile.get("top_downstream_targets", [])
    labeled_downstream = []
    for d in downstream:
        target = d.get("target", "")
        if target.startswith("L_"):
            continue  # Skip logit targets
        # Parse target format: "0_491" -> "L0/N491"
        parts_t = target.split("_")
        if len(parts_t) >= 2:
            target_id = f"L{parts_t[0]}/N{parts_t[1]}"
            label = downstream_labels.get(target_id, "")
            if label:
                labeled_downstream.append((target_id, label, d.get("avg_weight", 0)))

    if labeled_downstream:
        parts.append("\n=== DOWNSTREAM NEURONS (with known functions) ===")
        for target_id, label, weight in sorted(labeled_downstream, key=lambda x: -abs(x[2]))[:6]:
            sign = "ACTIVATES" if weight > 0 else "INHIBITS"
            parts.append(f"  {sign}: \"{label}\" ({target_id}, w={weight:+.3f})")

    context = "\n".join(parts)

    return f"""You are analyzing a neuron in Llama-3.1-8B to understand its OUTPUT FUNCTION.

{context}

Analyze this neuron and provide a structured response:

1. INTERPRETABILITY: How confident are you that this neuron has a clear, unifying pattern?
   - "high" = Clear, obvious unifying pattern
   - "medium" = Likely pattern but some noise
   - "low" = No clear unifying pattern

2. TYPE: What category describes this neuron's output effect?
   - "semantic" = Promotes/suppresses tokens with shared meaning
   - "formatting" = Affects punctuation, whitespace, capitalization
   - "structural" = Sentence boundaries, list markers
   - "lexical" = Letter/character patterns
   - "routing" = Works through downstream neurons (not L31)
   - "unknown" = Cannot determine

3. SHORT_LABEL: A brief 3-8 word label summarizing the function.
   - If uninterpretable: "uninterpretable" or "uninterpretable-routing"

4. OUTPUT_FUNCTION: A 1-2 sentence description of what this neuron DOES when it fires.

5. INTERESTINGNESS_SCORE: Rate 1-10 how interesting this neuron is for interpretability research:
   - 10 = Highly interpretable semantic neuron with clear causal role
   - 7-9 = Interpretable neuron with interesting behavior
   - 4-6 = Average neuron, somewhat interpretable
   - 1-3 = Uninterpretable or purely structural

6. INTERESTINGNESS_REASON: Brief explanation of the score.

Respond in this exact format:
INTERPRETABILITY: [low/medium/high]
TYPE: [semantic/formatting/structural/lexical/routing/unknown]
SHORT_LABEL: [3-8 word label]
OUTPUT_FUNCTION: [1-2 sentence description]
INTERESTINGNESS_SCORE: [1-10]
INTERESTINGNESS_REASON: [brief explanation]"""


def build_input_prompt(
    profile: dict[str, Any],
    output_label: str,
    output_description: str,
    upstream_labels: dict[str, str]
) -> str:
    """Build prompt for INPUT function labeling (Pass 2)."""
    neuron_id = profile["neuron_id"]
    layer = profile["layer"]

    parts = []
    parts.append(f"Neuron: {neuron_id} (Layer {layer} of 31)")

    # Show known output function
    if output_label:
        parts.append(f"\n*** KNOWN OUTPUT FUNCTION: {output_label} ***")
        if output_description:
            parts.append(f"    {output_description}")

    # Transluce labels
    transluce_pos = profile.get("transluce_label_positive", "")
    transluce_neg = profile.get("transluce_label_negative", "")
    if transluce_pos:
        parts.append(f"\nActivates on: {transluce_pos}")
    if transluce_neg:
        parts.append(f"Suppressed by: {transluce_neg}")

    # Input projection
    input_proj = profile.get("input_projection", {})
    activates = input_proj.get("activates", [])
    suppresses = input_proj.get("suppresses", [])

    if activates or suppresses:
        parts.append("\n=== INPUT PROJECTION (static token sensitivity) ===")
        if activates:
            parts.append("ACTIVATING tokens:")
            for t in activates[:5]:
                parts.append(f"  {repr(t.get('token', '?')):25s} ({t.get('weight', 0):+.6f})")
        if suppresses:
            parts.append("SUPPRESSING tokens:")
            for t in suppresses[:5]:
                parts.append(f"  {repr(t.get('token', '?')):25s} ({t.get('weight', 0):+.6f})")

    # Upstream neurons with labels
    upstream = profile.get("top_upstream_sources", [])
    labeled_upstream = []
    for u in upstream:
        source = u.get("source", "")
        if source.startswith("E_"):
            continue  # Skip embedding sources
        parts_s = source.split("_")
        if len(parts_s) >= 2:
            source_id = f"L{parts_s[0]}/N{parts_s[1]}"
            label = upstream_labels.get(source_id, "")
            if label:
                labeled_upstream.append((source_id, label, u.get("avg_weight", 0)))

    if labeled_upstream:
        parts.append("\n=== UPSTREAM NEURONS (with known functions) ===")
        for source_id, label, weight in sorted(labeled_upstream, key=lambda x: -abs(x[2]))[:6]:
            sign = "EXCITES" if weight > 0 else "INHIBITS"
            parts.append(f"  {sign}: \"{label}\" ({source_id}, w={weight:+.3f})")

    context = "\n".join(parts)

    return f"""You are analyzing a neuron in Llama-3.1-8B to understand its INPUT FUNCTION.

{context}

Your task: Determine what TRIGGERS this neuron to fire.

Provide a structured response:

1. INTERPRETABILITY: How confident are you that there's a clear triggering pattern?
   - "high" = Clear, obvious trigger
   - "medium" = Likely pattern but incomplete evidence
   - "low" = Cannot determine

2. TYPE: What category describes the input trigger?
   - "token-pattern" = Triggered by specific tokens/words
   - "context" = Triggered by semantic context
   - "position" = Triggered by position in sequence
   - "upstream-gated" = Controlled by upstream neurons
   - "combination" = Multiple factors
   - "unknown" = Cannot determine

3. SHORT_LABEL: A brief 3-8 word label for the input trigger.

4. INPUT_FUNCTION: A 1-2 sentence description of what triggers this neuron.

Respond in this exact format:
INTERPRETABILITY: [low/medium/high]
TYPE: [token-pattern/context/position/upstream-gated/combination/unknown]
SHORT_LABEL: [3-8 word label]
INPUT_FUNCTION: [1-2 sentence description]"""


def parse_output_response(response: str) -> dict[str, Any]:
    """Parse structured LLM response for output labeling."""
    import re

    result = {
        "interpretability": "medium",
        "type": "",
        "short_label": "",
        "output_function": "",
        "interestingness_score": 5,
        "interestingness_reason": "",
    }

    # Clean markdown
    clean = re.sub(r'\*+', '', response)

    patterns = {
        "interpretability": r'(?i)INTERPRETABILITY[:\s]+(\w+)',
        "type": r'(?i)\bTYPE[:\s]+([\w-]+)',
        "short_label": r'(?i)SHORT[_\s]?LABEL[:\s]+["\']?([^"\'\n]+)',
        "output_function": r'(?i)OUTPUT[_\s]?FUNCTION[:\s]+(.+?)(?=\n[A-Z_]+:|$)',
        "interestingness_score": r'(?i)INTERESTINGNESS[_\s]?SCORE[:\s]+(\d+)',
        "interestingness_reason": r'(?i)INTERESTINGNESS[_\s]?REASON[:\s]+(.+?)(?=\n[A-Z_]+:|$)',
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, clean, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if field == "interpretability":
                value = value.lower()
                if value not in ("low", "medium", "high"):
                    value = "medium"
            elif field == "interestingness_score":
                try:
                    value = min(10, max(1, int(value)))
                except ValueError:
                    value = 5
            result[field] = value

    return result


def parse_input_response(response: str) -> dict[str, Any]:
    """Parse structured LLM response for input labeling."""
    import re

    result = {
        "interpretability": "medium",
        "type": "",
        "short_label": "",
        "input_function": "",
    }

    clean = re.sub(r'\*+', '', response)

    patterns = {
        "interpretability": r'(?i)INTERPRETABILITY[:\s]+(\w+)',
        "type": r'(?i)\bTYPE[:\s]+([\w-]+)',
        "short_label": r'(?i)SHORT[_\s]?LABEL[:\s]+["\']?([^"\'\n]+)',
        "input_function": r'(?i)INPUT[_\s]?FUNCTION[:\s]+(.+?)(?=\n[A-Z_]+:|$)',
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, clean, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if field == "interpretability":
                value = value.lower()
                if value not in ("low", "medium", "high"):
                    value = "medium"
            result[field] = value

    return result


# =============================================================================
# SLURM Job Management
# =============================================================================

def generate_slurm_script(
    job_dir: Path,
    n_tasks: int,
    partition: str,
    time_limit: str,
    mem: str,
    repo_dir: Path,
) -> str:
    """Generate SLURM array job script for batch labeling."""
    return f"""#!/bin/bash
#SBATCH --job-name=neuron-label
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --partition={partition}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
#SBATCH --time={time_limit}
#SBATCH --output={job_dir}/logs/task_%a_%j.out
#SBATCH --error={job_dir}/logs/task_%a_%j.err

echo "=== Batch Neuron Labeling Worker ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================="

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

.venv/bin/python scripts/batch_neuron_labeling.py \\
    --worker \\
    --job-dir {job_dir} \\
    --task-id $SLURM_ARRAY_TASK_ID

echo "=== Worker Complete ==="
echo "End time: $(date)"
"""


def generate_finalize_script(job_dir: Path, repo_dir: Path, partition: str) -> str:
    """Generate SLURM script for finalize job."""
    return f"""#!/bin/bash
#SBATCH --job-name=label-finalize
#SBATCH --partition={partition}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output={job_dir}/logs/finalize_%j.out
#SBATCH --error={job_dir}/logs/finalize_%j.err

echo "=== Batch Labeling Finalize ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd {repo_dir}
source .venv/bin/activate 2>/dev/null || true

.venv/bin/python scripts/batch_neuron_labeling.py \\
    --finalize \\
    --job-dir {job_dir}

echo "=== Finalize Complete ==="
echo "End time: $(date)"
"""


def create_labeling_job(
    edge_stats_path: Path,
    n_workers: int,
    output_dir: Path,
    job_output_dir: Path,
    partition: str,
    time_limit: str,
    mem: str,
    min_appearances: int,
    batch_size: int,
    model: str,
    label_pass: str,
) -> Path:
    """Create job directory and files for batch labeling."""
    job_id = f"label_{label_pass}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = job_output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir()
    (job_dir / "partial").mkdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load edge stats and filter neurons
    with open(edge_stats_path) as f:
        edge_stats = json.load(f)

    profiles = edge_stats.get("profiles", [])
    filtered = [p for p in profiles if p.get("appearance_count", 0) >= min_appearances]

    # Sort by layer for proper ordering
    if label_pass == "output":
        # Output pass: L31 -> L0
        filtered.sort(key=lambda x: (-x["layer"], -x.get("appearance_count", 0)))
    else:
        # Input pass: L0 -> L31
        filtered.sort(key=lambda x: (x["layer"], -x.get("appearance_count", 0)))

    print(f"Total neurons after filtering: {len(filtered)}", file=sys.stderr)

    # Distribute neurons across workers
    neurons_per_worker = len(filtered) // n_workers
    extra = len(filtered) % n_workers

    for task_id in range(n_workers):
        start = task_id * neurons_per_worker + min(task_id, extra)
        end = start + neurons_per_worker + (1 if task_id < extra else 0)
        task_neurons = filtered[start:end]

        task_file = job_dir / f"neurons_task_{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_neurons, f)

    # Save config
    config = {
        "n_workers": n_workers,
        "total_neurons": len(filtered),
        "edge_stats_path": str(edge_stats_path),
        "output_dir": str(output_dir),
        "min_appearances": min_appearances,
        "batch_size": batch_size,
        "model": model,
        "label_pass": label_pass,
    }
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate SLURM scripts
    repo_dir = Path(__file__).parent.parent.resolve()
    script = generate_slurm_script(job_dir, n_workers, partition, time_limit, mem, repo_dir)
    with open(job_dir / "submit.sh", "w") as f:
        f.write(script)

    finalize_script = generate_finalize_script(job_dir, repo_dir, partition)
    with open(job_dir / "finalize.sh", "w") as f:
        f.write(finalize_script)

    return job_dir


def submit_labeling_job(job_dir: Path) -> tuple[int | None, int | None]:
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


# =============================================================================
# Worker Execution
# =============================================================================

async def run_worker_async(job_dir: Path, task_id: int):
    """Execute worker for a single SLURM task."""
    if not OPENAI_AVAILABLE:
        print("Error: openai package not installed", file=sys.stderr)
        sys.exit(1)

    print(f"Worker {task_id} starting...")

    # Load config
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    batch_size = config.get("batch_size", 64)
    model = config.get("model", "gpt-5")
    label_pass = config.get("label_pass", "output")
    output_dir = Path(config["output_dir"])

    # Load neurons for this task
    neurons_file = job_dir / f"neurons_task_{task_id}.json"
    with open(neurons_file) as f:
        neurons = json.load(f)

    print(f"Task {task_id}: Processing {len(neurons)} neurons ({label_pass} pass)")

    # Load existing labels for compositional prompts
    existing_labels = {}
    labels_file = output_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            data = json.load(f)
        for lbl in data.get("labels", []):
            nid = lbl.get("neuron_id", "")
            if label_pass == "output":
                existing_labels[nid] = lbl.get("output_label", "")
            else:
                existing_labels[nid] = lbl.get("input_label", "")

    # Initialize async client
    client = AsyncOpenAI()

    # Process in batches
    results = []
    processed = 0
    failed = 0

    for i in range(0, len(neurons), batch_size):
        batch = neurons[i:i + batch_size]

        # Build prompts
        prompts = []
        for profile in batch:
            if label_pass == "output":
                prompt = build_output_prompt(profile, existing_labels)
            else:
                output_label = existing_labels.get(profile["neuron_id"], "")
                prompt = build_input_prompt(profile, output_label, "", existing_labels)
            prompts.append(prompt)

        # Call LLM in parallel with rate limiting
        async def call_llm(prompt: str) -> str:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4000,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"ERROR: {e}"

        tasks = [call_llm(p) for p in prompts]
        responses = await asyncio.gather(*tasks)

        # Parse responses
        for profile, response in zip(batch, responses):
            nid = profile["neuron_id"]

            if response.startswith("ERROR:"):
                print(f"  Task {task_id}: Error for {nid}: {response}", file=sys.stderr)
                failed += 1
                continue

            if label_pass == "output":
                parsed = parse_output_response(response)
                label = NeuronLabel(
                    neuron_id=nid,
                    layer=profile["layer"],
                    neuron_idx=profile.get("neuron", 0),
                    output_label=parsed.get("short_label", ""),
                    output_description=parsed.get("output_function", ""),
                    output_type=parsed.get("type", ""),
                    output_interpretability=parsed.get("interpretability", "medium"),
                    interestingness_score=parsed.get("interestingness_score", 5),
                    interestingness_reason=parsed.get("interestingness_reason", ""),
                    appearance_count=profile.get("appearance_count", 0),
                    domain_specificity=profile.get("domain_specificity", 0.0),
                )
            else:
                parsed = parse_input_response(response)
                # Merge with existing output label if available
                label = NeuronLabel(
                    neuron_id=nid,
                    layer=profile["layer"],
                    neuron_idx=profile.get("neuron", 0),
                    input_label=parsed.get("short_label", ""),
                    input_description=parsed.get("input_function", ""),
                    input_type=parsed.get("type", ""),
                    input_interpretability=parsed.get("interpretability", "medium"),
                    appearance_count=profile.get("appearance_count", 0),
                    domain_specificity=profile.get("domain_specificity", 0.0),
                )

            results.append(label.to_dict())
            processed += 1

            # Update existing labels for compositional prompts
            if label_pass == "output":
                existing_labels[nid] = label.output_label
            else:
                existing_labels[nid] = label.input_label

        print(f"Task {task_id}: Processed {processed}/{len(neurons)} neurons")

        # Rate limiting: small delay between batches
        await asyncio.sleep(0.5)

    # Save partial results
    partial_file = job_dir / "partial" / f"task_{task_id}.json"
    with open(partial_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "processed": processed,
            "failed": failed,
            "labels": results,
        }, f, indent=2)

    print(f"Task {task_id}: Complete. Processed {processed}, failed {failed}")


def run_worker(job_dir: Path, task_id: int):
    """Sync wrapper for async worker."""
    asyncio.run(run_worker_async(job_dir, task_id))


# =============================================================================
# Finalize
# =============================================================================

def run_finalize(job_dir: Path):
    """Combine labels from all workers."""
    print("Finalizing results...")

    with open(job_dir / "config.json") as f:
        config = json.load(f)

    n_workers = config["n_workers"]
    output_dir = Path(config["output_dir"])
    label_pass = config.get("label_pass", "output")

    # Collect all labels
    all_labels = []
    total_processed = 0
    total_failed = 0

    for task_id in range(n_workers):
        partial_file = job_dir / "partial" / f"task_{task_id}.json"
        if not partial_file.exists():
            print(f"Warning: Missing partial file for task {task_id}", file=sys.stderr)
            continue

        with open(partial_file) as f:
            partial = json.load(f)

        total_processed += partial["processed"]
        total_failed += partial["failed"]
        all_labels.extend(partial["labels"])

        print(f"Task {task_id}: {partial['processed']} processed, {partial['failed']} failed")

    # Merge with existing labels if doing input pass
    if label_pass == "input":
        labels_file = output_dir / "labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                existing = json.load(f)
            existing_by_id = {l["neuron_id"]: l for l in existing.get("labels", [])}

            # Merge input labels into existing output labels
            for lbl in all_labels:
                nid = lbl["neuron_id"]
                if nid in existing_by_id:
                    existing_by_id[nid].update({
                        "input_label": lbl["input_label"],
                        "input_description": lbl["input_description"],
                        "input_type": lbl["input_type"],
                        "input_interpretability": lbl["input_interpretability"],
                    })
                else:
                    existing_by_id[nid] = lbl

            all_labels = list(existing_by_id.values())

    # Sort by layer and neuron
    all_labels.sort(key=lambda x: (x["layer"], x.get("neuron_idx", 0)))

    # Compute statistics
    interestingness_dist = defaultdict(int)
    type_dist = defaultdict(int)
    interpretability_dist = defaultdict(int)

    for lbl in all_labels:
        interestingness_dist[lbl.get("interestingness_score", 5)] += 1
        if label_pass == "output":
            type_dist[lbl.get("output_type", "unknown")] += 1
            interpretability_dist[lbl.get("output_interpretability", "medium")] += 1
        else:
            type_dist[lbl.get("input_type", "unknown")] += 1
            interpretability_dist[lbl.get("input_interpretability", "medium")] += 1

    # Save combined labels
    labels_file = output_dir / "labels.json"
    with open(labels_file, "w") as f:
        json.dump({
            "metadata": {
                "total_neurons": len(all_labels),
                "processed": total_processed,
                "failed": total_failed,
                "label_pass": label_pass,
                "model": config.get("model", "gpt-5"),
                "timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                "interestingness_distribution": dict(interestingness_dist),
                "type_distribution": dict(type_dist),
                "interpretability_distribution": dict(interpretability_dist),
            },
            "labels": all_labels,
        }, f, indent=2)

    # Also save as simpler lookup format
    lookup_file = output_dir / "label_lookup.json"
    lookup = {}
    for lbl in all_labels:
        nid = lbl["neuron_id"]
        lookup[nid] = {
            "output": lbl.get("output_label", ""),
            "input": lbl.get("input_label", ""),
            "type": lbl.get("output_type", "") or lbl.get("input_type", ""),
            "interestingness": lbl.get("interestingness_score", 5),
        }
    with open(lookup_file, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Labeling complete ({label_pass} pass)!")
    print(f"  Total neurons: {len(all_labels)}")
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Output: {labels_file}")
    print(f"  Lookup: {lookup_file}")
    print(f"\nType distribution: {dict(type_dist)}")
    print(f"Interpretability: {dict(interpretability_dist)}")
    print(f"Interestingness scores: min={min(interestingness_dist.keys())}, max={max(interestingness_dist.keys())}")


# =============================================================================
# Local Mode (single node, no SLURM)
# =============================================================================

async def run_local_async(
    edge_stats_path: Path,
    output_dir: Path,
    min_appearances: int,
    batch_size: int,
    model: str,
    label_pass: str,
):
    """Run labeling locally without SLURM."""
    if not OPENAI_AVAILABLE:
        print("Error: openai package not installed", file=sys.stderr)
        sys.exit(1)

    print(f"Running local labeling ({label_pass} pass)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load edge stats
    with open(edge_stats_path) as f:
        edge_stats = json.load(f)

    profiles = edge_stats.get("profiles", [])
    filtered = [p for p in profiles if p.get("appearance_count", 0) >= min_appearances]

    # Sort by layer
    if label_pass == "output":
        filtered.sort(key=lambda x: (-x["layer"], -x.get("appearance_count", 0)))
    else:
        filtered.sort(key=lambda x: (x["layer"], -x.get("appearance_count", 0)))

    print(f"Processing {len(filtered)} neurons...")

    # Load existing labels
    existing_labels = {}
    labels_file = output_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            data = json.load(f)
        for lbl in data.get("labels", []):
            nid = lbl.get("neuron_id", "")
            if label_pass == "output":
                existing_labels[nid] = lbl.get("output_label", "")
            else:
                existing_labels[nid] = lbl.get("input_label", "")

    client = AsyncOpenAI()
    results = []
    processed = 0
    failed = 0

    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]

        prompts = []
        for profile in batch:
            if label_pass == "output":
                prompt = build_output_prompt(profile, existing_labels)
            else:
                output_label = existing_labels.get(profile["neuron_id"], "")
                prompt = build_input_prompt(profile, output_label, "", existing_labels)
            prompts.append(prompt)

        async def call_llm(prompt: str) -> str:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4000,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"ERROR: {e}"

        tasks = [call_llm(p) for p in prompts]
        responses = await asyncio.gather(*tasks)

        for profile, response in zip(batch, responses):
            nid = profile["neuron_id"]

            if response.startswith("ERROR:"):
                print(f"  Error for {nid}: {response}", file=sys.stderr)
                failed += 1
                continue

            if label_pass == "output":
                parsed = parse_output_response(response)
                label = NeuronLabel(
                    neuron_id=nid,
                    layer=profile["layer"],
                    neuron_idx=profile.get("neuron", 0),
                    output_label=parsed.get("short_label", ""),
                    output_description=parsed.get("output_function", ""),
                    output_type=parsed.get("type", ""),
                    output_interpretability=parsed.get("interpretability", "medium"),
                    interestingness_score=parsed.get("interestingness_score", 5),
                    interestingness_reason=parsed.get("interestingness_reason", ""),
                    appearance_count=profile.get("appearance_count", 0),
                )
            else:
                parsed = parse_input_response(response)
                label = NeuronLabel(
                    neuron_id=nid,
                    layer=profile["layer"],
                    neuron_idx=profile.get("neuron", 0),
                    input_label=parsed.get("short_label", ""),
                    input_description=parsed.get("input_function", ""),
                    input_type=parsed.get("type", ""),
                    input_interpretability=parsed.get("interpretability", "medium"),
                    appearance_count=profile.get("appearance_count", 0),
                )

            results.append(label.to_dict())
            processed += 1

            if label_pass == "output":
                existing_labels[nid] = label.output_label
            else:
                existing_labels[nid] = label.input_label

        print(f"Processed {processed}/{len(filtered)} neurons")
        await asyncio.sleep(0.5)

    # Save results
    all_labels = results
    if label_pass == "input" and labels_file.exists():
        with open(labels_file) as f:
            existing = json.load(f)
        existing_by_id = {l["neuron_id"]: l for l in existing.get("labels", [])}
        for lbl in all_labels:
            nid = lbl["neuron_id"]
            if nid in existing_by_id:
                existing_by_id[nid].update({
                    "input_label": lbl["input_label"],
                    "input_description": lbl["input_description"],
                    "input_type": lbl["input_type"],
                    "input_interpretability": lbl["input_interpretability"],
                })
            else:
                existing_by_id[nid] = lbl
        all_labels = list(existing_by_id.values())

    all_labels.sort(key=lambda x: (x["layer"], x.get("neuron_idx", 0)))

    with open(labels_file, "w") as f:
        json.dump({
            "metadata": {
                "total_neurons": len(all_labels),
                "processed": processed,
                "failed": failed,
                "label_pass": label_pass,
                "model": model,
                "timestamp": datetime.now().isoformat(),
            },
            "labels": all_labels,
        }, f, indent=2)

    print(f"\nComplete! Saved {len(all_labels)} labels to {labels_file}")


def run_local(
    edge_stats_path: Path,
    output_dir: Path,
    min_appearances: int,
    batch_size: int,
    model: str,
    label_pass: str,
):
    """Sync wrapper for local mode."""
    asyncio.run(run_local_async(
        edge_stats_path, output_dir, min_appearances, batch_size, model, label_pass
    ))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch neuron labeling with LLM"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--worker", action="store_true",
                           help="Run as worker (called by SLURM)")
    mode_group.add_argument("--finalize", action="store_true",
                           help="Finalize and merge results")
    mode_group.add_argument("--local", action="store_true",
                           help="Run locally without SLURM")

    # Launcher arguments
    parser.add_argument("--edge-stats", type=Path,
                        help="Path to edge statistics JSON")
    parser.add_argument("--n-workers", type=int, default=10,
                        help="Number of SLURM workers (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/fabric_labels"),
                        help="Output directory for labels")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create job but don't submit")

    # SLURM options
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition")
    parser.add_argument("--time", default="4:00:00",
                        help="Time limit per task")
    parser.add_argument("--mem", default="16G",
                        help="Memory per task")

    # Labeling options
    parser.add_argument("--min-appearances", type=int, default=50,
                        help="Minimum appearances to label a neuron")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for parallel API calls")
    parser.add_argument("--model", default="gpt-5",
                        help="OpenAI model to use")
    parser.add_argument("--pass", dest="label_pass", choices=["output", "input"],
                        default="output",
                        help="Labeling pass: output or input")

    # Worker/finalize arguments
    parser.add_argument("--job-dir", type=Path,
                        help="Job directory (for worker/finalize modes)")
    parser.add_argument("--task-id", type=int,
                        help="Task ID (for worker mode)")

    # Job directory
    parser.add_argument("--job-output-dir", type=Path, default=Path("outputs/labeling_jobs"),
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

    # Local mode
    if args.local:
        if not args.edge_stats:
            print("Local mode requires --edge-stats", file=sys.stderr)
            sys.exit(1)
        run_local(
            args.edge_stats,
            args.output_dir,
            args.min_appearances,
            args.batch_size,
            args.model,
            args.label_pass,
        )
        return

    # Launcher mode
    if not args.edge_stats:
        print("Launcher mode requires --edge-stats", file=sys.stderr)
        sys.exit(1)

    job_dir = create_labeling_job(
        edge_stats_path=args.edge_stats,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        job_output_dir=args.job_output_dir,
        partition=args.partition,
        time_limit=args.time,
        mem=args.mem,
        min_appearances=args.min_appearances,
        batch_size=args.batch_size,
        model=args.model,
        label_pass=args.label_pass,
    )

    print(f"\nCreated labeling job: {job_dir}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Pass: {args.label_pass}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")

    if args.dry_run:
        print("\nDry run - job not submitted")
        print(f"To submit manually: sbatch {job_dir}/submit.sh")
        return

    array_id, finalize_id = submit_labeling_job(job_dir)
    if array_id:
        print("\nSubmitted to SLURM:")
        print(f"  Array job ID: {array_id}")
        print(f"  Finalize job ID: {finalize_id}")
        print("\nMonitor with: squeue -u $USER")
    else:
        print("\nFailed to submit to SLURM")
        print(f"Submit manually: sbatch {job_dir}/submit.sh")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch NeuronPI investigation using SLURM.

Distributes NeuronPI (with GPT peer review) across multiple GPUs.

Usage:
    # Investigate neurons from a JSON list (Llama default)
    python scripts/batch_slurm_neuronpi.py \
        --neurons-file /tmp/aspirin_cox_neurons.json \
        --agents-per-gpu 4 \
        --max-gpus 36 \
        --html-output-dir frontend/reports/aspirin-cox

    # Investigate Qwen3-32B neurons
    python scripts/batch_slurm_neuronpi.py \
        --neurons-file data/qwen3_resveratrol_neurons.json \
        --model-config qwen3-32b \
        --agents-per-gpu 1 \
        --html-output-dir neuron_reports/html

    # Dry run to see job distribution
    python scripts/batch_slurm_neuronpi.py \
        --neurons-file /tmp/aspirin_cox_neurons.json \
        --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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
                         html_output_dir: Path, edge_stats: str = None,
                         max_concurrent: int = 10, project_root: str = None,
                         max_review_iterations: int = 2,
                         model_config: str = None,
                         both_polarities: bool = False,
                         gpu_server_port: int = 8477,
                         labels_path: str = None) -> str:
    """Create a worker script for a single SLURM job using NeuronPI.

    The generated script:
    1. Starts a GPU inference server as a background subprocess
    2. Waits for the server to be healthy
    3. Launches N agent subprocesses (each runs run_neuron_pi with gpu_server_url)
    4. Waits for all agents to finish
    5. Kills the GPU server
    """

    if project_root is None:
        project_root = str(Path.cwd())

    # Build model config arg for gpu_server
    model_config_arg = model_config or "llama-3.1-8b"

    output_dir_abs = str(Path(output_dir).absolute())
    html_output_dir_abs = str(Path(html_output_dir).absolute())

    # The worker config is serialized as JSON and embedded in the generated script.
    # Each agent subprocess reads its own neuron spec from NEURONS[i] at runtime.
    worker_config = {
        'project_root': project_root,
        'output_dir': output_dir_abs,
        'html_output_dir': html_output_dir_abs,
        'edge_stats': edge_stats,
        'labels_path': labels_path,
        'max_review_iterations': max_review_iterations,
        'gpu_server_port': gpu_server_port,
        'model_config': model_config,
        'model_config_arg': model_config_arg,
        'both_polarities': both_polarities,
        'job_id': job_id,
    }

    script = f'''#!/usr/bin/env python3
"""Auto-generated NeuronPI worker script for SLURM job {job_id}.

Architecture: GPU server + agent subprocesses.
- One GPU server process hosts the model and serializes GPU operations.
- Each neuron investigation runs as an independent subprocess.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Use absolute path to project root
PROJECT_ROOT = "{project_root}"
sys.path.insert(0, PROJECT_ROOT)

NEURONS = {json.dumps(neurons)}
CONFIG = {repr(worker_config)}
OUTPUT_DIR = Path(CONFIG['output_dir'])
HTML_OUTPUT_DIR = Path(CONFIG['html_output_dir'])
GPU_SERVER_PORT = CONFIG['gpu_server_port']
MODEL_CONFIG = CONFIG['model_config_arg']
JOB_ID = CONFIG['job_id']


def wait_for_server(port, pid, timeout=600, interval=5):
    """Wait for the GPU server to become healthy."""
    url = f"http://localhost:{{port}}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            os.kill(pid, 0)
        except OSError:
            print(f"[Job {{JOB_ID}}] GPU server process (PID={{pid}}) died!", flush=True)
            return False
        try:
            req = urllib.request.urlopen(url, timeout=5)
            if req.status == 200:
                elapsed = int(time.time() - start)
                print(f"[Job {{JOB_ID}}] GPU server ready after {{elapsed}}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"[Job {{JOB_ID}}] GPU server failed to start within {{timeout}}s", flush=True)
    return False


def build_agent_script(neuron_spec, agent_index):
    """Build a self-contained Python script for one agent subprocess.

    Uses string concatenation (not f-strings) to construct lines of Python
    source code, avoiding nested f-string escaping issues.
    """
    layer = neuron_spec['layer']
    neuron_idx = neuron_spec['neuron']
    hypothesis = neuron_spec.get('hypothesis', '').replace('\\\\', '\\\\\\\\').replace("'", "\\\\'")
    edge_stats_path = CONFIG['edge_stats']
    edge_stats_line = "Path('" + edge_stats_path + "')" if edge_stats_path else "None"
    labels_path = CONFIG.get('labels_path')
    labels_line = "Path('" + labels_path + "')" if labels_path else "None"
    initial_label = neuron_spec.get('label', '').replace('\\\\', '\\\\\\\\').replace("'", "\\\\'")
    gpu_server_url = "http://localhost:" + str(GPU_SERVER_PORT)
    both_polarities = CONFIG['both_polarities']
    max_review_iters = str(CONFIG['max_review_iterations'])
    model_config = CONFIG['model_config']
    job_tag = "[Job " + str(JOB_ID) + "][Agent " + str(agent_index) + "]"

    lines = [
        "import asyncio",
        "import sys",
        "import shutil",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, '" + PROJECT_ROOT + "')",
    ]
    if model_config:
        lines.append("from neuron_scientist.tools import set_model_config")
        lines.append("set_model_config('" + model_config + "')")
    lines.extend([
        "from neuron_scientist.pi_agent import run_neuron_pi",
        "",
        "OUTPUT_DIR = Path('" + str(OUTPUT_DIR) + "')",
        "HTML_OUTPUT_DIR = Path('" + str(HTML_OUTPUT_DIR) + "')",
        "",
        "async def main():",
        "    neuron_id = 'L" + str(layer) + "/N" + str(neuron_idx) + "'",
        "    print('" + job_tag + " Starting: ' + neuron_id, flush=True)",
        "    try:",
        "        result = await run_neuron_pi(",
        "            neuron_id=neuron_id,",
        "            initial_label='" + initial_label + "',",
        "            initial_hypothesis='" + hypothesis + "',",
        "            edge_stats_path=" + edge_stats_line + ",",
        "            labels_path=" + labels_line + ",",
        "            output_dir=OUTPUT_DIR,",
        "            html_output_dir=HTML_OUTPUT_DIR,",
        "            model='sonnet',",
        "            scientist_model='opus',",
        "            max_review_iterations=" + max_review_iters + ",",
        "            skip_review=False,",
        "            skip_dashboard=" + str(not both_polarities) + ",",
        "            polarity_mode='positive',",
        "            gpu_server_url='" + gpu_server_url + "',",
        "        )",
        "        verdict = result.final_verdict[:50] if result.final_verdict else 'completed'",
        "        print(f'" + job_tag + " Positive done: {{neuron_id}} ({{verdict}}...)', flush=True)",
    ])

    if both_polarities:
        lines.extend([
            "        print('" + job_tag + " Starting NEGATIVE: ' + neuron_id, flush=True)",
            "        neg_result = await run_neuron_pi(",
            "            neuron_id=neuron_id,",
            "            initial_label='" + initial_label + "',",
            "            initial_hypothesis='',",
            "            edge_stats_path=" + edge_stats_line + ",",
            "            labels_path=" + labels_line + ",",
            "            output_dir=OUTPUT_DIR,",
            "            html_output_dir=HTML_OUTPUT_DIR,",
            "            model='sonnet',",
            "            scientist_model='opus',",
            "            max_review_iterations=" + max_review_iters + ",",
            "            skip_review=False,",
            "            skip_dashboard=True,",
            "            polarity_mode='negative',",
            "            gpu_server_url='" + gpu_server_url + "',",
            "        )",
            "        neg_verdict = neg_result.final_verdict[:50] if neg_result.final_verdict else 'completed'",
            "        print(f'" + job_tag + " Negative done: {{neuron_id}} ({{neg_verdict}}...)', flush=True)",
            "        safe_id = neuron_id.replace('/', '_')",
            "        pos_path = OUTPUT_DIR / (safe_id + '_investigation.json')",
            "        neg_path = OUTPUT_DIR / (safe_id + '_negative_investigation.json')",
            "        if pos_path.exists() and neg_path.exists():",
            "            try:",
            "                import subprocess as sp",
            "                sp.run([sys.executable, 'scripts/investigation/generate_html_report.py', str(pos_path), '--v2', '--model', 'opus', '-o', str(HTML_OUTPUT_DIR), '--negative-investigation', str(neg_path)], timeout=600, check=True)",
            "                print('" + job_tag + " Merged dashboard: ' + neuron_id, flush=True)",
            "            except Exception as e:",
            "                print(f'" + job_tag + " Dashboard failed: {{e}}', flush=True)",
        ])
    else:
        lines.extend([
            "        html_name = 'L" + str(layer) + "_N" + str(neuron_idx) + ".html'",
            "        html_dest = HTML_OUTPUT_DIR / html_name",
            "        if html_dest.exists():",
            "            print('" + job_tag + " HTML: ' + str(html_dest), flush=True)",
            "        elif result.dashboard_path:",
            "            html_src = Path(result.dashboard_path)",
            "            if html_src.exists() and html_src != html_dest:",
            "                shutil.copy2(html_src, html_dest)",
            "                print('" + job_tag + " HTML copied: ' + str(html_dest), flush=True)",
        ])

    lines.extend([
        "        print('" + job_tag + " SUCCESS: ' + neuron_id, flush=True)",
        "    except Exception as e:",
        "        print(f'" + job_tag + " FAILED: {{neuron_id}} - {{e}}', flush=True)",
        "        import traceback",
        "        traceback.print_exc()",
        "        sys.exit(1)",
        "",
        "asyncio.run(main())",
    ])

    return "\\n".join(lines) + "\\n"


def main():
    """Start GPU server, launch agent subprocesses, wait for completion."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start GPU server as background subprocess
    print(f"[Job {{JOB_ID}}] Starting GPU server (model={{MODEL_CONFIG}}, port={{GPU_SERVER_PORT}})...", flush=True)
    gpu_server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "neuron_scientist.gpu_server",
            "--model-config", MODEL_CONFIG,
            "--port", str(GPU_SERVER_PORT),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Set up cleanup on exit
    def cleanup(signum=None, frame=None):
        print(f"[Job {{JOB_ID}}] Cleaning up GPU server (PID={{gpu_server_proc.pid}})...", flush=True)
        gpu_server_proc.terminate()
        try:
            gpu_server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            gpu_server_proc.kill()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Wait for server to be healthy
    if not wait_for_server(GPU_SERVER_PORT, gpu_server_proc.pid):
        print(f"[Job {{JOB_ID}}] GPU server failed to start. Aborting.", flush=True)
        cleanup()
        sys.exit(1)

    # Launch agent subprocesses with staggered starts to reduce API thundering herd
    STAGGER_DELAY = 10  # seconds between agent launches
    MAX_RETRIES = 1     # retry failed agents once

    def launch_agent(neuron_spec, agent_index):
        """Launch a single agent subprocess, return (index, spec, proc, script_path)."""
        agent_script = build_agent_script(neuron_spec, agent_index)
        script_fd, script_path = tempfile.mkstemp(suffix='.py', prefix=f'agent_{{agent_index}}_')
        with os.fdopen(script_fd, 'w') as f:
            f.write(agent_script)
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
        )
        nid = f"L{{neuron_spec['layer']}}/N{{neuron_spec['neuron']}}"
        print(f"[Job {{JOB_ID}}] Launched agent {{agent_index}}: {{nid}} (PID={{proc.pid}})", flush=True)
        return (agent_index, neuron_spec, proc, script_path)

    print(f"[Job {{JOB_ID}}] Launching {{len(NEURONS)}} agents (stagger={{STAGGER_DELAY}}s)...", flush=True)
    # active: agent_index -> (neuron_spec, proc, script_path, attempt)
    active = {{}}
    final_results = {{}}  # agent_index -> returncode

    # Staggered initial launch
    for i, neuron_spec in enumerate(NEURONS):
        idx, spec, proc, spath = launch_agent(neuron_spec, i)
        active[idx] = (spec, proc, spath, 0)
        if i < len(NEURONS) - 1:
            time.sleep(STAGGER_DELAY)

    # Poll until all agents finish, retrying failures immediately
    while active:
        for idx in list(active.keys()):
            spec, proc, spath, attempt = active[idx]
            rc = proc.poll()
            if rc is None:
                continue  # still running
            neuron_id = f"L{{spec['layer']}}/N{{spec['neuron']}}"
            # Clean up script
            try:
                os.unlink(spath)
            except OSError:
                pass

            if rc == 0:
                print(f"[Job {{JOB_ID}}] Agent {{idx}} ({{neuron_id}}) completed successfully", flush=True)
                final_results[idx] = 0
                del active[idx]
            elif attempt < MAX_RETRIES:
                print(f"[Job {{JOB_ID}}] Agent {{idx}} ({{neuron_id}}) FAILED (exit {{rc}}), retrying...", flush=True)
                _, spec2, proc2, spath2 = launch_agent(spec, idx)
                active[idx] = (spec2, proc2, spath2, attempt + 1)
            else:
                print(f"[Job {{JOB_ID}}] Agent {{idx}} ({{neuron_id}}) FAILED (exit {{rc}}), no retries left", flush=True)
                final_results[idx] = rc
                del active[idx]

        if active:
            time.sleep(5)

    # Shutdown GPU server
    print(f"[Job {{JOB_ID}}] Shutting down GPU server...", flush=True)
    cleanup()

    failed = sum(1 for rc in final_results.values() if rc != 0)
    success = len(NEURONS) - failed
    print(f"[Job {{JOB_ID}}] All agents finished. Success: {{success}}/{{len(NEURONS)}}, Failed: {{failed}}", flush=True)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    return script


def submit_slurm_job(
    worker_script: str,
    job_id: int,
    job_name: str,
    partition: str = "h200-reserved",
    output_dir: Path = None,
    time_limit: str = "6:00:00",
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
#SBATCH --gpus=1
#SBATCH --time={time_limit}
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
    parser = argparse.ArgumentParser(description="Batch NeuronPI investigation via SLURM")

    # Input
    parser.add_argument("--neurons-file", required=True,
                        help="JSON file with neuron list")

    # Model configuration
    parser.add_argument("--model-config",
                        choices=["llama-3.1-8b", "qwen3-32b", "qwen3-8b"],
                        default=None,
                        help="Model config to use (default: llama-3.1-8b)")

    # SLURM configuration
    parser.add_argument("--agents-per-gpu", type=int, default=10,
                        help="Number of parallel agents per GPU (default: 10)")
    parser.add_argument("--max-gpus", type=int, default=36,
                        help="Maximum number of GPUs to use (default: 36)")
    parser.add_argument("--partition", default="h200-reserved",
                        help="SLURM partition (default: h200-reserved)")
    parser.add_argument("--time-limit", default="6:00:00",
                        help="Time limit per job (default: 6:00:00)")

    # GPU server configuration
    parser.add_argument("--gpu-server-port", type=int, default=8477,
                        help="Port for the GPU inference server (default: 8477)")

    # NeuronPI options
    parser.add_argument("--max-review-iterations", type=int, default=2,
                        help="Max GPT review iterations (default: 2)")
    parser.add_argument("--edge-stats", help="Edge stats file for connectivity analysis")
    parser.add_argument("--labels-path", help="Path to neuron labels JSON file")

    # Output
    parser.add_argument("--output-dir", default="neuron_reports/json",
                        help="Output directory for investigation JSONs (default: neuron_reports/json)")
    parser.add_argument("--html-output-dir", default="neuron_reports/html",
                        help="Output directory for HTML dashboards (default: neuron_reports/html)")

    # Polarity options
    parser.add_argument("--both-polarities", action="store_true",
                        help="Run both positive and negative polarity investigations per neuron. "
                             "Each neuron gets two sequential runs on the same GPU.")

    # Options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without submitting jobs")
    parser.add_argument("--job-prefix", default="neuronpi",
                        help="Prefix for SLURM job names")

    args = parser.parse_args()

    # Get neurons to investigate
    neurons = get_neurons_from_file(args.neurons_file)
    source = f"file: {args.neurons_file}"

    if not neurons:
        print("No neurons to investigate!")
        sys.exit(0)

    # Calculate job distribution
    agents_per_gpu = args.agents_per_gpu
    max_gpus = args.max_gpus
    total_neurons = len(neurons)

    # Each GPU handles agents_per_gpu neurons in parallel
    neurons_per_gpu = agents_per_gpu

    # Create batches
    batches = []
    for i in range(0, total_neurons, neurons_per_gpu):
        batch = neurons[i:i + neurons_per_gpu]
        batches.append(batch)

    # Limit to max GPUs - redistribute
    if len(batches) > max_gpus:
        all_neurons = [n for batch in batches for n in batch]
        neurons_per_job = (len(all_neurons) + max_gpus - 1) // max_gpus
        batches = []
        for i in range(0, len(all_neurons), neurons_per_job):
            batches.append(all_neurons[i:i + neurons_per_job])

    print("=" * 60)
    print("SLURM BATCH NEURONPI INVESTIGATION")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Model config: {args.model_config or 'llama-3.1-8b (default)'}")
    print(f"Total neurons: {total_neurons}")
    print(f"Agents per GPU: {agents_per_gpu}")
    print(f"Max GPUs: {max_gpus}")
    print(f"Jobs to submit: {len(batches)}")
    print(f"Max review iterations: {args.max_review_iterations}")
    print(f"Partition: {args.partition}")
    print(f"Time limit: {args.time_limit}")
    print(f"HTML output: {args.html_output_dir}")
    print(f"GPU server port: {args.gpu_server_port}")
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
            print(f"  L{n['layer']}/N{n['neuron']}")
        return

    # Create output directories
    Path(args.html_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_dir = Path(args.output_dir) / f"slurm_neuronpi_{timestamp}"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    print(f"\nSubmitting {len(batches)} jobs...")
    job_ids = []
    for i, batch in enumerate(batches):
        worker_script = create_worker_script(
            neurons=batch,
            job_id=i,
            output_dir=Path(args.output_dir),
            html_output_dir=Path(args.html_output_dir),
            edge_stats=args.edge_stats,
            max_concurrent=args.agents_per_gpu,
            project_root=str(Path.cwd()),
            max_review_iterations=args.max_review_iterations,
            model_config=args.model_config,
            both_polarities=args.both_polarities,
            gpu_server_port=args.gpu_server_port,
            labels_path=args.labels_path,
        )

        slurm_job_id = submit_slurm_job(
            worker_script=worker_script,
            job_id=i,
            job_name=f"{args.job_prefix}-{i}",
            partition=args.partition,
            output_dir=slurm_dir,
            time_limit=args.time_limit,
        )

        if slurm_job_id:
            job_ids.append(slurm_job_id)
            print(f"  Submitted job {i+1}/{len(batches)}: SLURM ID {slurm_job_id}")

    print(f"\nSubmitted {len(job_ids)} jobs.")
    print(f"SLURM scripts: {slurm_dir}")
    print(f"Investigation JSONs: {args.output_dir}")
    print(f"HTML dashboards: {args.html_output_dir}")
    print("\nMonitor with: squeue -u $USER")

    # Save job manifest
    manifest = {
        'timestamp': timestamp,
        'source': source,
        'total_neurons': total_neurons,
        'jobs': len(batches),
        'slurm_job_ids': job_ids,
        'output_dir': str(args.output_dir),
        'html_output_dir': str(args.html_output_dir),
        'neurons': [(n['layer'], n['neuron']) for n in neurons]
    }
    manifest_path = slurm_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

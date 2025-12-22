"""SLURM launcher for harvest pipeline.

Submits harvest jobs to SLURM cluster programmatically.

Usage:
    spd-harvest <wandb_path> --n_batches 1000
    spd-harvest <wandb_path> --n_batches 8000 --n_gpus 8
"""

import subprocess
from datetime import datetime
from pathlib import Path

from spd.log import logger
from spd.settings import DEFAULT_PARTITION_NAME, REPO_ROOT


def _generate_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _submit_slurm_job(script_content: str, script_path: Path) -> str:
    """Write script and submit to SLURM, returning job ID."""
    with open(script_path, "w") as f:
        f.write(script_content)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit SLURM job: {result.stderr}")

    job_id = result.stdout.strip().split()[-1]
    return job_id


def harvest(
    wandb_path: str,
    n_batches: int,
    n_gpus: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "24:00:00",
) -> None:
    """Submit harvest job to SLURM.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process.
        n_gpus: Number of GPUs for distributed harvesting. If None, uses single GPU.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI to include.
        partition: SLURM partition name.
        time: Job time limit.
    """
    job_id = _generate_job_id()
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    sbatch_scripts_dir = Path.home() / "sbatch_scripts"
    sbatch_scripts_dir.mkdir(exist_ok=True)

    gres = f"gpu:{n_gpus}" if n_gpus else "gpu:1"
    job_name = f"harvest-{job_id}"

    # Build the harvest command with all args
    cmd_parts = [
        "python -m spd.harvest.scripts.run_harvest",
        f'"{wandb_path}"',
        f"--n_batches {n_batches}",
        f"--batch_size {batch_size}",
        f"--ci_threshold {ci_threshold}",
        f"--activation_examples_per_component {activation_examples_per_component}",
        f"--activation_context_tokens_per_side {activation_context_tokens_per_side}",
        f"--pmi_token_top_k {pmi_token_top_k}",
    ]
    if n_gpus:
        cmd_parts.append(f"--n_gpus {n_gpus}")

    harvest_cmd = " \\\n    ".join(cmd_parts)

    script_content = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --output={slurm_logs_dir}/slurm-%j.out

set -euo pipefail

echo "=== Harvest ==="
echo "WANDB_PATH: {wandb_path}"
echo "N_BATCHES: {n_batches}"
echo "N_GPUS: {n_gpus or 1}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "==============="

cd {REPO_ROOT}
source .venv/bin/activate

{harvest_cmd}

echo "Harvest complete!"
"""

    script_path = sbatch_scripts_dir / f"harvest_{job_id}.sh"
    slurm_job_id = _submit_slurm_job(script_content, script_path)

    # Rename to include SLURM job ID
    final_script_path = sbatch_scripts_dir / f"harvest_{slurm_job_id}.sh"
    script_path.rename(final_script_path)

    # Create empty log file for tailing
    (slurm_logs_dir / f"slurm-{slurm_job_id}.out").touch()

    logger.section("Harvest job submitted!")
    logger.values(
        {
            "Job ID": slurm_job_id,
            "WandB path": wandb_path,
            "N batches": n_batches,
            "N GPUs": n_gpus or 1,
            "Log": f"~/slurm_logs/slurm-{slurm_job_id}.out",
            "Script": str(final_script_path),
        }
    )

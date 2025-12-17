"""SLURM launcher for autointerp pipeline.

Submits interpret jobs to SLURM cluster programmatically.

Usage:
    spd-interpret <wandb_path>
    spd-interpret <wandb_path> --budget_usd 100
"""

import subprocess
from datetime import datetime
from pathlib import Path

from spd.autointerp.interpret import OpenRouterModelName
from spd.log import logger
from spd.settings import REPO_ROOT


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


def launch_interpret_job(
    wandb_path: str,
    model: OpenRouterModelName,
    partition: str,
    time: str,
) -> None:
    """Submit interpret job to SLURM (CPU-only, IO-bound).

    Args:
        wandb_path: WandB run path for the target decomposition run.
        model: OpenRouter model to use for interpretation.
        partition: SLURM partition name.
        time: Job time limit.
    """
    job_id = _generate_job_id()
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    sbatch_scripts_dir = Path.home() / "sbatch_scripts"
    sbatch_scripts_dir.mkdir(exist_ok=True)

    job_name = f"interpret-{job_id}"

    cmd_parts = [
        "python -m spd.autointerp.scripts.run_interpret",
        f'"{wandb_path}"',
        f"--model {model.value}",
    ]
    interpret_cmd = " \\\n    ".join(cmd_parts)

    script_content = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time={time}
#SBATCH --output={slurm_logs_dir}/slurm-%j.out

set -euo pipefail

echo "=== Interpret ==="
echo "WANDB_PATH: {wandb_path}"
echo "MODEL: {model.value}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "================="

cd {REPO_ROOT}
source .venv/bin/activate

# OPENROUTER_API_KEY should be in .env or environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

{interpret_cmd}

echo "Interpret complete!"
"""

    script_path = sbatch_scripts_dir / f"interpret_{job_id}.sh"
    slurm_job_id = _submit_slurm_job(script_content, script_path)

    # Rename to include SLURM job ID
    final_script_path = sbatch_scripts_dir / f"interpret_{slurm_job_id}.sh"
    script_path.rename(final_script_path)

    # Create empty log file for tailing
    (slurm_logs_dir / f"slurm-{slurm_job_id}.out").touch()

    logger.section("Interpret job submitted!")
    logger.values(
        {
            "Job ID": slurm_job_id,
            "WandB path": wandb_path,
            "Model": model.value,
            "Log": f"~/slurm_logs/slurm-{slurm_job_id}.out",
            "Script": str(final_script_path),
        }
    )

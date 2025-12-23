"""SLURM launcher for autointerp pipeline.

Submits interpret jobs to SLURM cluster programmatically.

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --budget_usd 100
"""

from datetime import datetime

from spd.autointerp.interpret import OpenRouterModelName
from spd.log import logger
from spd.settings import REPO_ROOT, SBATCH_SCRIPTS_DIR, SLURM_LOGS_DIR
from spd.utils.command_utils import submit_slurm_script


def _generate_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def launch_interpret_job(
    wandb_path: str,
    model: OpenRouterModelName,
    partition: str,
    time: str,
    max_examples_per_component: int,
) -> None:
    """Submit interpret job to SLURM (CPU-only, IO-bound).

    Args:
        wandb_path: WandB run path for the target decomposition run.
        model: OpenRouter model to use for interpretation.
        partition: SLURM partition name.
        time: Job time limit.
        max_examples_per_component: Maximum number of activation examples per component.
    """
    job_id = _generate_job_id()
    SLURM_LOGS_DIR.mkdir(exist_ok=True)
    SBATCH_SCRIPTS_DIR.mkdir(exist_ok=True)

    job_name = f"interpret-{job_id}"

    cmd_parts = [
        "python -m spd.autointerp.scripts.run_interpret",
        f'"{wandb_path}"',
        f"--model {model.value}",
        f"--max_examples_per_component {max_examples_per_component}",
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
#SBATCH --output={SLURM_LOGS_DIR}/slurm-%j.out

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

    script_path = SBATCH_SCRIPTS_DIR / f"interpret_{job_id}.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    slurm_job_id = submit_slurm_script(script_path)

    # Rename to include SLURM job ID
    final_script_path = SBATCH_SCRIPTS_DIR / f"interpret_{slurm_job_id}.sh"
    script_path.rename(final_script_path)

    # Create empty log file for tailing
    (SLURM_LOGS_DIR / f"slurm-{slurm_job_id}.out").touch()

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

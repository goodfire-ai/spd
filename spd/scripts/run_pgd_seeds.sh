#!/bin/bash
#SBATCH --job-name=pgd-seeds-2hours
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=~/slurm_logs/pgd-seeds-%A_%a.out
#SBATCH --array=0-15

set -euo pipefail

mkdir -p ~/slurm_logs

cd /mnt/polished-lake/home/oli/spd
source /mnt/polished-lake/home/oli/spd/.venv/bin/activate

SWEEP_ID=$(date +%Y%m%d_%H%M%S)

python -u spd/pgd_saturation_test.py --seed "${SLURM_ARRAY_TASK_ID}" --sweep_id "${SWEEP_ID}"

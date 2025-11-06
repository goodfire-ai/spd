#!/bin/bash
#SBATCH --job-name=pgd-seeds-2hours
#SBATCH --partition=h200-reserved
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/pgd-seeds-%A_%a.out
#SBATCH --array=0-15

set -euo pipefail

mkdir -p ~/slurm_logs

cd /mnt/polished-lake/home/oli/spd
source /mnt/polished-lake/home/oli/spd/.venv/bin/activate

python -u spd/pgd_saturation_test.py

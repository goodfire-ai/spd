#!/bin/bash
#SBATCH --job-name=pgd-broadcast-cmp
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/pgd_broadcast_cmp_%j.out

source /mnt/polished-lake/home/braun/spd/.venv/bin/activate
cd /mnt/polished-lake/home/braun/spd

python scripts/eval_pgd_broadcast_comparison.py

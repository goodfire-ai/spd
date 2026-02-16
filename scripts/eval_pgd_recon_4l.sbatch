#!/bin/bash
#SBATCH --job-name=pgd-recon-eval-4l
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/pgd_recon_eval_4l_%j.out

source /mnt/polished-lake/home/braun/spd/.venv/bin/activate
cd /mnt/polished-lake/home/braun/spd

PYTHONUNBUFFERED=1 python scripts/eval_pgd_recon_4l.py

#!/bin/bash
#SBATCH --job-name=pgd-recon-1gpu
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/pgd_recon_eval_1gpu_%j.out

source /mnt/polished-lake/home/braun/spd/.venv/bin/activate
cd /mnt/polished-lake/home/braun/spd

python scripts/eval_pgd_recon.py

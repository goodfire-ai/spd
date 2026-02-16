#!/bin/bash
#SBATCH --job-name=pgd-recon-trainpath
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/pgd_recon_eval_trainpath_%j.out

source /mnt/polished-lake/home/braun/spd/.venv/bin/activate
cd /mnt/polished-lake/home/braun/spd

torchrun --nproc_per_node=8 --standalone scripts/eval_pgd_recon_training_path.py

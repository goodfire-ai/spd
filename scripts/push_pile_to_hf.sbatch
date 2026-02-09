#!/bin/bash
#SBATCH --job-name=push-pile-hf-6h
#SBATCH --partition=h200-reserved
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

source /mnt/polished-lake/home/braun/spd-main/.venv/bin/activate
python /mnt/polished-lake/home/braun/spd-main/scripts/push_pile_to_hf.py

#!/bin/bash
#SBATCH --job-name=precache-pile-7h
#SBATCH --partition=h200-reserved
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

source /mnt/polished-lake/home/braun/spd-main/.venv/bin/activate
python /mnt/polished-lake/home/braun/spd-main/scripts/precache_pile_dataset.py

#!/bin/bash
#SBATCH --job-name=push-pile-tok
#SBATCH --partition=h200-reserved
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --output=%x-%j.out

source /mnt/polished-lake/home/braun/spd-main/.venv/bin/activate
python /mnt/polished-lake/home/braun/spd-main/scripts/push_pile_tokenized_to_hf.py

#!/bin/bash
#SBATCH --job-name=analyze-sft-v4
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_sft_v4_%j.out
#SBATCH --error=logs/analyze_sft_v4_%j.err

mkdir -p logs outputs/medical_sft_v4

cd /mnt/polished-lake/home/ctigges/code/attribution-graphs
source observatory_repo/.env

echo "Starting SFT v4 contrastive analysis at $(date)"

.venv/bin/python scripts/analyze.py \
    "The primary hormone involved in the regulation of blood pressure and volume is" \
    --answer-prefix " Answer:" \
    --contrastive " Ald" " Ren" \
    --output outputs/medical_sft_v4/ \
    --tau 0.15 \
    --model checkpoints/sft_raas_v4

echo "Analysis completed at $(date)"

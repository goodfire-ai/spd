#!/bin/bash
#SBATCH --job-name=analyze-sft-raas
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_sft_raas_%j.out
#SBATCH --error=logs/analyze_sft_raas_%j.err

mkdir -p logs outputs/medical_sft

cd /mnt/polished-lake/home/ctigges/code/attribution-graphs

# Source environment for neurondb
source observatory_repo/.env

echo "Starting SFT model contrastive analysis at $(date)"

# Contrastive attribution: Aldosterone vs Renin on SFT v2 model
.venv/bin/python scripts/analyze.py \
    "The primary hormone involved in the regulation of blood pressure and volume is" \
    --answer-prefix " Answer:" \
    --contrastive " Ald" " Ren" \
    --output outputs/medical_sft/ \
    --tau 0.05 \
    --model checkpoints/sft_raas_v2

echo "Analysis completed at $(date)"

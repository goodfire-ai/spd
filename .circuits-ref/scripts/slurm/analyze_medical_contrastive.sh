#!/bin/bash
#SBATCH --job-name=contrastive-medical
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/contrastive_medical_%j.out
#SBATCH --error=logs/contrastive_medical_%j.err

cd /mnt/polished-lake/home/ctigges/code/attribution-graphs

# Set up environment
export PYTHONPATH="observatory_repo/lib/neurondb:observatory_repo/lib/util:$PYTHONPATH"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env: PG_HOST=$PG_HOST"
fi

echo "Starting contrastive attribution analysis at $(date)"
echo "================================================"

mkdir -p outputs/medical_contrastive

# Case 2: Blood pressure hormone - Aldosterone (correct) vs Renin (wrong)
# Contrastive: trace logit(Ald) - logit(Ren)
# Using --answer-prefix to match how we filtered (forces direct medical term completion)
echo ""
echo "=== Case 2: Blood pressure hormone ==="
echo "Contrastive: Aldosterone (correct) vs Renin (wrong)"
.venv/bin/python scripts/analyze.py \
    "The primary hormone involved in the regulation of blood pressure and volume is" \
    --answer-prefix " Answer:" \
    --contrastive " Ald" " Ren" \
    --output outputs/medical_contrastive/ \
    --tau 0.05

# Case 3: Opioid side effect - Constipation (correct) vs Dependence (wrong)
# Contrastive: trace logit(Const) - logit(Dep)
echo ""
echo "=== Case 3: Opioid side effect ==="
echo "Contrastive: Constipation (correct) vs Dependence (wrong)"
.venv/bin/python scripts/analyze.py \
    "A common side effect of opioid analgesics is" \
    --answer-prefix " Answer:" \
    --contrastive " Const" " Dep" \
    --output outputs/medical_contrastive/ \
    --tau 0.05

# Case 1: Ground-glass hepatocytes - Hepatitis (correct) vs Alpha (wrong)
# Contrastive: trace logit(Hep) - logit(Alpha)
echo ""
echo "=== Case 1: Ground-glass hepatocytes ==="
echo "Contrastive: Hepatitis B (correct) vs Alpha (wrong)"
.venv/bin/python scripts/analyze.py \
    "The histopathological feature known as 'ground-glass hepatocytes' is characteristic of" \
    --answer-prefix " Answer:" \
    --contrastive " Hep" " Alpha" \
    --output outputs/medical_contrastive/ \
    --tau 0.05

echo ""
echo "================================================"
echo "Contrastive analysis completed at $(date)"
echo "Output directory: outputs/medical_contrastive/"

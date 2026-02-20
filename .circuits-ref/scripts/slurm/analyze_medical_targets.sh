#!/bin/bash
#SBATCH --job-name=analyze-medical
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_medical_%j.out
#SBATCH --error=logs/analyze_medical_%j.err

cd /mnt/polished-lake/home/ctigges/code/attribution-graphs

# Set up environment
export PYTHONPATH="observatory_repo/lib/neurondb:observatory_repo/lib/util:$PYTHONPATH"

# Load environment variables (API keys, database connection)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env: PG_HOST=$PG_HOST"
fi

echo "Starting medical circuit analysis at $(date)"
echo "================================================"

# Create output directory
mkdir -p outputs/medical_wrong_predictions

# Target Case 1: Ground-glass hepatocytes (Alpha vs Hepatitis B)
# Model predicts Alpha-1 antitrypsin deficiency, but correct is Hepatitis B
echo ""
echo "=== Case 1: Ground-glass hepatocytes ==="
echo "Expected: hepatitis B, Model predicts: Alpha"
.venv/bin/python scripts/analyze.py \
    "The histopathological feature known as 'ground-glass hepatocytes' is characteristic of" \
    --output outputs/medical_wrong_predictions/ \
    --k 5 \
    --tau 0.005

# Target Case 2: Blood pressure/volume hormone (Renin vs Aldosterone)
# Model predicts Renin, but correct is Aldosterone
echo ""
echo "=== Case 2: Blood pressure hormone ==="
echo "Expected: aldosterone, Model predicts: Renin"
.venv/bin/python scripts/analyze.py \
    "The primary hormone involved in the regulation of blood pressure and volume is" \
    --output outputs/medical_wrong_predictions/ \
    --k 5 \
    --tau 0.005

# Target Case 3: Opioid side effect (Dependence vs Constipation)
# Model predicts Dependence, but correct is Constipation (most "common")
echo ""
echo "=== Case 3: Opioid side effect ==="
echo "Expected: constipation, Model predicts: Dependence"
.venv/bin/python scripts/analyze.py \
    "A common side effect of opioid analgesics is" \
    --output outputs/medical_wrong_predictions/ \
    --k 5 \
    --tau 0.005

echo ""
echo "================================================"
echo "Analysis completed at $(date)"
echo "Output directory: outputs/medical_wrong_predictions/"

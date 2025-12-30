#!/bin/bash
# Submit tokenization test on a single exclusive node
# Usage: ./submit_single_node_test.sh <n_gpus>  (1 or 8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N_GPUS=${1:-1}
PARTITION="h200-reserved"
JOB_NAME="tok_single_${N_GPUS}gpu"
SCRIPT_PATH="${SCRIPT_DIR}/phase4b_tokenization_scaling.py"

echo "=== Single Node Tokenization Test ==="
echo "GPUs: $N_GPUS"
echo ""

mkdir -p ~/slurm_logs ~/sbatch_scripts

MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

SLURM_SCRIPT=$(mktemp /tmp/slurm_single_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "=== SLURM Job Info ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "GPUs: ${N_GPUS}"
echo ""

torchrun \\
    --standalone \\
    --nproc_per_node=${N_GPUS} \\
    ${SCRIPT_PATH}

echo ""
echo "=== Job Complete ==="
EOF

chmod +x "$SLURM_SCRIPT"

echo "Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

echo ""
echo "Submitted job: $JOB_ID"
echo "Log file: ~/slurm_logs/slurm-${JOB_ID}.out"
echo ""
echo "To monitor: tail -f ~/slurm_logs/slurm-${JOB_ID}.out"

mv "$SLURM_SCRIPT" ~/sbatch_scripts/single_${N_GPUS}gpu_${JOB_ID}.sh

#!/bin/bash
# Submit tokenization scaling test with different GPU counts
#
# Usage: ./submit_scaling_test.sh <n_gpus>
#   n_gpus: 1, 8, or 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N_GPUS=${1:-1}
PARTITION="h200-reserved"
JOB_NAME="tokenize_scale_${N_GPUS}gpu"
SCRIPT_PATH="${SCRIPT_DIR}/phase4b_tokenization_scaling.py"

# Calculate nodes and gpus per node
if [[ $N_GPUS -eq 1 ]]; then
    N_NODES=1
    GPUS_PER_NODE=1
elif [[ $N_GPUS -le 8 ]]; then
    N_NODES=1
    GPUS_PER_NODE=$N_GPUS
else
    N_NODES=2
    GPUS_PER_NODE=8
fi

echo "=== Tokenization Scaling Test ==="
echo "GPUs: $N_GPUS"
echo "Nodes: $N_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo ""

# Create log directory
mkdir -p ~/slurm_logs

# Generate a unique port
MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

# Create SLURM script
SLURM_SCRIPT=$(mktemp /tmp/slurm_scale_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks=${N_NODES}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --time=00:30:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "=== SLURM Job Info ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Nodes: \$SLURM_JOB_NODELIST"
echo "GPUs: ${N_GPUS}"
echo ""

srun --cpus-per-task=128 bash -c 'torchrun \\
    --nnodes=${N_NODES} \\
    --node_rank=\$SLURM_PROCID \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --master_addr=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1) \\
    --master_port=${MASTER_PORT} \\
    ${SCRIPT_PATH}'

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
echo "To monitor:"
echo "  tail -f ~/slurm_logs/slurm-${JOB_ID}.out"

mkdir -p ~/sbatch_scripts
mv "$SLURM_SCRIPT" ~/sbatch_scripts/scale_${N_GPUS}gpu_${JOB_ID}.sh

#!/bin/bash
# Submit optimized dataloader test on 2 nodes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N_NODES=2
GPUS_PER_NODE=8
PARTITION="h200-reserved"
JOB_NAME="opt_dataloader_2node"
SCRIPT_PATH="${SCRIPT_DIR}/test_optimized_dataloader.py"

echo "=== Optimized DataLoader Test (2 nodes) ==="
echo "Nodes: $N_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo ""

mkdir -p ~/slurm_logs ~/sbatch_scripts

MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

SLURM_SCRIPT=$(mktemp /tmp/slurm_opt2_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks=${N_NODES}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "=== SLURM Job Info ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Nodes: \$SLURM_JOB_NODELIST"
echo ""

srun --cpus-per-task=64 bash -c 'torchrun \\
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
echo "To monitor: tail -f ~/slurm_logs/slurm-${JOB_ID}.out"

mv "$SLURM_SCRIPT" ~/sbatch_scripts/opt_2node_${JOB_ID}.sh

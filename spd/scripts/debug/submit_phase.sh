#!/bin/bash
# Submit a debug phase script to SLURM
#
# Usage: ./submit_phase.sh <phase_number> [n_nodes]
#   phase_number: 1-5
#   n_nodes: number of nodes (default: 2)
#
# Example: ./submit_phase.sh 1 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PHASE=${1:-1}
N_NODES=${2:-2}
GPUS_PER_NODE=8
TOTAL_GPUS=$((N_NODES * GPUS_PER_NODE))
PARTITION="h200-reserved"
JOB_NAME="debug_phase${PHASE}"

# Map phase number to script
case $PHASE in
    1) SCRIPT="phase1_minimal.py" ;;
    2) SCRIPT="phase2_spd_init.py" ;;
    3) SCRIPT="phase3_model_loading.py" ;;
    4) SCRIPT="phase4_dataset.py" ;;
    5) SCRIPT="phase5_full_init.py" ;;
    *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "=== Debug Phase $PHASE ==="
echo "Script: $SCRIPT"
echo "Nodes: $N_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo ""

# Create log directory
mkdir -p ~/slurm_logs

# Generate a unique port based on timestamp
MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

# Create SLURM script
SLURM_SCRIPT=$(mktemp /tmp/slurm_debug_XXXXXX.sh)

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

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "=== SLURM Job Info ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Nodes: \$SLURM_JOB_NODELIST"
echo "Tasks: \$SLURM_NTASKS"
echo ""

# Multi-node torchrun via srun
# Each node runs torchrun with 8 processes, SLURM_PROCID gives node rank
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
echo ""
echo "To check status:"
echo "  squeue -j ${JOB_ID}"

# Save script path for reference
mv "$SLURM_SCRIPT" ~/sbatch_scripts/debug_phase${PHASE}_${JOB_ID}.sh
echo ""
echo "Script saved to: ~/sbatch_scripts/debug_phase${PHASE}_${JOB_ID}.sh"

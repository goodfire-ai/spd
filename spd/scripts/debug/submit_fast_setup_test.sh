#!/bin/bash
# Test the optimized setup (no uv sync)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N_GPUS=${1:-8}
PARTITION="h200-reserved"
JOB_NAME="fast_setup_${N_GPUS}gpu"

echo "=== Fast Setup Test (no uv sync) ==="
echo "GPUs: $N_GPUS"
echo ""

mkdir -p ~/slurm_logs ~/sbatch_scripts

MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

SLURM_SCRIPT=$(mktemp /tmp/slurm_fast_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << OUTER_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out

set -euo pipefail

echo "=== SLURM Job Started ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo ""

TOTAL_START=\$(date +%s.%N)

echo "[\$(date +%s.%N)] Step 1: Creating workspace..."
WORK_DIR="\$HOME/slurm_workspaces/fast-test-\$SLURM_JOB_ID"
mkdir -p "\$WORK_DIR"
trap 'rm -rf "\$WORK_DIR"' EXIT

echo "[\$(date +%s.%N)] Step 2: Git clone..."
CLONE_START=\$(date +%s.%N)
git clone "${REPO_ROOT}" "\$WORK_DIR" --quiet
CLONE_END=\$(date +%s.%N)
echo "[\$(date +%s.%N)] Git clone done: \$(echo "\$CLONE_END - \$CLONE_START" | bc)s"

cd "\$WORK_DIR"

[ -f "${REPO_ROOT}/.env" ] && cp "${REPO_ROOT}/.env" .env

echo "[\$(date +%s.%N)] Step 3: Activate existing venv (no uv sync!)..."
VENV_START=\$(date +%s.%N)
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="\$WORK_DIR:\${PYTHONPATH:-}"
VENV_END=\$(date +%s.%N)
echo "[\$(date +%s.%N)] Venv activated: \$(echo "\$VENV_END - \$VENV_START" | bc)s"

echo ""
echo "[\$(date +%s.%N)] Step 4: torchrun starting..."
TORCHRUN_START=\$(date +%s.%N)
torchrun \\
    --standalone \\
    --nproc_per_node=${N_GPUS} \\
    spd/scripts/debug/test_fast_setup.py
TORCHRUN_END=\$(date +%s.%N)

TOTAL_END=\$(date +%s.%N)
echo ""
echo "=== Timing Summary ==="
echo "Git clone:  \$(echo "\$CLONE_END - \$CLONE_START" | bc)s"
echo "Venv setup: \$(echo "\$VENV_END - \$VENV_START" | bc)s (was ~60s with uv sync!)"
echo "torchrun:   \$(echo "\$TORCHRUN_END - \$TORCHRUN_START" | bc)s"
echo "TOTAL:      \$(echo "\$TOTAL_END - \$TOTAL_START" | bc)s"
echo ""
echo "=== Job Complete ==="
OUTER_EOF

chmod +x "$SLURM_SCRIPT"

echo "Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

echo ""
echo "Submitted job: $JOB_ID"
echo "Log file: ~/slurm_logs/slurm-${JOB_ID}.out"
echo ""
echo "To monitor: tail -f ~/slurm_logs/slurm-${JOB_ID}.out"

mv "$SLURM_SCRIPT" ~/sbatch_scripts/fast_setup_${N_GPUS}gpu_${JOB_ID}.sh

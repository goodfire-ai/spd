#!/bin/bash
# Submit test that mimics full spd-run setup with git clone + uv sync
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

N_GPUS=${1:-8}
PARTITION="h200-reserved"
JOB_NAME="full_setup_test_${N_GPUS}gpu"

echo "=== Full Setup Timing Test ==="
echo "GPUs: $N_GPUS"
echo "This test mimics what spd-run does: git clone + uv sync + torchrun"
echo ""

mkdir -p ~/slurm_logs ~/sbatch_scripts

MASTER_PORT=$((20000 + ($(date +%s) % 20000)))

SLURM_SCRIPT=$(mktemp /tmp/slurm_full_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << 'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:NGPUS_PLACEHOLDER
#SBATCH --time=00:30:00
#SBATCH --output=LOGS_PLACEHOLDER/slurm-%j.out

set -euo pipefail

echo "=== SLURM Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo ""

# Time everything
TOTAL_START=$(date +%s.%N)

echo "[$(date +%s.%N)] Step 1: Creating workspace..."
WORK_DIR="$HOME/slurm_workspaces/test-setup-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT
echo "[$(date +%s.%N)] Workspace created: $WORK_DIR"

echo ""
echo "[$(date +%s.%N)] Step 2: Git clone..."
CLONE_START=$(date +%s.%N)
git clone "REPO_ROOT_PLACEHOLDER" "$WORK_DIR" --quiet
CLONE_END=$(date +%s.%N)
echo "[$(date +%s.%N)] Git clone done: $(echo "$CLONE_END - $CLONE_START" | bc)s"

cd "$WORK_DIR"

echo ""
echo "[$(date +%s.%N)] Step 3: Copy .env..."
[ -f "REPO_ROOT_PLACEHOLDER/.env" ] && cp "REPO_ROOT_PLACEHOLDER/.env" .env
echo "[$(date +%s.%N)] .env copied"

echo ""
echo "[$(date +%s.%N)] Step 4: uv sync..."
UV_START=$(date +%s.%N)
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
UV_END=$(date +%s.%N)
echo "[$(date +%s.%N)] uv sync done: $(echo "$UV_END - $UV_START" | bc)s"

source .venv/bin/activate

echo ""
echo "[$(date +%s.%N)] Step 5: torchrun starting..."
TORCHRUN_START=$(date +%s.%N)
torchrun \
    --standalone \
    --nproc_per_node=NGPUS_PLACEHOLDER \
    spd/scripts/debug/test_full_setup_timing.py
TORCHRUN_END=$(date +%s.%N)
echo "[$(date +%s.%N)] torchrun done: $(echo "$TORCHRUN_END - $TORCHRUN_START" | bc)s"

TOTAL_END=$(date +%s.%N)
echo ""
echo "=== Timing Summary ==="
echo "Git clone: $(echo "$CLONE_END - $CLONE_START" | bc)s"
echo "uv sync:   $(echo "$UV_END - $UV_START" | bc)s"
echo "torchrun:  $(echo "$TORCHRUN_END - $TORCHRUN_START" | bc)s"
echo "TOTAL:     $(echo "$TOTAL_END - $TOTAL_START" | bc)s"
echo ""
echo "=== Job Complete ==="
OUTER_EOF

# Replace placeholders
sed -i "s|JOBNAME_PLACEHOLDER|${JOB_NAME}|g" "$SLURM_SCRIPT"
sed -i "s|PARTITION_PLACEHOLDER|${PARTITION}|g" "$SLURM_SCRIPT"
sed -i "s|NGPUS_PLACEHOLDER|${N_GPUS}|g" "$SLURM_SCRIPT"
sed -i "s|REPO_ROOT_PLACEHOLDER|${REPO_ROOT}|g" "$SLURM_SCRIPT"
sed -i "s|LOGS_PLACEHOLDER|$HOME/slurm_logs|g" "$SLURM_SCRIPT"

chmod +x "$SLURM_SCRIPT"

echo "Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

echo ""
echo "Submitted job: $JOB_ID"
echo "Log file: ~/slurm_logs/slurm-${JOB_ID}.out"
echo ""
echo "To monitor: tail -f ~/slurm_logs/slurm-${JOB_ID}.out"

mv "$SLURM_SCRIPT" ~/sbatch_scripts/full_setup_${N_GPUS}gpu_${JOB_ID}.sh

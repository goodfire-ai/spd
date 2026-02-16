#!/bin/bash
#SBATCH --job-name=pgd-recon-snapshot
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/pgd_recon_eval_snapshot_%j.out

# Run from a temporary clone at the snapshot branch so we use EXACTLY the old code
TMPDIR=$(mktemp -d)
git clone --depth 1 --branch snapshot/run_20260122_123532 /mnt/polished-lake/home/braun/spd "$TMPDIR/spd"
cd "$TMPDIR/spd"

# Use the main repo's venv but install this snapshot's package
source /mnt/polished-lake/home/braun/spd/.venv/bin/activate
uv pip install -e "$TMPDIR/spd" --quiet

# Copy the eval script into the snapshot
cp /mnt/polished-lake/home/braun/spd/scripts/eval_pgd_recon_training_path.py "$TMPDIR/spd/scripts/"

torchrun --nproc_per_node=8 --standalone scripts/eval_pgd_recon_training_path.py

# Cleanup
rm -rf "$TMPDIR"

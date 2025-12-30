#!/bin/bash
# Test uv sync timing to see if it's the bottleneck
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Testing uv sync timing ==="
echo ""

# Create a test workspace
TEST_WORKSPACE="/tmp/test_uv_sync_$$"
mkdir -p "$TEST_WORKSPACE"
trap 'rm -rf "$TEST_WORKSPACE"' EXIT

echo "Step 1: Clone repository..."
START=$(date +%s.%N)
git clone "$REPO_ROOT" "$TEST_WORKSPACE/repo" --quiet
END=$(date +%s.%N)
echo "  git clone: $(echo "$END - $START" | bc) seconds"

cd "$TEST_WORKSPACE/repo"

echo ""
echo "Step 2: uv sync (cold cache)..."
START=$(date +%s.%N)
# Simulate what the SLURM script does
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
END=$(date +%s.%N)
echo "  uv sync: $(echo "$END - $START" | bc) seconds"

echo ""
echo "=== Done ==="

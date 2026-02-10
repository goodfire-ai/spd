#!/bin/bash
#SBATCH --job-name=eval-debug
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/slurm-%j.out

set -euo pipefail
umask 002

REPO_ROOT="/mnt/polished-lake/home/braun/spd"
CONFIGS_DIR="${REPO_ROOT}/eval_debug/configs"
RESULTS_FILE="${REPO_ROOT}/eval_debug/results.txt"

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "========================================" | tee "${RESULTS_FILE}"
echo "Eval Debug Tests - $(date)" | tee -a "${RESULTS_FILE}"
echo "Host: $(hostname)" | tee -a "${RESULTS_FILE}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"

# Track results for summary
declare -A TIMES
declare -A STATUSES

for config in $(ls "${CONFIGS_DIR}"/*.yaml | sort); do
    name=$(basename "${config}" .yaml)

    echo "" | tee -a "${RESULTS_FILE}"
    echo "========================================" | tee -a "${RESULTS_FILE}"
    echo "Starting test: ${name}" | tee -a "${RESULTS_FILE}"
    echo "Time: $(date)" | tee -a "${RESULTS_FILE}"
    echo "Config: ${config}" | tee -a "${RESULTS_FILE}"
    echo "========================================" | tee -a "${RESULTS_FILE}"

    # Monitor memory in background
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5 > "/tmp/gpu_mem_${name}.log" 2>&1 &
    MEM_PID=$!

    START_TIME=$SECONDS

    # Run with torchrun (8 GPUs) and capture exit code
    # Use timeout of 15 minutes per test to avoid hanging
    set +e
    timeout 900 python -m torch.distributed.run \
        --standalone \
        --nproc_per_node 8 \
        "${REPO_ROOT}/spd/experiments/lm/lm_decomposition.py" \
        "${config}" 2>&1 | tee -a "${RESULTS_FILE}"
    EXIT_CODE=$?
    set -e

    ELAPSED=$((SECONDS - START_TIME))

    # Stop memory monitor
    kill ${MEM_PID} 2>/dev/null || true
    wait ${MEM_PID} 2>/dev/null || true

    # Get peak GPU memory
    if [ -f "/tmp/gpu_mem_${name}.log" ]; then
        PEAK_MEM=$(grep -v "memory" "/tmp/gpu_mem_${name}.log" | awk -F',' '{print $1}' | sed 's/ MiB//' | sort -n | tail -1)
        rm -f "/tmp/gpu_mem_${name}.log"
    else
        PEAK_MEM="unknown"
    fi

    if [ ${EXIT_CODE} -eq 0 ]; then
        STATUS="OK"
    elif [ ${EXIT_CODE} -eq 124 ]; then
        STATUS="TIMEOUT(15min)"
    elif [ ${EXIT_CODE} -eq 137 ]; then
        STATUS="OOM_KILLED"
    else
        STATUS="FAILED(rc=${EXIT_CODE})"
    fi

    TIMES[${name}]=${ELAPSED}
    STATUSES[${name}]="${STATUS}"

    echo "" | tee -a "${RESULTS_FILE}"
    echo ">>> ${name}: ${ELAPSED}s [${STATUS}] peak_gpu_mem=${PEAK_MEM}MiB" | tee -a "${RESULTS_FILE}"
    echo "" | tee -a "${RESULTS_FILE}"
done

# Print summary
echo "" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "SUMMARY" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
for name in $(echo "${!TIMES[@]}" | tr ' ' '\n' | sort); do
    printf "  %-30s %6ss  [%s]\n" "${name}" "${TIMES[${name}]}" "${STATUSES[${name}]}" | tee -a "${RESULTS_FILE}"
done
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "Done at $(date)" | tee -a "${RESULTS_FILE}"

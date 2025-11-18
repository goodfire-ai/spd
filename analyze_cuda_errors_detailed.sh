#!/bin/bash

# Detailed CUDA Error Analysis - Creates a table of all failures
# Classifies errors and extracts key information

set -euo pipefail

echo "=========================================="
echo "Detailed CUDA Error Analysis"
echo "=========================================="
echo ""

# Define user directories to check
USER_DIRS=(
    "/mnt/polished-lake/home/oli/slurm_logs"
    "/mnt/polished-lake/home/lucius/slurm_logs"
)

# Error patterns to search for (excluding OOM errors and API key issues)
PATTERNS="device-side assert triggered|NCCL WARN Cuda failure|illegal memory access|misaligned address|RuntimeError.*CUDA|cuda runtime error|CUDA error|unspecified launch failure|CUDA kernel errors|cudaErrorIllegalAddress|cudaErrorAssert|CUBLAS_STATUS|CUDNN_STATUS_EXECUTION_FAILED|NCCL error|collective operation failed|ProcessGroupNCCL.*terminated with exception"

# Function to check if error is just API/network issue
is_api_error_only() {
    local file=$1
    # Check if file only has API key errors and no real CUDA errors
    if sudo rg -q "verifying the API key" "$file" 2>/dev/null; then
        if ! sudo rg -q "illegal memory access|misaligned address|device-side assert|NCCL WARN Cuda failure" "$file" 2>/dev/null; then
            return 0  # True - only API error
        fi
    fi
    return 1  # False - has real CUDA errors
}

# Time filter: last 2 weeks (14 days)
TIME_FILTER="-mtime -14"

echo "Searching for CUDA errors in user logs (last 2 weeks)..."
echo ""

# Print table header
printf "%-12s | %-10s | %-30s | %-25s | %-10s\n" "Job ID" "User" "Error Category" "Node" "GPU Rank"
printf "%s\n" "------------------------------------------------------------------------------------------------------------------------------------------------"

# Function to classify error
classify_error() {
    local file=$1

    if sudo rg -q "illegal memory access" "$file" 2>/dev/null; then
        echo "Illegal Memory Access"
    elif sudo rg -q "misaligned address" "$file" 2>/dev/null; then
        echo "Misaligned Address"
    elif sudo rg -q "device-side assert triggered" "$file" 2>/dev/null; then
        echo "Device-Side Assert"
    elif sudo rg -q "NCCL.*collective operation failed" "$file" 2>/dev/null; then
        echo "NCCL Collective Failed"
    elif sudo rg -q "NCCL.*error" "$file" 2>/dev/null; then
        echo "NCCL Error"
    elif sudo rg -q "ProcessGroupNCCL.*terminated" "$file" 2>/dev/null; then
        echo "ProcessGroup Terminated"
    elif sudo rg -q "Expected all tensors.*same device" "$file" 2>/dev/null; then
        echo "Device Placement Error"
    elif sudo rg -q "CUBLAS_STATUS" "$file" 2>/dev/null; then
        echo "cuBLAS Error"
    elif sudo rg -q "CUDNN_STATUS_EXECUTION_FAILED" "$file" 2>/dev/null; then
        echo "cuDNN Execution Failed"
    else
        echo "Other CUDA Error"
    fi
}

# Function to get affected rank
get_rank() {
    local file=$1

    # Try NCCL format [4]
    local rank=$(sudo rg -o "\[[0-9]+\].*NCCL WARN Cuda" "$file" 2>/dev/null | \
                 grep -oP "\[[0-9]+\]" | sed 's/\[//;s/\]//' | sort -u | head -1)

    if [ -z "$rank" ]; then
        # Try torch format [rank4]
        rank=$(sudo rg -o "\[rank[0-9]+\].*CUDA error|\[rank[0-9]+\].*torch.AcceleratorError" "$file" 2>/dev/null | \
               grep -oP "\[rank[0-9]+\]" | sed 's/\[rank//;s/\]//' | sort -u | head -1)
    fi

    if [ -n "$rank" ]; then
        echo "$rank"
    else
        echo "-"
    fi
}

for user_dir in "${USER_DIRS[@]}"; do
    username=$(basename $(dirname "$user_dir"))

    if [ ! -d "$user_dir" ]; then
        continue
    fi

    # Find files with CUDA errors (last 2 weeks only)
    recent_files=$(sudo find "$user_dir" -name "slurm-*.out" -type f $TIME_FILTER 2>/dev/null || true)

    if [ -z "$recent_files" ]; then
        continue
    fi

    affected_files=$(echo "$recent_files" | xargs -r sudo rg -l "$PATTERNS" 2>/dev/null || true)

    if [ -z "$affected_files" ]; then
        continue
    fi

    # Process each affected file
    echo "$affected_files" | while read -r file; do
        # Skip if it's only an API error
        if is_api_error_only "$file"; then
            continue
        fi

        job_id=$(basename "$file" | grep -oP "slurm-\K[0-9]+")

        if [ -z "$job_id" ]; then
            continue
        fi

        # Get node from SLURM metadata
        node=$(sacct -j "$job_id" --format=NodeList --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
        if [ -z "$node" ] || [ "$node" == "None" ]; then
            node="-"
        fi

        # Classify the error
        error_category=$(classify_error "$file")

        # Get affected rank
        rank=$(get_rank "$file")

        # Print table row
        printf "%-12s | %-10s | %-30s | %-25s | %-10s\n" "$job_id" "$username" "$error_category" "$node" "$rank"
    done
done

echo ""
echo "Analysis complete."

h200-dev-145-040

# CUDA Error Analysis - oli (Last 2 Weeks)

## Summary

Total failed jobs: 7
Hardware-related errors: 6 (illegal memory access, misaligned address, device-side assert)
Network/timeout errors: 1 (NCCL collective timeout)
Errors by Node
h200-dev-145-040 (6 jobs)
All errors occurred on GPU rank 4:
Job ID	Error Type	Rank
28201	Misaligned Address	4
28170	Illegal Memory Access	4
28169	Illegal Memory Access	4
14945	Device-Side Assert	4
14742	Illegal Memory Access	4
Pattern: 5 hardware-related memory errors, all on rank 4
h200-reserved-145-003 (1 job)
Job ID	Error Type	Ranks
2760	NCCL Collective Timeout	3,4,5
Pattern: Network/communication timeout across multiple ranks
Conclusion
Node h200-dev-145-040 at GPU rank 4 shows consistent hardware-related CUDA errors (illegal memory access, misaligned address, device-side assert). This pattern strongly suggests a faulty GPU at that specific rank on that node.

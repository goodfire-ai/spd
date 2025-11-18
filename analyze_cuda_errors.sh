#!/bin/bash

# Script to analyze CUDA errors across SLURM logs
# Identifies hardware failures by node and GPU rank

set -euo pipefail

echo "=========================================="
echo "CUDA Error Analysis Across SLURM Logs"
echo "=========================================="
echo ""

# Define user directories to check
USER_DIRS=(
    "/mnt/polished-lake/home/oli/slurm_logs"
    "/mnt/polished-lake/home/lucius/slurm_logs"
)

# Error patterns to search for (excluding OOM errors)
PATTERNS="device-side assert triggered|NCCL WARN Cuda failure|illegal memory access|misaligned address|RuntimeError.*CUDA|cuda runtime error|CUDA error|unspecified launch failure|CUDA kernel errors|cudaErrorIllegalAddress|cudaErrorAssert|CUBLAS_STATUS|CUDNN_STATUS_EXECUTION_FAILED|NCCL error|collective operation failed|ProcessGroupNCCL.*terminated with exception"

# Time filter: last 2 weeks (14 days)
TIME_FILTER="-mtime -14"

echo "Searching for CUDA errors in user logs (last 2 weeks)..."
echo ""

for user_dir in "${USER_DIRS[@]}"; do
    username=$(basename $(dirname "$user_dir"))
    echo "=========================================="
    echo "User: $username"
    echo "Directory: $user_dir"
    echo "=========================================="

    if [ ! -d "$user_dir" ]; then
        echo "Directory not found, skipping..."
        echo ""
        continue
    fi

    # Count total log files (last 2 weeks)
    total_files=$(sudo find "$user_dir" -name "slurm-*.out" -type f $TIME_FILTER 2>/dev/null | wc -l)
    echo "Total log files (last 2 weeks): $total_files"

    # Find files with CUDA errors (last 2 weeks only)
    recent_files=$(sudo find "$user_dir" -name "slurm-*.out" -type f $TIME_FILTER 2>/dev/null || true)

    if [ -z "$recent_files" ]; then
        affected_files=""
    else
        affected_files=$(echo "$recent_files" | xargs -r sudo rg -l "$PATTERNS" 2>/dev/null || true)
    fi

    if [ -z "$affected_files" ]; then
        echo "Files with CUDA errors: 0"
    else
        num_affected=$(echo "$affected_files" | wc -l)
        echo "Files with CUDA errors: $num_affected"
        echo ""
        echo "Failed jobs:"
        echo "$affected_files" | while read -r file; do
            echo "  FAIL: $file"
        done
    fi

    # Analyze which nodes have errors (get from SLURM metadata)
    echo ""
    if [ -n "$affected_files" ]; then
        node_jobs=$(echo "$affected_files" | while read file; do
            job_id=$(basename "$file" | grep -oP "slurm-\K[0-9]+")
            if [ -n "$job_id" ]; then
                # Get node list from sacct
                nodes=$(sacct -j "$job_id" --format=NodeList --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
                if [ -n "$nodes" ] && [ "$nodes" != "None" ]; then
                    echo "$nodes"
                fi
            fi
        done | sort | uniq -c)

        if [ -z "$node_jobs" ]; then
            echo "Affected nodes: None"
        else
            echo "Affected nodes:"
            echo "$node_jobs" | while read count node; do
                echo "  - $count jobs failed on node $node"
            done
        fi
    else
        echo "Affected nodes: None"
    fi

    # Analyze which GPU ranks have errors (deduplicated by job)
    if [ -n "$affected_files" ]; then
        # Get unique (file, rank) pairs, then count unique files per rank
        all_ranks=$(echo "$affected_files" | xargs -r sudo rg -l "\[[0-9]+\].*NCCL WARN Cuda|\[rank[0-9]+\].*CUDA error|\[rank[0-9]+\].*torch.AcceleratorError" 2>/dev/null | \
                    while read file; do
                        sudo rg -o "\[[0-9]+\].*NCCL WARN Cuda|\[rank[0-9]+\].*CUDA error|\[rank[0-9]+\].*torch.AcceleratorError" "$file" 2>/dev/null | \
                        grep -oP "\[[0-9]+\]|\[rank[0-9]+\]" | sed 's/\[rank//;s/\[//;s/\]//' | sort -u | \
                        while read rank; do
                            echo "$file $rank"
                        done
                    done | awk '{print $1, $2}' | sort -u | awk '{print $2}' | sort | uniq -c)

        if [ -n "$all_ranks" ]; then
            echo ""
            echo "GPU rank breakdown:"
            echo "$all_ranks" | while read count rank; do
                echo "  - $count jobs had errors on rank [$rank]"
            done
        fi
    fi

    echo ""
done


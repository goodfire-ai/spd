#!/bin/bash
# Test script for distributed training with mpirun

# Activate virtual environment
source .venv/bin/activate

echo "Testing single GPU (non-distributed)..."
python tests/manual_test_ddp.py

# Check if we have multiple GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)

if [ "$GPU_COUNT" -gt 1 ]; then
    echo -e "\nTesting with 2 GPUs using mpirun..."
    mpirun -np 2 \
        -x CUDA_VISIBLE_DEVICES=0,1 \
        python tests/manual_test_ddp.py
else
    echo -e "\nOnly 1 GPU available, skipping multi-GPU test"
fi
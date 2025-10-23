#!/bin/bash
#SBATCH --job-name=test_mpi_gpu
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=test_mpi_gpu_%j.out

# Print environment info
echo "=== Environment Info ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# MPI flags to fix binding issues
MPI_FLAGS="--bind-to none --map-by slot"

# Test basic mpirun
echo "=== Testing basic MPI ==="
mpirun $MPI_FLAGS -np 4 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK on $(hostname)"'
echo ""

# Test GPU visibility
echo "=== Testing GPU visibility ==="
mpirun $MPI_FLAGS -np 4 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
echo ""

# Test with nvidia-smi if available
echo "=== Testing nvidia-smi per rank ==="
mpirun $MPI_FLAGS -np 4 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK GPU:"; nvidia-smi -L | head -n 1'
echo ""

# Test Python + PyTorch if available
if command -v python &> /dev/null; then
    echo "=== Testing Python + PyTorch ==="
    mpirun $MPI_FLAGS -np 4 python -c '
import os
import torch
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
print(f"Rank {rank}: PyTorch sees {torch.cuda.device_count()} GPUs, using device {torch.cuda.current_device()}")
'
fi

echo "=== Test complete ==="
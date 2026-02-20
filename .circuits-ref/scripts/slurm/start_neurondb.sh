#!/bin/bash
#SBATCH --job-name=neurondb-server
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/neurondb_%j.out
#SBATCH --error=logs/neurondb_%j.err

SCRIPT_DIR=/mnt/polished-lake/home/ctigges/code/attribution-graphs/scripts

srun --container-image=/mnt/polished-lake/home/ctigges/containers/neurondb.sqsh \
     --no-container-remap-root \
     --container-mounts=$SCRIPT_DIR:$SCRIPT_DIR \
     bash $SCRIPT_DIR/neurondb_init.sh

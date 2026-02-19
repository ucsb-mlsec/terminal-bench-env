#!/bin/bash
#SBATCH --job-name=train_multi
#SBATCH --output=train_multi_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu

# Load required modules
module load cuda/11.8

# Print job information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

# Set environment variables for multi-GPU training
export MASTER_PORT=12355
export WORLD_SIZE=4

# Run multi-GPU training
echo "Starting multi-GPU training with 4 GPUs..."
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    train_parallel.py \
    --batch_size=128 \
    --epochs=100

echo "Job completed at $(date)"
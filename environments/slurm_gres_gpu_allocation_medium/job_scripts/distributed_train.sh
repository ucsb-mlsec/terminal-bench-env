#!/bin/bash
#SBATCH --job-name=distributed_train
#SBATCH --output=distributed_train_%j.log
#SBATCH --time=96:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu

# Distributed training setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Starting distributed training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Load required modules
module load cuda/11.8
module load nccl

echo "CUDA and NCCL modules loaded"
echo "Starting distributed training across nodes"

# Run distributed training
srun python distributed_training.py \
    --distributed \
    --world-size=$SLURM_NTASKS \
    --rank=$SLURM_PROCID

echo "Distributed training completed"
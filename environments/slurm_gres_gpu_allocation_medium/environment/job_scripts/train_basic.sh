#!/bin/bash
#SBATCH --job-name=train_basic
#SBATCH --output=train_basic_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load required modules
module load cuda
module load python

# Set up environment
echo "Starting basic training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Run training
python train.py --epochs 100 --batch-size 32

echo "Training completed successfully"
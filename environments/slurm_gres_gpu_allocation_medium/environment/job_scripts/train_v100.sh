#!/bin/bash
#SBATCH --job-name=train_v100
#SBATCH --output=train_v100_%j.log
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

# Load required modules
module load cuda/11.2

# Print GPU information
echo "Starting V100 training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi

# Run training script
echo "Starting training with V100 GPUs..."
python train_v100_optimized.py

# Check completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with error code $?"
fi
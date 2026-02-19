#!/bin/bash
#SBATCH --job-name=inference_a100
#SBATCH --output=inference_a100_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Print job information
echo "Starting inference job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"

# Load required modules
module load cuda/12.0

# Print GPU information
echo "GPU devices:"
nvidia-smi --list-gpus

# Run inference
echo "Starting inference with A100-optimized model..."
python run_inference.py --model a100_optimized

echo "Inference completed successfully"
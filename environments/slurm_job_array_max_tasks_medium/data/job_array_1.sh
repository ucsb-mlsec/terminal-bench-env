#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --array=0-749
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Starting job array task $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Run the analysis script with the task ID as parameter
python process.py --task-id $SLURM_ARRAY_TASK_ID --input data_${SLURM_ARRAY_TASK_ID}.txt

echo "Completed task $SLURM_ARRAY_TASK_ID"
#!/bin/bash
#SBATCH --job-name=batch_process_5241
#SBATCH --output=/workspace/logs/job_%j.out
#SBATCH --error=/workspace/logs/job_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Original job script - starts from scratch
echo "Starting batch processing job"
echo "Job ID: $SLURM_JOB_ID"

# Set working directory
WORKDIR=/workspace/data/job_5241
cd $WORKDIR

# Load configuration
source /workspace/config/app.conf

# Process all chunks from beginning
for chunk in {1..50}; do
    echo "Processing chunk $chunk"
    python3 /workspace/scripts/process_chunk.py --chunk $chunk --workdir $WORKDIR
    
    # Save checkpoint every 10 minutes
    if [ $((chunk % 5)) -eq 0 ]; then
        python3 /workspace/scripts/save_checkpoint.py --job-id 5241 --chunk $chunk
    fi
done

echo "Batch processing complete"
#!/bin/bash
#SBATCH --job-name=protein_analysis
#SBATCH --output=logs/protein_%A_%a.out
#SBATCH --error=logs/protein_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-1249

# Protein structure analysis job array
# Process 1250 protein structures in parallel

echo "Starting protein analysis task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"

# Set up environment
module load bioinformatics
module load python/3.9

# Define input and output directories
INPUT_DIR="/data/proteins/raw"
OUTPUT_DIR="/data/proteins/processed"
PROTEIN_FILE="${INPUT_DIR}/protein_${SLURM_ARRAY_TASK_ID}.pdb"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Process protein structure
echo "Analyzing protein structure: ${PROTEIN_FILE}"
python /tools/analyze_protein.py \
    --input ${PROTEIN_FILE} \
    --output ${OUTPUT_DIR}/result_${SLURM_ARRAY_TASK_ID}.csv \
    --method folding_prediction \
    --iterations 5000

# Generate visualization
echo "Creating structure visualization"
python /tools/visualize_structure.py \
    --input ${PROTEIN_FILE} \
    --output ${OUTPUT_DIR}/viz_${SLURM_ARRAY_TASK_ID}.png

# Calculate binding affinity
echo "Computing binding affinity scores"
/tools/binding_calc --struct ${PROTEIN_FILE} --out ${OUTPUT_DIR}/binding_${SLURM_ARRAY_TASK_ID}.dat

# Compress results
gzip ${OUTPUT_DIR}/result_${SLURM_ARRAY_TASK_ID}.csv

echo "Protein analysis task ${SLURM_ARRAY_TASK_ID} completed successfully"
exit 0
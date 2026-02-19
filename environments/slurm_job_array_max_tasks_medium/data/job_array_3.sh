#!/bin/bash
#SBATCH --job-name=genomics_pipeline
#SBATCH --array=0-3499
#SBATCH --output=genomics_%A_%a.out
#SBATCH --error=genomics_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=compute

# Genomics data processing pipeline
# Array task ID: $SLURM_ARRAY_TASK_ID

echo "Starting genomics analysis task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

# Define input and output directories
INPUT_DIR="/data/genomics/raw_samples"
OUTPUT_DIR="/data/genomics/processed/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${OUTPUT_DIR}

# Load required modules
module load bioinformatics/2.4
module load python/3.9

# Get sample file for this array task
SAMPLE_FILE="${INPUT_DIR}/sample_${SLURM_ARRAY_TASK_ID}.fastq"

echo "Processing sample file: ${SAMPLE_FILE}"

# Quality control step
echo "Step 1: Quality control analysis"
fastqc ${SAMPLE_FILE} -o ${OUTPUT_DIR}/qc/

# Alignment step
echo "Step 2: Sequence alignment"
bwa mem -t ${SLURM_CPUS_PER_TASK} \
    /ref/genome/hg38.fa \
    ${SAMPLE_FILE} > ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.sam

# Convert to BAM format
echo "Step 3: Converting to BAM format"
samtools view -@ ${SLURM_CPUS_PER_TASK} -bS \
    ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.sam > \
    ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.bam

# Sort BAM file
echo "Step 4: Sorting BAM file"
samtools sort -@ ${SLURM_CPUS_PER_TASK} \
    ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.bam \
    -o ${OUTPUT_DIR}/sorted_${SLURM_ARRAY_TASK_ID}.bam

# Variant calling
echo "Step 5: Variant calling"
bcftools mpileup -f /ref/genome/hg38.fa \
    ${OUTPUT_DIR}/sorted_${SLURM_ARRAY_TASK_ID}.bam | \
    bcftools call -mv -o ${OUTPUT_DIR}/variants_${SLURM_ARRAY_TASK_ID}.vcf

# Generate statistics
echo "Step 6: Generating statistics"
python3 <<EOF
import json
import time

task_id = ${SLURM_ARRAY_TASK_ID}
stats = {
    'task_id': task_id,
    'timestamp': time.time(),
    'reads_processed': 1000000 + (task_id * 1337),
    'variants_called': 50000 + (task_id * 42),
    'quality_score': 0.95 + (task_id % 100) * 0.0001
}

with open('${OUTPUT_DIR}/stats_${SLURM_ARRAY_TASK_ID}.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Statistics saved for task {task_id}")
EOF

# Cleanup intermediate files
echo "Step 7: Cleaning up intermediate files"
rm -f ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.sam
rm -f ${OUTPUT_DIR}/aligned_${SLURM_ARRAY_TASK_ID}.bam

echo "Genomics pipeline completed for task ${SLURM_ARRAY_TASK_ID}"
echo "Finished at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
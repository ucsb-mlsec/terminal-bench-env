#!/bin/bash
#$ -t 0-49
#$ -cwd
#$ -o /logs/
#$ -e /logs/

# Process 50 genome sequence files in parallel
# This script is supposed to process sample_001.fasta through sample_050.fasta

# Base directory for input files
INPUT_DIR="/data/sequences"

# Broken mapping: Using SGE_TASK_ID directly without proper zero-padding
# This creates filenames like sample_0.fasta, sample_1.fasta, ..., sample_49.fasta
# But the actual files are named sample_001.fasta through sample_050.fasta
INPUT_FILE="${INPUT_DIR}/sample_$(printf '%02d' $SGE_TASK_ID).fasta"

echo "Processing task ${SGE_TASK_ID}"
echo "Input file: ${INPUT_FILE}"

# Check if file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: File ${INPUT_FILE} does not exist!"
    exit 1
fi

# Run the processing tool (placeholder command)
some_processing_tool ${INPUT_FILE}

echo "Task ${SGE_TASK_ID} completed"
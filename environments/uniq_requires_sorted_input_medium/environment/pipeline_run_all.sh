#!/bin/bash

# Main Pipeline Orchestrator
# This script runs all stages of the analytics pipeline in sequence
# to process server access logs and generate unique visitor counts

# Create output directory if it doesn't exist
mkdir -p /opt/analytics/data/output/

# Stage 1: Extract
echo "Running Stage 1: Extract..."
bash /opt/analytics/pipeline/pipeline_stage1_extract.sh

# Stage 2: Filter
echo "Running Stage 2: Filter..."
bash /opt/analytics/pipeline/pipeline_stage2_filter.sh

# Stage 3: Deduplicate
echo "Running Stage 3: Deduplicate..."
bash /opt/analytics/pipeline/pipeline_stage3_dedupe.sh

# Stage 4: Count
echo "Running Stage 4: Count..."
bash /opt/analytics/pipeline/pipeline_stage4_count.sh

# Display final results
echo "Pipeline complete. Final unique visitor count:"
cat /opt/analytics/data/output/final_count.txt
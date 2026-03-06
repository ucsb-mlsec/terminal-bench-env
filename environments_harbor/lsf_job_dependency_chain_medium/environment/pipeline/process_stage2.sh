#!/bin/bash

echo "Starting Stage 2: Quality filtering..."

# Check if stage1 output exists
if [ ! -f "data/stage1_output.txt" ]; then
    echo "Error: Stage 1 output file not found. Stage 1 must complete first."
    exit 1
fi

# Simulate processing time
sleep 2

# Read from stage1 output and create filtered output
echo "Quality filtering results from Stage 1 data:" > data/stage2_output.txt
echo "========================================" >> data/stage2_output.txt
cat data/stage1_output.txt >> data/stage2_output.txt
echo "" >> data/stage2_output.txt
echo "Quality Metrics:" >> data/stage2_output.txt
echo "  - Read quality score: 38.5" >> data/stage2_output.txt
echo "  - GC content: 52.3%" >> data/stage2_output.txt
echo "  - Adapter contamination: 0.8%" >> data/stage2_output.txt
echo "  - Reads passing filter: 94.2%" >> data/stage2_output.txt
echo "  - Mean read length: 150bp" >> data/stage2_output.txt
echo "Quality filtering completed at $(date)" >> data/stage2_output.txt

echo "Stage 2 completed successfully"
exit 0
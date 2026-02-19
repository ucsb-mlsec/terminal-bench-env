#!/bin/bash
# Critical Data Processing Pipeline
# Processes server logs through multiple stages with intermediate outputs

# Pipeline configuration
INPUT_DIR="/var/incoming/logs"
STAGE1_OUT="/tmp/pipeline/stage1.log"
STAGE2_OUT="/var/pipeline/output/stage2.log"
FINAL_OUT="/mnt/backup/logs/final.log"

# Create output directories if they don't exist
mkdir -p "$(dirname "$STAGE1_OUT")"
mkdir -p "$(dirname "$STAGE2_OUT")"
mkdir -p "$(dirname "$FINAL_OUT")"

# Process all log files in the input directory
for logfile in "$INPUT_DIR"/*.log; do
    if [ -f "$logfile" ]; then
        echo "Processing: $logfile"
        
        # Stage 1: Extract ERROR and WARN level logs, write to stage1
        # Stage 2: Parse timestamp and message, write to stage2
        # Stage 3: Sort by severity and deduplicate, write to final output
        cat "$logfile" | \
            grep -E "(ERROR|WARN|INFO)" | \
            tee "$STAGE1_OUT" | \
            awk '{print $1, $2, $3, $NF}' | \
            tee "$STAGE2_OUT" | \
            sort -k3 | \
            uniq > "$FINAL_OUT"
        
        # Mark file as processed
        mv "$logfile" "$logfile.processed"
        
        echo "Completed: $logfile"
    fi
done

echo "Pipeline processing complete"
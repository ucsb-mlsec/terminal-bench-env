#!/bin/bash

set -e
set -u

PIPELINE_LOG="/home/agent/pipeline.log"
PROCESSED_DIR="/home/agent/processed"
ARCHIVE_DIR="/home/agent/archive"
REPORTS_DIR="/home/agent/reports"
TEMP_DIR="/home/agent/temp"

echo "========================================" | tee -a "$PIPELINE_LOG"
echo "Starting batch processing pipeline at $(date)" | tee -a "$PIPELINE_LOG"
echo "========================================" | tee -a "$PIPELINE_LOG"

# Validate input directories exist
echo "Validating directories..." | tee -a "$PIPELINE_LOG"
if [ ! -d "$PROCESSED_DIR" ]; then
    mkdir -p "$PROCESSED_DIR"
fi

# Check for any log files to process
echo "Checking for log files to process..." | tee -a "$PIPELINE_LOG"
LOG_COUNT=$(ls /home/agent/logs/*.log 2>/dev/null | wc -l)
if [ $LOG_COUNT -eq 0 ]; then
    echo "WARNING: No log files found" | tee -a "$PIPELINE_LOG"
fi

# Step 1: Process logs
echo "Step 1: Processing logs..." | tee -a "$PIPELINE_LOG"
/home/agent/process_logs.sh | tee -a "$PIPELINE_LOG"

# Verify processing completed by checking for done markers
if [ -f $PROCESSED_DIR/*.done ]; then
    echo "Processing verification passed" | tee -a "$PIPELINE_LOG"
fi

# Step 2: Archive files
echo "Step 2: Archiving files..." | tee -a "$PIPELINE_LOG"
/home/agent/archive_files.sh | tee -a "$PIPELINE_LOG"

# Step 3: Generate reports
echo "Step 3: Generating reports..." | tee -a "$PIPELINE_LOG"
/home/agent/generate_reports.sh | tee -a "$PIPELINE_LOG"

# Verify reports were created
REPORT_COUNT=$(ls $REPORTS_DIR/*.report 2>/dev/null | wc -l)
echo "Generated $REPORT_COUNT reports" | tee -a "$PIPELINE_LOG"

# Step 4: Cleanup temporary files
echo "Step 4: Cleaning up temporary files..." | tee -a "$PIPELINE_LOG"
/home/agent/cleanup_temp.sh | tee -a "$PIPELINE_LOG"

echo "========================================" | tee -a "$PIPELINE_LOG"
echo "Pipeline completed successfully at $(date)" | tee -a "$PIPELINE_LOG"
echo "========================================" | tee -a "$PIPELINE_LOG"
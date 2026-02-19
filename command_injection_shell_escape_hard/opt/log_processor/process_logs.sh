#!/bin/bash

# Main log processing script
# Processes user-submitted log files and extracts relevant information

LOG_DIR="/opt/log_processor/logs"
OUTPUT_DIR="/opt/log_processor/output"

# Function to process a single log file
process_log_file() {
    local logfile=$1
    local output_name=$2
    
    # VULNERABLE: Unquoted variable in cat command
    if [ -f $logfile ]; then
        echo "Processing log file: $logfile"
        
        # Count error entries - VULNERABLE: unquoted variable
        error_count=$(cat $logfile | grep -c "ERROR")
        warning_count=$(cat $logfile | grep -c "WARN")
        
        # Generate output filename - VULNERABLE: unquoted variable in echo and redirection
        echo "Log Analysis Report" > $OUTPUT_DIR/$output_name
        echo "===================" >> $OUTPUT_DIR/$output_name
        echo "Source: $logfile" >> $OUTPUT_DIR/$output_name
        echo "Errors: $error_count" >> $OUTPUT_DIR/$output_name
        echo "Warnings: $warning_count" >> $OUTPUT_DIR/$output_name
        
        # Extract recent errors - VULNERABLE: unquoted variable
        echo "" >> $OUTPUT_DIR/$output_name
        echo "Recent Errors:" >> $OUTPUT_DIR/$output_name
        grep "ERROR" $logfile | tail -10 >> $OUTPUT_DIR/$output_name
        
        echo "Processed: $logfile -> $OUTPUT_DIR/$output_name"
    else
        echo "Error: Log file not found: $logfile"
        return 1
    fi
}

# Main execution
if [ $# -lt 1 ]; then
    echo "Usage: $0 <logfile> [output_name]"
    exit 1
fi

LOGFILE=$1
OUTFILE=${2:-"report.txt"}

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Process the log file - VULNERABLE: unquoted variables passed to function
process_log_file $LOGFILE $OUTFILE

exit 0